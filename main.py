import os
import pathlib

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms

from config import Config
from datasets import Dataset,normalize
from logger import Logger
from losses import get_loss_criterion
from models import get_model
from utils import get_optimizer, get_scheduler, iou_score


class Model:
    def __init__(self, config=None):
        if config is None:
            config = Config()
        self.config = config
    def setup(self):
        config = self.config
        self.root = pathlib.Path(config.path)
        if config.name is None:
            name = f"{config.dataset}_{config.model}"
            config.name = name + ("_wDS"
                                  if config.deep_supervision else "_woDS")
        os.makedirs(f"networks/{config.name}", exist_ok=True)
        config.save(f"networks/{config.name}/config.yml")
        # Define loss function(criterion)
        self.criterion = get_loss_criterion(config).cuda()
        cudnn.benchmark = True
        # Create model
        print(f"=>Creating model {config.model}")
        model = get_model(config)
        self.model = model.cuda()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = get_optimizer(config, params)
        self.scheduler = get_scheduler(config, self.optimizer)
        dataset_path = config.path
        # Data loading
        train_dataset = Dataset(root=self.root, mode="train", config=config)
        val_dataset = Dataset(root=self.root, mode="val", config=config)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )

    def compute_output(self, input, target, weight_map):
        # compute output
        if self.config.deep_supervision:
            outputs = self.model(input)
            loss = 0
            for output in outputs:
                loss += self.criterion(output, target, weight_map)
            loss /= len(outputs)
            iou = iou_score(
                outputs[-1],
                target, self.config.labels, self.config.ignore_labels)
        else:
            output = self.model(input)
            loss = self.criterion(output, target, weight_map)
            # print(loss)
            iou = iou_score(
                output, target, self.config.labels, self.config.ignore_labels
            )
        return loss, iou

    @Logger(name="train")
    def train_iter(self, input, target, weight_map):
        # compute output
        loss, iou = self.compute_output(input, target, weight_map)
        # compute gradient and do optimizing step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), iou

    @Logger(name="val")
    def val_iter(self, input, target, weight_map):
        # compute output
        loss, iou = self.compute_output(input, target, weight_map)
        return loss.item(), iou

    def train_epoch(self, train_loader):
        # switch to train mode
        self.model.train()
        for input, target, weight_map, _ in train_loader:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            self.train_iter(input, target, weight_map)

    def validate_epoch(self, val_loader):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            for input, target, weight_map, _ in val_loader:
                input = input.cuda()
                target = target.cuda()
                self.val_iter(input, target, weight_map)
        self.model.train()

    def train(self,predict=False):
        config = self.config
        best_iou = 0
        trigger = 0
        for epoch in range(config.epochs):
            # train for one epoch
            self.train_epoch(self.train_loader)
            # evaluate on validation set
            self.validate_epoch(self.val_loader)
            # test images and save
            if predict:
                self.test(epoch)
            if config.scheduler == "CosineAnnealingLR":
                self.scheduler.step()
                config.learning_rate_current = self.scheduler.get_last_lr()[0]
            elif config.scheduler == "ReduceLROnPlateau":
                self.scheduler.step(Logger.data["val"]["loss"])
                config.learning_rate_current = self.scheduler.get_last_lr()[0]
            trigger += 1

            if Logger.data["val"]["iou"] > best_iou:
                torch.save(
                    self.model.state_dict(),
                    "networks/%s/model.pth" %
                    config.name)
                best_iou = Logger.data["val"]["iou"]
                # pbar.display("=> saved best model")
                trigger = 0

            # early stopping
            if config.early_stopping >= 0 and trigger >= config.early_stopping:
                print("=> early stopping")
                break
            torch.cuda.empty_cache()

    def predict(self, epoch, image_paths=None, is_in=False):
        self.model.eval()

        with torch.no_grad():
            if image_paths is None:
                image_paths = self.config.test_images
            for idx, image_path in enumerate(image_paths):
                image = Image.open(image_path)
                image = normalize(image).unsqueeze(0).to(self.config.device)
                if self.config.deep_supervision:
                    output = self.model(image)[-1]
                else:
                    output = self.model(image)
                output = torch.sigmoid(output).cpu()
                plt.imsave(
                    f"./networks/{self.config.name}/{idx}-{epoch}.png",
                    torch.argmax(output[0], 0),
                )
                del image
        self.model.train()
    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        tensorboard_formatter=None, skip_train_validation=False):
        state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   tensorboard_formatter=tensorboard_formatter,
                   skip_train_validation=skip_train_validation)
    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=logger)
if __name__ == "__main__":
    config = Config()
    if "KAGGLE_CONTAINER_NAME" in os.environ:
        config.path = "/kaggle/input/lithofaces-dataset/lithofaces.h5"
        config.batch_size = 32
        config.num_workers = 2
        test_path = "/kaggle/input/lithofaces-test-image/"
        config.test_images = [
            test_path + "0ce6b3901bcd961e6e8d4911b60b4547.JPG",
            test_path + "2010.136.1_200X130909_011.JPG"]
    else:
        config.path = "/home/lao/Data/lithofaces.h5"
        config.batch_size = 64
        config.num_workers = 12
        test_path = "../data/segmentation/images/"
        config.test_images = [
            test_path + "116_image_130909_041.JPG",
            test_path + "122_image_201023_002.JPG"]
    config.model = "UNet"
    config.loss = "BCEPixelWiseDiceLoss"
    config.deep_supervision = False
    config.loss_alpha = 1.0
    config.loss_beta = 1.0
    config.loss_gamme = 250.0
    config.ignore_labels = ["C3A"]
    config.epochs = 2
    # weight should be none when use diceloss
    config.weight = None
    # config.learning_rate = 0.01
    Config.check_classes(config)
    model = Model(config=config)
    print("=>Setting Dataset.")
    model.setup()
    print("=>Training.")
    Logger.init(
        model.train_loader,
        model.val_loader,
        progress=True,config=config)
    model.train()
    Logger.close()