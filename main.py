import os
import pathlib

import h5py
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

from config import Config
from datasets import Dataset, normalize
from logger import Logger
from losses import get_loss_criterion
from models import get_model
from utils import get_optimizer, get_scheduler, iou_score, iou_pytorch


class Model:
    def __init__(self, config=None):
        if config is None:
            config = Config()
        self.config = config

    def setup(self):
        # reset Dataset
        Dataset.full_idx = None
        Dataset.dataset = None
        config = self.config
        self.root = pathlib.Path(config.dataset.path)
        if config.name is None:
            name = f"{config.dataset.name}_{config.model.name}"
            config.name = name + ("_wDS"
                                  if config.model.deep_supervision else "_woDS")
        os.makedirs(f"networks/{config.name}", exist_ok=True)
        config.save(f"networks/{config.name}/config.yml")
        # Define loss function(criterion)
        self.criterion = get_loss_criterion(config).to(self.config.device)
        if self.config.device=="cuda":
            cudnn.benchmark = True
        # Create model
        print(f"=>Creating model {config.model.name}")
        model = get_model(config)
        self.model = model.to(self.config.device)
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = get_optimizer(config, params)
        self.scheduler = get_scheduler(config, self.optimizer)
        # Data loading
        with h5py.File(self.config.dataset.path, mode="r") as file:
            Dataset.full_idx = file["idx"][:].tolist()
        idx_head = list(set(i.split("-")[0] for i in Dataset.full_idx))
        # 按head分train val
        # shuffle train val idx
        train_idx_head, val_idx_head = train_test_split(
            idx_head, test_size=0.2, random_state=42, shuffle=True)
        train_idx = []
        val_idx = []
        for idx in Dataset.full_idx:
            if idx.split("-")[0] in train_idx_head:
                train_idx.append(idx)
            else:
                val_idx.append(idx)
        # np.random.shuffle(train_idx)
        # np.random.shuffle(val_idx)
        train_dataset = Dataset(
            root=self.root,
            mode="train",
            config=config,
            idx=train_idx)
        val_dataset = Dataset(
            root=self.root,
            mode="val",
            config=config,
            idx=val_idx)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            drop_last=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            drop_last=False,
        )

    def compute_output(self, input, target, target2=None):
        # compute output
        if self.config.train.on == "distance":
            shape_distance, neighbor_distance = self.model(input)
            shape_distance = shape_distance.squeeze()
            neighbor_distance = neighbor_distance.squeeze()
            loss = self.criterion(target,target2,shape_distance, neighbor_distance)
            iou_border = iou_pytorch(
                target > torch.tensor(
                    [0.5],
                    requires_grad=False).to(self.config.device),
                shape_distance,device=self.config.device)
            iou_cells = iou_pytorch(
                target2 > torch.tensor(
                    [0.5],
                    requires_grad=False).to(self.config.device),
                neighbor_distance,device=self.config.device
            )
            iou = (iou_border + iou_cells) / 2
        else:
            if self.config.model.deep_supervision:
                outputs = self.model(input)
                loss = 0
                for output in outputs:
                    loss += self.criterion(output, target, target2)
                loss /= len(outputs)
                iou = iou_score(
                    outputs[-1],
                    target, self.config.dataset.labels, self.config.dataset.ignore_labels)
            else:
                output = self.model(input)
                loss = self.criterion(output, target, target2)
                # print(loss)
                iou = iou_score(
                    output,
                    target,
                    self.config.dataset.labels,
                    self.config.dataset.ignore_labels)
        return loss, iou

    @Logger(name="train")
    def train_iter(self, input, target, target2):
        # compute output
        loss, iou = self.compute_output(input, target, target2)
        # compute gradient and do optimizing step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), iou

    @Logger(name="val")
    def val_iter(self, input, target, target2):
        # compute output
        loss, iou = self.compute_output(input, target, target2)
        return loss.item(), iou

    def train_epoch(self, train_loader):
        # switch to train mode
        device=self.config.device
        self.model.train()
        for data in train_loader:
            if self.config.train.on == "masks" or self.config.train.on == "edges":
                input, target, _ = data
                input = input.to(device,non_blocking=True)
                target = target.to(device,non_blocking=True)
                target2 = None
            elif self.config.train.on == "distance":
                input, target, target2, _ = data
                input = input.to(device,non_blocking=True)
                target = target.float().to(device,non_blocking=True)
                target2 = target2.float().to(device,non_blocking=True)
            self.train_iter(input, target, target2)

    def validate_epoch(self, val_loader):
        # switch to evaluate mode
        device=self.config.device
        self.model.eval()
        with torch.no_grad():
            for data in val_loader:
                if self.config.train.on == "masks" or self.config.train.on == "edges":
                    input, target, _ = data
                    input = input.to(device,non_blocking=True)
                    target = target.to(device,non_blocking=True)
                    target2 = None
                elif self.config.train.on == "distance":
                    input, target, target2, _ = data
                    input = input.to(device,non_blocking=True)
                    target = target.float().to(device,non_blocking=True)
                    target2 = target2.float().to(device,non_blocking=True)
                self.val_iter(input, target, target2)
        self.model.train()

    def train(self, predict=False):
        config = self.config
        best_iou = 0
        trigger = 0
        for epoch in range(config.train.epochs):
            # train for one epoch
            self.train_epoch(self.train_loader)
            # evaluate on validation set
            self.validate_epoch(self.val_loader)
            # test images and save
            if predict:
                self.predict(epoch)
            if config.schedule.name == "CosineAnnealingLR":
                self.scheduler.step()
                config.schedule.learning_rate_current = self.scheduler.get_last_lr()[0]
            elif config.schedule.name == "ReduceLROnPlateau":
                self.scheduler.step(Logger.data["val"]["loss"])
                config.schedule.learning_rate_current = self.scheduler.get_last_lr()[0]
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
            if config.train.early_stopping >= 0 and trigger >= config.train.early_stopping:
                print("=> early stopping")
                break
            if self.config.device=="cuda":
                torch.cuda.empty_cache()

    def predict(self, epoch, image_paths=None, is_in=False):
        self.model.eval()

        with torch.no_grad():
            if image_paths is None:
                image_paths = self.config.test.images
            for idx, image_path in enumerate(image_paths):
                image = Image.open(image_path)
                image = normalize(image).unsqueeze(0).to(self.config.device)
                if self.config.train.on == "distance":
                    shape_distance, neighbor_distance = self.model(image)
                    shape_distance = shape_distance.squeeze().cpu()
                    neighbor_distance = neighbor_distance.squeeze().cpu()
                    plt.imsave(
                        f"./networks/{self.config.name}/{idx}-{epoch}-shape.png",
                        shape_distance,
                    )
                    plt.imsave(
                        f"./networks/{self.config.name}/{idx}-{epoch}-neighbor.png",
                        neighbor_distance,
                    )
                else:
                    if self.config.model.deep_supervision:
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
    # @classmethod
    # def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
    #                     tensorboard_formatter=None, skip_train_validation=False):
    #     state = utils.load_checkpoint(checkpoint_path, model, optimizer)
    #     checkpoint_dir = os.path.split(checkpoint_path)[0]
    #     return cls(model, optimizer, lr_scheduler,
    #                loss_criterion, eval_criterion,
    #                torch.device(state['device']),
    #                loaders, checkpoint_dir,
    #                eval_score_higher_is_better=state['eval_score_higher_is_better'],
    #                best_eval_score=state['best_eval_score'],
    #                num_iterations=state['num_iterations'],
    #                num_epoch=state['epoch'],
    #                max_num_epochs=state['max_num_epochs'],
    #                max_num_iterations=state['max_num_iterations'],
    #                validate_after_iters=state['validate_after_iters'],
    #                log_after_iters=state['log_after_iters'],
    #                validate_iters=state['validate_iters'],
    #                tensorboard_formatter=tensorboard_formatter,
    #                skip_train_validation=skip_train_validation)
    # def _save_checkpoint(self, is_best):
    #     # remove `module` prefix from layer names when using `nn.DataParallel`
    #     # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
    #     if isinstance(self.model, nn.DataParallel):
    #         state_dict = self.model.module.state_dict()
    #     else:

    #         state_dict = self.model.state_dict()

    #     utils.save_checkpoint({
    #         'epoch': self.num_epoch + 1,
    #         'num_iterations': self.num_iterations,
    #         'model_state_dict': state_dict,
    #         'best_eval_score': self.best_eval_score,
    #         'eval_score_higher_is_better': self.eval_score_higher_is_better,
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'device': str(self.device),
    #         'max_num_epochs': self.max_num_epochs,
    #         'max_num_iterations': self.max_num_iterations,
    #         'validate_after_iters': self.validate_after_iters,
    #         'log_after_iters': self.log_after_iters,
    #         'validate_iters': self.validate_iters
    #     }, is_best, checkpoint_dir=self.checkpoint_dir,
    #         logger=logger)


if __name__ == "__main__":
    config = Config()
    if "KAGGLE_CONTAINER_NAME" in os.environ:
        config.dataset.path = "/kaggle/input/lithofaces-dataset/lithofaces.h5"
        config.train.batch_size = 8
        config.dataset.num_workers = 2
        test_path = "/kaggle/input/lithofaces-test-image/"
        config.test.images = [
            test_path + "0ce6b3901bcd961e6e8d4911b60b4547.JPG",
            test_path + "2010.136.1_200X130909_011.JPG"]
    else:
        config.dataset.path = "/home/lao/Data/lithofaces.h5"
        config.train.batch_size = 16
        config.dataset.num_workers = 12
        test_path = "../data/segmentation/images/"
        config.test.images = [
            test_path + "116_image_130909_041.JPG",
            test_path + "122_image_201023_002.JPG"]
    config.model.name = "DUNet"
    config.model.deep_supervision = False
    config.loss.name = "DistanceLoss"
    config.train.on = "distance"
    # train on distance
    config.loss.alpha = 1.0
    # config.loss_alpha = 0.0075
    # config.loss_alpha = 1.0
    config.loss.beta = 1.0
    config.loss.gamma = 250.0
    config.device='cuda'
    # if train on edges ignore all labels
    # config.ignore_labels = ["Alite", "Blite", "C3A", "Pore"]
    config.dataset.ignore_labels = ["C3A"]
    config.train.epochs = 200
    # weight should be none when use diceloss
    config.loss.weight = None
    # config.learning_rate = 0.01
    Config.check_classes(config)
    model = Model(config=config)
    print("=>Setting Dataset.")
    model.setup()
    print("=>Training.")
    Logger.init(
        model.train_loader,
        model.val_loader,
        progress=True, config=config)
    model.train(predict=False)
    Logger.close()
