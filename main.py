import os
import pathlib

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms

from config import Config
from datasets import Dataset
from logger import Logger
from losses import get_loss_criterion
from models import get_model
from utils import get_optimizer, get_scheduler, iou_score


class Model:
    def __init__(self, config=None):
        if config is None:
            config = Config()
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
        self.config = config
        self.secondloss = False
        self.thirdloss = False

    def setup_dataset(self, dataset_path):
        # Data loading code
        config = self.config
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

    def train(self):
        config = self.config
        best_iou = 0
        trigger = 0
        for epoch in range(config.epochs):
            # train for one epoch
            self.train_epoch(self.train_loader)
            # evaluate on validation set
            self.validate_epoch(self.val_loader)
            # test an image and save
            self.test(epoch)
            self.test(
                epoch,
                "../data/segmentation/images/122_image_201023_002.JPG",
                True)
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

    def test(self, epoch, image_path=None, is_in=False):
        def normalize(image):
            # Calculate from whole lithofaces data
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[
                            0.280313914506407,
                            0.41555059997248583,
                            0.3112942716287795,
                        ],
                        std=[
                            0.16130980680117304,
                            0.19598465956271507,
                            0.14531163979659875,
                        ],
                    ),
                ]
            )(image)

        if image_path is None:
            image_path = self.config.test_image
        image = Image.open(image_path)
        image = normalize(image).unsqueeze(0).to(self.config.device)
        # compute output
        self.model.eval()

        with torch.no_grad():
            if self.config.deep_supervision:
                output = self.model(image)[-1]
            else:
                output = self.model(image)
            output = torch.sigmoid(output).cpu()
        if is_in:
            plt.imsave(
                f"./networks/{self.config.name}/T{epoch}.png",
                torch.argmax(output[0], 0),
            )
        else:
            plt.imsave(
                f"./networks/{self.config.name}/{epoch}.png",
                torch.argmax(
                    output[0],
                    0))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    config = Config()
    if "KAGGLE_CONTAINER_NAME" in os.environ:
        config.path = "/kaggle/input/lithofaces-dataset/lithofaces.h5"
        config.batch_size = 32
        config.num_workers = 2
    else:
        config.path = "/home/lao/Data/lithofaces.h5"
        config.batch_size = 20
        config.num_workers = 12
    config.model = "NestedUNet"
    config.test_image = "../data/segmentation/images/116_image_130909_041.JPG"
    config.loss = "BCEPixelWiseDiceLoss"
    config.loss_alpha = 1.0
    config.loss_beta = 1.0
    config.loss_gamme = 250.0
    config.ignore_labels = ["C3A"]
    config.epochs = 200
    # weight should be none when use diceloss
    config.weight = None
    # config.learning_rate = 0.01
    Config.check_classes(config)
    model = Model(config=config)
    print("=>Setting Dataset.")
    model.setup_dataset(config.path)
    print("=>Training.")
    Logger.config = config
    Logger.init(
        model.train_loader,
        model.val_loader,
        epochs=config.epochs,
        progress=True)
    model.train()
