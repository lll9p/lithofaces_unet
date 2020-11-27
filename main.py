import os
import pathlib
from collections import OrderedDict

import cv2
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

from config import Config
from dataset import Dataset
from losses import get_loss_criterion
from models import get_model
from utils import AverageMeter, get_optimizer, get_scheduler, iou_score


class Model:
    def __init__(self, config=None, model="UNet_3Plus_DeepSup"):
        if config is None:
            config = Config()
        self.root = pathlib.Path(config.path)
        if config.name is None:
            name = f"{config.dataset}_{config.model}"
            config.name = name + ("_wDS"
                                  if config.deep_supervision else "_woDS")
        os.makedirs(f"networks/{config.name}", exist_ok=True)
        print("-" * 20)
        for key in config.__dict__:
            print(f"{key}: {config.__dict__[key]}")
        print("-" * 20)
        config.save(f"networks/{config.name}/config.yml")
        # Defile loss function(criterion)
        self.criterion = get_loss_criterion(config).cuda()

        cudnn.benchmark = True

        # Create model
        print(f"=>Ccreating model {config.model}")
        model = get_model(config)
        self.model = model.cuda()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.config = config
        self.secondloss = False
        self.thirdloss = False
        self.optimizer = get_optimizer(config, params)
        self.scheduler = get_scheduler(config, self.optimizer)

    def setup_dataset(self, dataset_path):
        # Data loading code
        config = self.config
        # datasets = get_datasets(file_path=dataset_path, modes=["train", "val"])
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

    def train_epoch(self, train_loader, pbar):
        # def train(config, train_loader, model, criterion, optimizer):
        config = self.config
        avg_meters = {"loss": AverageMeter(), "iou": AverageMeter()}
        # Set to train mode
        self.model.train()

        # pbar = tqdm(total=len(train_loader), position=1, leave=True)
        pbar.total = len(train_loader)
        for input, target, weight_map, _ in train_loader:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            # output shape is[batch_size, nb_classes, height, width]
            # (16, 1, 256, 256)
            # https://discuss.pytorch.org/t/weighted-pixelwise-nllloss2d/7766/6
            if self.config.deep_supervision:
                outputs = self.model(input)
                loss = 0
                for output in outputs:
                    loss += self.criterion(output, target, weight_map)
                loss /= len(outputs)
                iou = iou_score(
                    outputs[-1], target, config.labels, config.ignore_labels
                )
            else:
                output = self.model(input)
                loss = self.criterion(output, target, weight_map)
                # print(loss)
                iou = iou_score(
                    output,
                    target,
                    config.labels,
                    config.ignore_labels)

            # compute gradient and do optimizing step
            # print(output.shape)
            # print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_meters["loss"].update(loss.item(), input.size(0))
            avg_meters["iou"].update(iou, input.size(0))

            postfix = OrderedDict(
                [("loss", avg_meters["loss"].avg),
                 ("iou", avg_meters["iou"].avg), ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        # pbar.close()

        return OrderedDict(
            [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)]
        )

    def validate_epoch(self, val_loader, pbar):
        config = self.config
        avg_meters = {"loss": AverageMeter(), "iou": AverageMeter()}

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            # pbar = tqdm(total=len(val_loader), position=2, leave=True)
            pbar.total = len(val_loader)
            for input, target, weight_map, _ in val_loader:
                input = input.cuda()
                target = target.cuda()

                # compute output
                if config.deep_supervision:
                    outputs = self.model(input)
                    loss = 0
                    for output in outputs:
                        loss += self.criterion(output, target, weight_map)
                    loss /= len(outputs)
                    iou = iou_score(
                        outputs[-1], target, config.labels, config.ignore_labels
                    )
                else:
                    output = self.model(input)
                    loss = self.criterion(output, target, weight_map)
                    iou = iou_score(
                        output, target, config.labels, config.ignore_labels)

                avg_meters["loss"].update(loss.item(), input.size(0))
                avg_meters["iou"].update(iou, input.size(0))

                postfix = OrderedDict(
                    [("loss", avg_meters["loss"].avg),
                     ("iou", avg_meters["iou"].avg), ])
                pbar.set_postfix(postfix)
                pbar.update(1)
            # pbar.close()

        return OrderedDict(
            [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)]
        )

    def train(self):
        config = self.config
        log = OrderedDict(
            [
                ("epoch", []),
                ("lr", []),
                ("loss", []),
                ("iou", []),
                ("val_loss", []),
                ("val_iou", []),
            ]
        )

        best_iou = 0
        trigger = 0
        pbar = tqdm(
            desc="Trainning",
            total=config.epochs,
            position=0,
            leave=True)
        pbar_train = tqdm(desc="Train epoch", position=1, leave=True)
        pbar_val = tqdm(desc="Validate epoch", position=1, leave=True)
        for epoch in range(config.epochs):
            pbar_train.reset()
            pbar_val.reset()
            if epoch <= 2:
                pbar.display(f"Epoch [{epoch}/{config.epochs}]")
            else:
                pbar.display(
                    f"Epoch [{epoch}/{config.epochs}]"
                    + f'Loss {log["loss"][-1]:.4f} - '
                    + f'IOU {log["iou"][-1]:.4f} - '
                    + f'val_Loss {log["val_loss"][-1]:.4f} - '
                    + f'val_IOU {log["val_iou"][-1]:.4f}'
                )
            if epoch > 40 and self.secondloss == False:
                self.config.loss = "DiceLoss"
                self.criterion = get_loss_criterion(config).cuda()
                self.secondloss = True
            if epoch > 75 and self.thirdloss == False:
                self.config.loss = "PixelWiseDiceLoss"
                self.criterion = get_loss_criterion(config).cuda()
                self.thirdloss = True
            # train for one epoch
            train_log = self.train_epoch(self.train_loader, pbar_train)
            # evaluate on validation set
            val_log = self.validate_epoch(self.val_loader, pbar_val)

            if config.scheduler == "CosineAnnealingLR":
                self.scheduler.step()
            elif config.scheduler == "ReduceLROnPlateau":
                self.scheduler.step(val_log["loss"])

            log["epoch"].append(epoch)
            log["lr"].append(config.learning_rate)
            log["loss"].append(train_log["loss"])
            log["iou"].append(train_log["iou"])
            log["val_loss"].append(val_log["loss"])
            log["val_iou"].append(val_log["iou"])

            pd.DataFrame(log).to_csv(
                f"networks/{config.name}/log.csv", index=False)

            trigger += 1

            if val_log["iou"] > best_iou:
                torch.save(
                    self.model.state_dict(),
                    "networks/%s/model.pth" %
                    config.name)
                best_iou = val_log["iou"]
                # pbar.display("=> saved best model")
                trigger = 0

            # early stopping
            if config.early_stopping >= 0 and trigger >= config.early_stopping:
                pbar.display("=> early stopping")
                break
            pbar.update(1)
            torch.cuda.empty_cache()
        pbar.close()
        pbar_train.close()
        pbar_train.close()

    def test(self):
        config = self.config
        with open(f"networks/{config.name}/config.yml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        print("-" * 20)
        for key in config.__dict__:
            print("%s: %s" % (key, str(config.__dict__[key])))
        print("-" * 20)

        cudnn.benchmark = True

        # create model
        print(f"=> creating model {config.model}")
        model = models.__dict__[config.model](
            config.num_classes, config.input_channels, config.deep_supervision
        )

        model = model.cuda()

        model.load_state_dict(torch.load(f"networks/{config.name}/model.pth"))
        model.eval()

        avg_meter = AverageMeter()

        for c in range(config.num_classes):
            os.makedirs(
                os.path.join(
                    "outputs",
                    config.name,
                    str(c)),
                exist_ok=True)
        with torch.no_grad():
            for input, target, weight_map, meta in tqdm(
                self.val_loader, total=len(self.val_loader)
            ):
                input = input.cuda()
                target = target.cuda()

                # compute output
                if config.deep_supervision:
                    output = model(input)[-1]
                else:
                    output = model(input)

                iou = iou_score(output, target)
                avg_meter.update(iou, input.size(0))

                output = torch.sigmoid(output).cpu().numpy()

                for i in range(len(output)):
                    for c in range(config.num_classes):
                        cv2.imwrite(
                            os.path.join(
                                "outputs", config.name, str(c), meta[i] + ".jpg"
                            ),
                            (output[i, c] * 255).astype("uint8"),
                        )

        print("IoU: %.4f" % avg_meter.avg)

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
        config.num_workers = 8
    config.loss = "BCEDiceLoss"
    config.ignore_labels=["C3A"]
    config.epochs = 100
    config.weight = None
    Config.check_classes(config)
    model = Model(config=config)
    model.setup_dataset(config.path)
    model.train()
