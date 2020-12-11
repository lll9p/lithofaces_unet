from functools import partial, wraps
import tqdm.autonotebook as tqdm
import csv
from collections import OrderedDict


class Logger:
    epoch_data = dict()
    data = dict()
    config = None
    file = None
    epoch = 0
    learning_rate = 0.001
    iter_nums = dict()
    progress = True
    bars = dict()

    def __init__(self, name=None):
        # if self.config is None:
        # raise "Please add config for logger!"
        Logger.epoch_data[name] = dict(loss=[], iou=[])
        Logger.data[name] = dict(loss=0.0, iou=0.0)
        self.name = name

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            loss, iou = func(*args, **kwargs)
            Logger.epoch_data[self.name]["loss"].append(loss)
            Logger.epoch_data[self.name]["iou"].append(iou)
            # 求平均
            Logger.data[self.name]["loss"] = self.average("loss")
            Logger.data[self.name]["iou"] = self.average("iou")
            if Logger.progress:
                # update bar at every iter
                postfix_str = (
                    f'L:{Logger.data[self.name]["loss"]:.3f}/'
                    f'I:{Logger.data[self.name]["iou"]:.3f}'
                )
                Logger.bars[self.name].set_postfix_str(postfix_str)
                Logger.bars[self.name].update(1)
                # Logger.bars["epoch"].display(f'{Logger.bars["train"].n},{Logger.bars["val"].n},{Logger.iter_nums["train"]}')
                # if iter over epoch then update epoch bar
                if (
                    Logger.bars["train"].n+Logger.bars["val"].n
                    >= Logger.iter_nums["train"] + Logger.iter_nums["val"]
                ):
                    # trigger save
                    if Logger.config is not None:
                        Logger.learning_rate = Logger.config.learning_rate_current
                    self.save()
                    Logger.epoch = Logger.bars["epoch"].n
                    Logger.bars["train"].reset()
                    Logger.bars["val"].reset()
                    postfix_str = (
                        f'L:{Logger.data["train"]["loss"]:.3f}/'
                        f'I:{Logger.data["train"]["iou"]:.3f}/'
                        f'VL:{Logger.data["val"]["loss"]:.3f}/'
                        f'VI:{Logger.data["val"]["iou"]:.3f}'
                    )
                    Logger.bars["epoch"].set_postfix_str(postfix_str)
                    Logger.bars["epoch"].update(1)
                    # clear tmp data and state
                    Logger.epoch_data[self.name] = dict(loss=[], iou=[])
            return loss, iou

        return wrapped_function

    def __get__(self, obj, cls):
        return partial(self.__call__, obj)

    def average(self, param):
        _sum = sum(Logger.epoch_data[self.name][param])
        _count = len(Logger.epoch_data[self.name][param])
        return _sum / _count

    def save(self):
        if Logger.config is None:
            return
        if Logger.file is None:
            Logger.file = f"networks/{Logger.config.name}/log.csv"
            with open(Logger.file, "w") as file:
                file.write("epoch,learning_rate,loss,iou,val_loss,val_iou\n")
        data = (
            Logger.epoch,
            Logger.learning_rate,
            Logger.data["train"]["loss"],
            Logger.data["train"]["iou"],
            Logger.data["val"]["loss"],
            Logger.data["val"]["iou"],
        )
        with open(Logger.file, "a") as file:
            file.write(",".join(map(str, data)) + "\n")

    @staticmethod
    def init(train_loader, val_loader, epochs, progress=True):
        Logger.progress = progress
        Logger.iter_nums["epoch"] = epochs
        Logger.iter_nums["train"] = len(train_loader)
        Logger.iter_nums["val"] = len(val_loader)
        if progress:
            bar_format = (
                "{desc}:{postfix} {percentage:3.0f}% "
                "{n_fmt}/{total_fmt} {elapsed}<{remaining},{rate_fmt}"
            )
            Logger.progress = progress
            # creating epoch bar
            epoch_bar = tqdm.tqdm(
                desc="Epoch",
                total=Logger.iter_nums["epoch"],
                position=0,
                leave=True,
                dynamic_ncols=True,
                bar_format=bar_format,
            )
            # creating train bar
            train_bar = tqdm.tqdm(
                desc="Train",
                total=Logger.iter_nums["train"],
                position=1,
                leave=True,
                dynamic_ncols=True,
                bar_format=bar_format,
            )
            # creating val bar
            val_bar = tqdm.tqdm(
                desc="Val  ",
                total=Logger.iter_nums["val"],
                position=2,
                leave=True,
                dynamic_ncols=True,
                bar_format=bar_format,
            )
            Logger.bars["epoch"] = epoch_bar
            Logger.bars["train"] = train_bar
            Logger.bars["val"] = val_bar

    @staticmethod
    def get(name, param):
        data = Logger.data[name][param]
        return data
