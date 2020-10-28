import os
from collections import OrderedDict

import cv2
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm.autonotebook import tqdm

import losses
import models
from dataset import Dataset, get_datasets
from utils import AverageMeter, iou_score, parse_args, str2bool
import pathlib


class Model():
    def __init__(self, config=None, model='UNet_3Plus_DeepSup'):
        if config is None:
            config = parse_args().parse_args()
        self.root = pathlib.Path(config.path)
        if config.name is None:
            name = f'{config.dataset}_{config.model}'
            config.name = name + \
                ('_wDS' if config.deep_supervision else '_woDS')
        os.makedirs(f'models/{config.name}', exist_ok=True)
        print('-' * 20)
        for key in config.__dict__:
            print(f'{key}: {config.__dict__[key]}')
        print('-' * 20)
        with open('models/%s/config.yml' % config.name, 'w') as f:
            yaml.dump(config, f)
        # Defile loss function(criterion)
        self.criterion = losses.__dict__[config.loss]().cuda()

        cudnn.benchmark = True

        # Create model
        print("=>Ccreating model {config.model}")
        (num_classes,
         input_channels,
         deep_supervision) = (config.num_classes,
                              config.input_channels,
                              config.deep_supervision)
        model = models.__dict__[config.model](num_classes,
                                              input_channels,
                                              deep_supervision)
        self.model = model.cuda()
        params = filter(lambda p: p.requires_grad,
                        self.model.parameters())
        if config.optimizer == 'Adam':
            optimizer = optim.Adam(
                params, lr=config.learning_rate,
                weight_decay=config.weight_decay)
        elif config.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=config.learning_rate,
                                  momentum=config.momentum,
                                  nesterov=config.nesterov,
                                  weight_decay=config.weight_decay)
        else:
            raise NotImplementedError
        if config.scheduler == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs,
                eta_min=config.min_learning_rate)
        elif config.scheduler == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=config.factor,
                                                       patience=config.patience,
                                                       verbose=1,
                                                       min_lr=config.min_learning_rate)
        elif config.scheduler == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[
                                                     int(e) for e in config.milestones.split(',')],
                                                 gamma=config.gamma)
        elif config.scheduler == 'ConstantLR':
            scheduler = None
        else:
            raise NotImplementedError

        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler

    def setup_dataset(self):
        # Data loading code
        config = self.config
        datasets = get_datasets(path=self.root)
        train_dataset = Dataset(
            datasets, root=self.root, mode='train')
        val_dataset = Dataset(
            datasets, root=self.root, mode='val')
        test_dataset = Dataset(
            datasets, root=self.root, mode='test')

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False)

    def train_epoch(self, train_loader, pbar):
        # def train(config, train_loader, model, criterion, optimizer):
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter()}
        # Set to train mode
        self.model.train()

        #pbar = tqdm(total=len(train_loader), position=1, leave=True)
        pbar.total = len(train_loader)
        for input, target, _ in train_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            # output shape is[batch_size, nb_classes, height, width]
            # (16, 1, 256, 256)
            # https://discuss.pytorch.org/t/weighted-pixelwise-nllloss2d/7766/6
            if self.config.deep_supervision:
                outputs = self.model(input)
                loss = 0
                for output in outputs:
                    if self.config.loss == "MSSSIM":
                        loss -= self.criterion(output, target)
                    else:
                        loss += self.criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = self.model(input)
                if self.config.loss == "MSSSIM":
                    loss = -self.criterion(output, target)
                else:
                    loss = self.criterion(output, target)
                # print(loss)
                iou = iou_score(output, target)

            # compute gradient and do optimizing step
            # print(output.shape)
            # print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        # pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg)])

    def validate_epoch(self, val_loader, pbar):
        config = self.config
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter()}

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            #pbar = tqdm(total=len(val_loader), position=2, leave=True)
            pbar.total = len(val_loader)
            for input, target, _ in val_loader:
                input = input.cuda()
                target = target.cuda()

                # compute output
                if config.deep_supervision:
                    outputs = self.model(input)
                    loss = 0
                    for output in outputs:
                        if config.loss == "MSSSIM":
                            loss -= self.criterion(output, target)
                        else:
                            loss += self.criterion(output, target)
                    loss /= len(outputs)
                    iou = iou_score(outputs[-1], target)
                else:
                    output = self.model(input)
                    if config.loss == "MSSSIM":
                        loss = -self.criterion(output, target)
                    else:
                        loss = self.criterion(output, target)
                    iou = iou_score(output, target)

                avg_meters['loss'].update(loss.item(), input.size(0))
                avg_meters['iou'].update(iou, input.size(0))

                postfix = OrderedDict([
                    ('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)
            # pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg)])

    def train(self):
        config = self.config
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('iou', []),
            ('val_loss', []),
            ('val_iou', []),
        ])

        best_iou = 0
        trigger = 0
        pbar = tqdm(desc="Trainning",
                    total=config.epochs, position=0, leave=True)
        pbar_train = tqdm(desc="Train epoch", position=1, leave=True)
        pbar_val = tqdm(desc="Validate epoch", position=1, leave=True)
        for epoch in range(config.epochs):
            pbar_train.reset()
            pbar_val.reset()
            if epoch <= 2:
                pbar.display(f'Epoch [{epoch}/{config.epochs}]')
            else:
                pbar.display(f'Epoch [{epoch}/{config.epochs}]' +
                             f'Loss {log["loss"][-1]:.4f} - ' +
                             f'IOU {log["iou"][-1]:.4f} - ' +
                             f'var_Loss {log["val_loss"][-1]:.4f} - ' +
                             f'val_IOU {log["val_iou"][-1]:.4f}'
                             )

            # train for one epoch
            train_log = self.train_epoch(self.train_loader, pbar_train)
            # evaluate on validation set
            val_log = self.validate_epoch(self.val_loader, pbar_val)

            if config.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
            elif config.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(val_log['loss'])

            log['epoch'].append(epoch)
            log['lr'].append(config.learning_rate)
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])

            pd.DataFrame(log).to_csv(
                f'models/{config.name}/log.csv', index=False)

            trigger += 1

            if val_log['iou'] > best_iou:
                torch.save(self.model.state_dict(), 'models/%s/model.pth' %
                           config.name)
                best_iou = val_log['iou']
                #pbar.display("=> saved best model")
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
        with open(f'models/{config.name}/config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        print('-'*20)
        for key in config.__dict__:
            print('%s: %s' % (key, str(config.__dict__[key])))
        print('-'*20)

        cudnn.benchmark = True

        # create model
        print(f"=> creating model {config.model}")
        model = models.__dict__[config.model](config.num_classes,
                                              config.input_channels,
                                              config.deep_supervision)

        model = model.cuda()

        model.load_state_dict(torch.load(f'models/{config.name}/model.pth'
                                         ))
        model.eval()

        avg_meter = AverageMeter()

        for c in range(config.num_classes):
            os.makedirs(os.path.join(
                'outputs', config.name, str(c)), exist_ok=True)
        with torch.no_grad():
            for input, target, meta in tqdm(self.test_loader, total=len(self.test_loader)):
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
                        cv2.imwrite(os.path.join('outputs', config.name, str(c), meta['img_id'][i] + '.jpg'),
                                    (output[i, c] * 255).astype('uint8'))

        print('IoU: %.4f' % avg_meter.avg)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    config = parse_args().parse_args()
    model = Model(config=config)
    model.setup_dataset()
    model.train()
