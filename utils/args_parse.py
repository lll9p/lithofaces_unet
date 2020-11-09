import argparse

import losses
import models

from .utils import str2bool

MODELS = models.__all__
LOSSES = losses.__all__


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: model+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', '-b', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--model', '-a', metavar='MODEL', default='NestedUNet',
                        choices=MODELS,
                        help='model architecture: ' +
                        ' | '.join(MODELS) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=11, type=int,
                        help='number of classes')
    parser.add_argument('--input_wide', default=224, type=int,
                        help='image width')
    parser.add_argument('--input_height', default=224, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSSES,
                        help='loss: ' +
                        ' | '.join(LOSSES) +
                        ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='minerals_224',
                        help='dataset name')
    parser.add_argument('--path', default='/kaggle/input/lithofaces-dataset-generate/data_256/',
                        help='dataset path')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
    parser.add_argument('--labels', default=["Alite", "Blite", "C3A", "fCaO", "Pore", "iAlite", "iBlite", "iC3A", "ifCaO", "iPore", "edges"],
                        type=lambda int(s): s.split(","),
                        help='labels')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--learning_rate', '--lr', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_learning_rate', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=4, type=int)

    return parser
