# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from .DSFD_vgg    import build_net_vgg
from .DSFD_resnet import build_net_resnet
from .DAINet      import build_net_dark
from .DAINet_yolov8 import build_net_yolo


def build_net(phase, num_classes=2, model='vgg'):
    if phase not in ('test', 'train'):
        print('ERROR: Phase: ' + phase + ' not recognized')
        return

    if model == 'vgg':
        return build_net_vgg(phase, num_classes)
    elif model == 'dark':
        return build_net_dark(phase, num_classes)
    elif model == 'yolo_dark':
        # YOLOv8n backbone + DAI-Net zero-shot components.
        # num_classes here means person classes only (1), not including background.
        return build_net_yolo(phase, num_classes=1)
    else:
        return build_net_resnet(phase, num_classes, model)


def basenet_factory(model='vgg'):
    if model in ('vgg', 'dark'):
        return 'vgg16_reducedfc.pth'
    elif model == 'yolo_dark':
        # No external pretrained base needed — YOLOv8n trains from scratch
        # (ImageNet pretrained backbone can be loaded separately if available)
        return None
    elif 'resnet' in model:
        return '{}.pth'.format(model)
    return None
