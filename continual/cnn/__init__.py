from continual.cnn.abstract import AbstractCNN
from continual.cnn.inception import InceptionV3
from continual.cnn.senet import legacy_seresnet18 as seresnet18
from continual.cnn.resnet import (
    resnet18, resnet34, resnet50, resnext50_32x4d, wide_resnet50_2
)
from continual.cnn.resnet_scs import resnet18_scs, resnet18_scs_avg, resnet18_scs_max
from continual.cnn.vgg import vgg16_bn, vgg16
from continual.cnn.resnet_rebuffi import CifarResNet as rebuffi
