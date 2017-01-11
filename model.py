from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.training


KERNEL_SIZE = 3
MAXPOOL_SIZE = 2


class VGGBlock(chainer.ChainList):
    def __init__(self, filter_num, layer_num):
        self.layer_num = layer_num
        # use comprehension to get different objects
        conv_layers = [L.Convolution2D(None, filter_num, KERNEL_SIZE, pad=1) for _ in range(layer_num)]

        super(VGGBlock, self).__init__(
            *conv_layers
        )

    def __call__(self, x):
        h = x
        for i in range(self.layer_num):
            h = F.relu(self[i](h))
        return F.max_pooling_2d(h, MAXPOOL_SIZE)


class VGG16(chainer.Chain):
    def __init__(self):
        super(VGG16, self).__init__(
            conv1=VGGBlock(64, 2),
            conv2=VGGBlock(128, 2),
            conv3=VGGBlock(256, 3),
            conv4=VGGBlock(512, 3),
            conv5=VGGBlock(512, 3),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, 1000)
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)

        h = self.fc6(h)
        h = self.fc7(h)
        return self.fc8(h)

