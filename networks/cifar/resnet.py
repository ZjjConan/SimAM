"""PyTorch implementation of ResNet

ResNet modifications written by Bichen Wu and Alvin Wan, based
off of ResNet implementation by Kuang Liu.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import functools
import torch.nn as nn
from .block import BasicBlock, BottleNect

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_class=10):
        super(ResNet, self).__init__()

        self.num_class = num_class
        self.in_channels = num_filters = 16

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.layer1 = self._make_layer(block, self.in_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(num_filters*2), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(num_filters*4), num_blocks[2], stride=2)
        self.linear = nn.Linear(int(num_filters*4*block(16,16,1).EXPANSION), num_class)

    def _make_layer(self, block, ou_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, ou_channels, stride))
            self.in_channels = int(ou_channels * block(16,16,1).EXPANSION)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNetWrapper(num_blocks, num_class=10, block=None, attention_module=None):

    b = lambda in_planes, planes, stride: \
        block(in_planes, planes, stride, attention_module=attention_module)

    return ResNet(b, num_blocks, num_class=num_class)


def ResNet20(num_class=10, block=None, attention_module=None):
    return ResNetWrapper(
        num_blocks = [3, 3, 3], 
        num_class = num_class, 
        block = block,
        attention_module = attention_module)

def ResNet32(num_class=10, block=None, attention_module=None):
    return ResNetWrapper(
        num_blocks = [5, 5, 5], 
        num_class = num_class, 
        block = block,
        attention_module = attention_module)


def ResNet56(num_class=10, block=None, attention_module=None):

    if block == BasicBlock:
        n_blocks = [9, 9, 9]
    elif block == BottleNect:
        n_blocks = [6, 6, 6]

    return ResNetWrapper(
            num_blocks = n_blocks, 
            num_class = num_class, 
            block = block,
            attention_module = attention_module)

def ResNet110(num_class=10, block=None, attention_module=None):
    
    if block == BasicBlock:
        n_blocks = [18, 18, 18]
    elif block == BottleNect:
        n_blocks = [12, 12, 12]

    return ResNetWrapper(
            num_blocks = n_blocks, 
            num_class = num_class, 
            block = block,
            attention_module = attention_module)


def ResNet164(num_class=10, block=None, attention_module=None):
    
    if block == BasicBlock:
        n_blocks = [27, 27, 27]
    elif block == BottleNect:
        n_blocks = [18, 18, 18]

    return ResNetWrapper(
            num_blocks = n_blocks,  
            num_class = num_class, 
            block = block,
            attention_module = attention_module)