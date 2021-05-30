import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, ou_channels, stride=1):
    return nn.Conv2d(in_channels, ou_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels, ou_channels, stride=1):
    return nn.Conv2d(in_channels, ou_channels, kernel_size=1, stride=stride, padding=0, bias=False)


# Basic Block in ResNet for CIFAR
class BasicBlock(nn.Module):

    EXPANSION = 1

    def __init__(self, in_channels, ou_channels, stride=1, attention_module=None):
        super(BasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(in_channels, ou_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(ou_channels)
        
        self.conv2 = conv3x3(ou_channels, ou_channels * self.EXPANSION, stride=1)
        self.bn2 = nn.BatchNorm2d(ou_channels * self.EXPANSION)

        if attention_module is not None:
            if type(attention_module) == functools.partial:
                module_name = attention_module.func.get_module_name()
            else:
                module_name = attention_module.get_module_name()


            if module_name == "simam":
                self.conv2 = nn.Sequential(
                    self.conv2,
                    attention_module(ou_channels * self.EXPANSION)
                )
            else:
                self.bn2 = nn.Sequential(
                    self.bn2, 
                    attention_module(ou_channels * self.EXPANSION)
                )  

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != ou_channels * self.EXPANSION:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, ou_channels * self.EXPANSION, stride=stride),
                nn.BatchNorm2d(ou_channels * self.EXPANSION)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        
        return self.relu(out)

# Bottlenect in ResNet for CIFAR
class BottleNect(nn.Module):

    EXPANSION = 4

    def __init__(self, in_channels, ou_channels, stride=1, attention_module=None):
        super(BottleNect, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv1x1(in_channels, ou_channels, stride=1)
        self.bn1 = nn.BatchNorm2d(ou_channels)

        self.conv2 = conv3x3(ou_channels, ou_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(ou_channels)

        self.conv3 = conv1x1(ou_channels, ou_channels * self.EXPANSION, stride=1)
        self.bn3 = nn.BatchNorm2d(ou_channels * self.EXPANSION)

        if attention_module is not None:
            if type(attention_module) == functools.partial:
                module_name = attention_module.func.get_module_name()
            else:
                module_name = attention_module.get_module_name()

            if module_name == "simam":
                self.conv2 = nn.Sequential(
                    self.conv2,
                    attention_module(ou_channels * self.EXPANSION)
                )
            else:
                self.bn3 = nn.Sequential(
                    self.bn3, 
                    attention_module(ou_channels * self.EXPANSION)
                )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != ou_channels * self.EXPANSION:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, ou_channels * self.EXPANSION, stride=stride),
                nn.BatchNorm2d(ou_channels * self.EXPANSION)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.shortcut(x)

        return self.relu(out)



# PreBasic Block in ResNet for CIFAR
class PreBasicBlock(nn.Module):

    EXPANSION = 1

    def __init__(self, in_channels, ou_channels, stride=1, attention_module=None):
        super(PreBasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, ou_channels, stride=stride)

        self.bn2 = nn.BatchNorm2d(ou_channels)
        self.conv2 = conv3x3(ou_channels, ou_channels * self.EXPANSION, stride=1)

        if attention_module is not None:
            self.conv2 = nn.Sequential(
                self.conv2,
                attention_module(ou_channels * self.EXPANSION)
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != ou_channels * self.EXPANSION:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, ou_channels * self.EXPANSION, stride=stride)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + self.shortcut(x)
        return out


# Bottlenect in ResNet for CIFAR
class PreBottleNect(nn.Module):

    EXPANSION = 4

    def __init__(self, in_channels, ou_channels, stride=1, attention_module=None):
        super(PreBottleNect, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv1x1(in_channels, ou_channels, stride=1)
  
        self.bn2 = nn.BatchNorm2d(ou_channels)
        self.conv2 = conv3x3(ou_channels, ou_channels, stride=stride)
        
        self.bn3 = nn.BatchNorm2d(ou_channels)
        self.conv3 = conv1x1(ou_channels, ou_channels * self.EXPANSION, stride=1)

        if attention_module is not None:
            if type(attention_module) == functools.partial:
                module_name = attention_module.func.get_module_name()
            else:
                module_name = attention_module.get_module_name()

            if module_name == "simam":
                self.conv2 = nn.Sequential(
                    self.conv2,
                    attention_module(ou_channels * self.EXPANSION)
                )
            else:
                self.conv3 = nn.Sequential(
                    self.conv3, 
                    attention_module(ou_channels * self.EXPANSION)
                )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != ou_channels * self.EXPANSION:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, ou_channels * self.EXPANSION, stride=stride)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = out + self.shortcut(x)

        return out


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, attention_module=None):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, out_planes, stride=stride)
        
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes, stride=1)

        if attention_module is not None:
            self.conv2 = nn.Sequential(
                self.conv2,
                attention_module(out_planes)
            )
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and conv1x1(in_planes, out_planes, stride=stride) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class InvertedResidualBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, attention_module=None):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = conv1x1(in_planes, planes, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, out_planes, stride=1)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

        if attention_module is not None:
            if type(attention_module) == functools.partial:
                module_name = attention_module.func.get_module_name()
            else:
                module_name = attention_module.get_module_name()

            if module_name == "simam":
                self.conv2 = nn.Sequential(
                    self.conv2,
                    attention_module(planes)
                )
            else:
                self.bn3 = nn.Sequential(
                    self.bn3,
                    attention_module(out_planes)
                )

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, out_planes, stride=1),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + self.shortcut(x) if self.stride == 1 else out
        
        return out