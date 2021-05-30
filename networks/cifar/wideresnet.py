import math
import functools
import torch.nn as nn
import torch.nn.functional as F
from .block import WideBasicBlock


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, block, depth, widen_factor=1, dropRate=0.0, num_class=10):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        # block = WideBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_class)
        self.fc.bias.data.zero_()
        # print(self.fc.bias.data)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     if hasattr(m, "bias"):
            #         m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def WideResNetWrapper(depth, widen_factor, dropRate=0, num_class=10, attention_module=None):

    b = lambda in_planes, planes, stride, dropRate: \
        WideBasicBlock(in_planes, planes, stride, dropRate, attention_module=attention_module)

    return WideResNet(b, depth, widen_factor, dropRate, num_class=num_class)


def WideResNet28x10(num_class=10, block=None, attention_module=None):

    return WideResNetWrapper(
        depth = 28,
        widen_factor = 10,
        dropRate = 0.3,
        num_class = num_class, 
        attention_module = attention_module)


def WideResNet40x10(num_class=10, block=None, attention_module=None):

    return WideResNetWrapper(
        depth = 40,
        widen_factor = 10,
        dropRate = 0.3,
        num_class = num_class, 
        attention_module = attention_module)