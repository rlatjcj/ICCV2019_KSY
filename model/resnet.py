import torch.nn as nn

from .seblock import SE_block

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1,
                 is_seblock=False,
                 se_ratio=16,
                 **kwargs):

        super(BasicBlock, self).__init__()
        self.is_seblock = is_seblock

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        if self.is_seblock:
            self.se_block = SE_block(planes, se_ratio)

        if inplanes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.is_seblock:
            out = self.se_block(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1,
                 is_seblock=False,
                 se_ratio=16,
                 **kwargs):

        super(Bottleneck, self).__init__()
        self.is_seblock = is_seblock

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        if self.is_seblock:
            self.se_block = SE_block(planes*self.expansion, se_ratio)
        
        # if inplanes != planes*self.expansion:
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes*self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes*self.expansion)
        )

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.is_seblock:
            out = self.se_block(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    blocks = {'basic': BasicBlock,
              'bottleneck': Bottleneck}

    def __init__(self, 
                 block,
                 num_blocks,
                 is_seblock=False,
                 se_ratio=16,
                 **kwargs):

        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bottom-up layers
        self.layer1 = self._make_layer(self.blocks[block],  64, num_blocks[0], stride=1, is_seblock=is_seblock, se_ratio=se_ratio)
        self.layer2 = self._make_layer(self.blocks[block], 128, num_blocks[1], stride=2, is_seblock=is_seblock, se_ratio=se_ratio)
        self.layer3 = self._make_layer(self.blocks[block], 256, num_blocks[2], stride=2, is_seblock=is_seblock, se_ratio=se_ratio)
        self.layer4 = self._make_layer(self.blocks[block], 512, num_blocks[3], stride=2, is_seblock=is_seblock, se_ratio=se_ratio)

    def _make_layer(self, block, planes, num_block, stride, is_seblock, se_ratio):
        strides = [stride] + [1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    inplanes=self.inplanes, 
                    planes=planes, 
                    stride=stride,
                    is_seblock=is_seblock,
                    se_ratio=se_ratio
                )
            )
            
            self.inplanes = planes * block.expansion
            
        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5

        
