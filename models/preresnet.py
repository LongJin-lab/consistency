# -*-coding:utf-8-*-

import torch.nn as nn




def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn_1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = conv3x3(inplanes, planes, stride)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn_1(x)
        out = self.relu(out)
        out = self.conv_1(out)

        out = self.bn_2(out)
        out = self.relu(out)
        out = self.conv_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out



class AB2BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, k1=1.0, k2=1.0, h=1.0):
        super(AB2BasicBlock, self).__init__()
        
        self.k1=k1
        self.k2=k2
        self.h=h
        
        self.bn_1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = conv3x3(inplanes, planes, stride)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_2 = conv3x3(planes, planes)
        
        self.downsample1 = nn.Sequential()
        self.downsample2 = nn.Sequential()
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample1 = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )
            self.downsample2 = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )
            
        self.stride = stride
        
        

    def forward(self, x):
        yn, fyn_1 = x
        
        fyn = self.conv_2(self.relu(self.bn_2(self.conv_1(self.relu(self.bn_1(yn))))))

        yn1 = self.k1*self.h*fyn
        
        yn1 += self.downsample1(yn)
        
        if type(fyn_1)!=type(1):
            yn1 += self.k2*self.downsample2(fyn_1)

        x = [yn1,fyn]

        return x

class AB2PreResNet(nn.Module):
    def __init__(self, depth, h=1.0, k1=1.0, k2=1.0, num_classes=1000, block_name="AB2BasicBlock"):
        super(AB2PreResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == "ab2basicblock":
            assert (
                depth - 2
            ) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
            block = AB2BasicBlock
        elif block_name.lower() == "bottleneck":
            assert (
                depth - 2
            ) % 9 == 0, "When use bottleneck, depth should be 9n+2 e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError("block_name shoule be Basicblock or Bottleneck")

        self.h = h
        self.k1 = k1
        self.k2 = k2

        self.inplanes = 16
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        print('self.inplanes:'+str(self.inplanes))
        print('self.k1:'+str(self.k1))
        layers.append(block(self.inplanes, planes, stride, self.k1, self.k2, self.h))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, self.k1, self.k2, self.h))

        return nn.Sequential(*layers)

    def forward(self, x):
        yn = self.conv_1(x)  # 32x32

        out=[yn,0]
        
        out = self.layer1(out)  # 32x32
        out=[out[0],0]
        
        out = self.layer2(out)  # 16x16
        out=[out[0],0]
        
        out = self.layer3(out)  # 8x8
        yn=out[0]
        
        out = self.bn(yn)
        out = self.relu(out)

        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn_1 = nn.BatchNorm2d(inplanes)
        self.conv_1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn_3 = nn.BatchNorm2d(planes)
        self.conv_3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn_1(x)
        out = self.relu(out)
        out = self.conv_1(out)

        out = self.bn_2(out)
        out = self.relu(out)
        out = self.conv_2(out)

        out = self.bn_3(out)
        out = self.relu(out)
        out = self.conv_3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class PreResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, block_name="BasicBlock"):
        super(PreResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == "basicblock":
            assert (
                depth - 2
            ) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == "bottleneck":
            assert (
                depth - 2
            ) % 9 == 0, "When use bottleneck, depth should be 9n+2 e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError("block_name shoule be Basicblock or Bottleneck")

        self.inplanes = 16
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ab2preresnet(depth, k1, k2, h, num_classes):
    return AB2PreResNet(depth=depth,  h=h, k1=k1, k2=k2, num_classes=num_classes)

def preresnet20(depth, num_classes):
    return PreResNet(depth=depth, num_classes=num_classes)


def preresnet32(num_classes):
    return PreResNet(depth=32, num_classes=num_classes)


def preresnet44(num_classes):
    return PreResNet(depth=44, num_classes=num_classes)


def preresnet56(num_classes):
    return PreResNet(depth=56, num_classes=num_classes)


def preresnet110(num_classes):
    return PreResNet(depth=110, num_classes=num_classes)


def preresnet1202(num_classes):
    return PreResNet(depth=1202, num_classes=num_classes)