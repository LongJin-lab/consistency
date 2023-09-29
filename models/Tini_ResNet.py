# %% cifar100 phase3
import torch
import torch.nn as nn
# from torchstat import stat
# from thop import profile
import torch.nn.functional as F

#%%

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_normal=True):
        super(BasicBlock, self).__init__()
        self.is_normal = is_normal
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.is_normal:
            out += self.shortcut(x)
        else:
            out += 1.5*self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlockV2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_normal=True):
        super(BasicBlockV2, self).__init__()
        self.is_normal = is_normal
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.is_normal:
            out += self.shortcut(x)
        else:
            out += 1.5*self.shortcut(x)
        return out

class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_normal=True):
        super(Bottleneck, self).__init__()
        self.is_normal = is_normal
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.is_normal:
            out += self.shortcut(x)
        else:
            out += 1.5*self.shortcut(x)
        out = F.relu(out)
        return out

class BottleneckV2(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(self.bn3(out))
        out += self.shortcut(x)
        return out

class TiniBlockV1( nn.Module ):
    expansion = 1
    def __init__( self, in_planes, planes, stride=2 ):
        super( TiniBlockV1, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

            self.shortcut1 = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

            self.shortcut2 = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    # 并排输入 wk, wk-1, wk-2
    def forward(self, x):
        w_k_2 = self.shortcut2(x[2])
        w_k_1 = self.shortcut1(x[1])
        w_k = self.shortcut(x[0])

        F_wk = F.relu(self.bn1(self.conv1(x[0])))
        F_wk = self.bn2(self.conv2(F_wk))

        out = 1.5*w_k - w_k_1 + 0.5*w_k_2 + F_wk
        out = F.relu(out)

        # 输出 wk+1,wk,wk-1
        return [out, w_k, w_k_1]

class TiniBlockV2( nn.Module ):
    expansion = 4
    def __init__( self, in_planes, planes, stride=2 ):
        super( TiniBlockV2, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

            self.shortcut1 = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

            self.shortcut2 = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )


    # 并排输入 wk, wk-1, wk-2
    def forward(self, x):

        w_k_2 = self.shortcut2(x[2])
        w_k_1 = self.shortcut1(x[1])
        w_k = self.shortcut(x[0])

        F_wk = F.relu(self.bn1(self.conv1(x[0])))
        F_wk = F.relu(self.bn2(self.conv2(F_wk)))
        F_wk = self.bn3(self.conv3(F_wk))

        out = 1.5*w_k - w_k_1 + 0.5*w_k_2 + F_wk
        out = F.relu(out)

        # 输出 wk+1,wk,wk-1
        return [out, w_k, w_k_1]

class TiniBoot( nn.Module ):
    expansion=1
    def __init__( self, block, in_planes, planes, stride=1 ):
        super( TiniBoot, self).__init__()
        
        self.is_need_four = (in_planes != planes*block.expansion)
        self.in_planes = in_planes
        
        if self.is_need_four:
            self.euler_v0 = self._make_layer( block, planes, 1, stride=1 )
            self.expansion = 4
        self.euler_v1 = self._make_layer( block, planes, 1, stride=1 )
        self.euler_v2 = self._make_layer( block, planes, 1, stride=1 )
        self.euler_v3 = self._make_layer( block, planes, 1, stride=1, is_normal=False )

    def _make_layer(self, block, planes, num_blocks, stride, is_normal=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, is_normal=is_normal))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x0):
        if self.is_need_four:
            w_k_2 = self.euler_v0( x0 )
        else:
            w_k_2 = x0
        w_k_1 = self.euler_v1(w_k_2)
        w_k = self.euler_v2(w_k_1)
        grad = self.euler_v3( w_k )
        out = -w_k_1 + (0.5)*w_k_2 + grad
        out = F.relu( out )
        return [w_k_1, w_k, out]

class TiniNet( nn.Module ):
    def __init__( self, boot_blocks, block, num_blocks, num_classes=10):
        super( TiniNet, self).__init__()

        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.boot = self._make_boot_layer( boot_blocks, 64, num_blocks[0] )
        self.tiny1 = self._make_layer( block, 128, num_blocks[1], stride=2 )
        self.tiny2 = self._make_layer( block, 256, num_blocks[2], stride=2 )
        self.tiny3 = self._make_layer( block, 512, num_blocks[3], stride=2 )

        self.linear = nn.Linear( 512, num_classes )

    def _make_layer(self, block, planes, num_blocks, stride ):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_boot_layer(self, boot_blocks, planes, num_blocks ):
        strides = [1] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(boot_blocks[0]( boot_blocks[1], self.in_planes, planes))
            self.in_planes = planes * boot_blocks[0].expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.boot(out)
        out = self.tiny1(out)
        out = self.tiny2(out)
        out = self.tiny3(out)
        out = F.avg_pool2d(out[0], 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class TiniBlockV1Profile( nn.Module ):
    expansion = 1
    def __init__( self, in_planes, planes, stride=2 ):
        super( TiniBlockV1Profile, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

            self.shortcut1 = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

            self.shortcut2 = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    # 并排输入 wk, wk-1, wk-2
    def forward(self, x, x_1, x_2):
        w_k_2 = self.shortcut2(x_2)
        w_k_1 = self.shortcut1(x_1)
        w_k = self.shortcut(x)

        F_wk = F.relu(self.bn1(self.conv1(x)))
        F_wk = self.bn2(self.conv2(F_wk))

        out = 1.5*w_k - w_k_1 + 0.5*w_k_2 + F_wk
        out = F.relu(out)

        # 输出 wk+1,wk,wk-1
        return out, w_k, w_k_1

class TiniBlockV2Profile( nn.Module ):
    expansion = 4
    def __init__( self, in_planes, planes, stride=2 ):
        super( TiniBlockV2Profile, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

            self.shortcut1 = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

            self.shortcut2 = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    # 并排输入 wk, wk-1, wk-2
    def forward(self, x, x_1, x_2):

        w_k_2 = self.shortcut2(x_2)
        w_k_1 = self.shortcut1(x_1)
        w_k = self.shortcut(x)

        F_wk = F.relu(self.bn1(self.conv1(x)))
        F_wk = F.relu(self.bn2(self.conv2(F_wk)))
        F_wk = self.bn3(self.conv3(F_wk))

        out = 1.5*w_k - w_k_1 + 0.5*w_k_2 + F_wk
        out = F.relu(out)

        # 输出 wk+1,wk,wk-1
        return out, w_k, w_k_1

class TiniBootProfile( nn.Module ):
    expansion=1
    def __init__( self, block, in_planes, planes, stride=1 ):
        super( TiniBootProfile, self).__init__()
        
        self.is_need_four = (in_planes != planes*block.expansion)
        self.in_planes = in_planes
        
        if self.is_need_four:
            self.euler_v0 = self._make_layer( block, planes, 1, stride=1 )
            self.expansion = 4
        self.euler_v1 = self._make_layer( block, planes, 1, stride=1 )
        self.euler_v2 = self._make_layer( block, planes, 1, stride=1 )
        self.euler_v3 = self._make_layer( block, planes, 1, stride=1, is_normal=False )

    def _make_layer(self, block, planes, num_blocks, stride, is_normal=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, is_normal=is_normal))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x0):
        if self.is_need_four:
            w_k_2 = self.euler_v0( x0 )
        else:
            w_k_2 = x0
        w_k_1 = self.euler_v1(w_k_2)
        w_k = self.euler_v2(w_k_1)
        grad = self.euler_v3( w_k )
        out = -w_k_1 + (0.5)*w_k_2 + grad
        out = F.relu( out )
        return w_k_1, w_k, out

class TiniNetProfile14( nn.Module ):
    def __init__( self, boot_blocks, block, num_classes=10):
        super( TiniNetProfile14, self).__init__()

        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.boot = boot_blocks[0]( boot_blocks[1], 64, 64 )
        
        self.tiny1 = block( 64, 128 )
        self.tiny2 = block( 128, 128, stride=1 )
        self.tiny3 = block( 128, 128, stride=1 )

        self.tiny4 = block( 128, 256 )
        # self.tiny = block( 256, 256, stride=1 )
        

        self.tiny5 = block( 256, 512 )
        self.tiny6 = block( 512, 512, stride=1 )

        self.linear = nn.Linear( 512, num_classes )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, w_k, w_k_1 = self.boot(out)
        out, w_k, w_k_1 = self.tiny1(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny2(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny3(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny4(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny5(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny6(out, w_k, w_k_1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class TiniNetProfile28( nn.Module ):
    def __init__( self, boot_blocks, block, num_classes=10):
        super( TiniNetProfile28, self).__init__()

        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.boot = boot_blocks[0]( boot_blocks[1], 64, 64 )
        
        self.tiny1 = block( 64, 128 )
        self.tiny2 = block( 128, 128, stride=1 )
        self.tiny3 = block( 128, 128, stride=1 )

        self.tiny4 = block( 128, 256 )
        self.tiny5 = block( 256, 256, stride=1 )
        self.tiny6 = block( 256, 256, stride=1 )
        self.tiny7 = block( 256, 256, stride=1 )
        self.tiny8 = block( 256, 256, stride=1 )
        self.tiny9 = block( 256, 256, stride=1 )

        self.tiny10 = block( 256, 512 )
        self.tiny11 = block( 512, 512, stride=1 )
        self.tiny12 = block( 512, 512, stride=1 )

        self.linear = nn.Linear( 512, num_classes )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, w_k, w_k_1 = self.boot(out)
        out, w_k, w_k_1 = self.tiny1(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny2(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny3(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny4(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny5(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny6(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny7(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny8(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny9(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny10(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny11(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny12(out, w_k, w_k_1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class TiniNetProfile44( nn.Module ):
    def __init__( self, boot_blocks, block, num_classes=10):
        super( TiniNetProfile44, self).__init__()

        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.boot = boot_blocks[0]( boot_blocks[1], 64, 64 )

        self.tiny1 = block( 256, 128 )
        self.tiny2 = block( 512, 128, stride=1 )
        self.tiny3 = block( 512, 128, stride=1 )

        self.tiny4 = block( 512, 256 )
        self.tiny5 = block( 1024, 256, stride=1 )
        self.tiny6 = block( 1024, 256, stride=1 )
        self.tiny7 = block( 1024, 256, stride=1 )
        self.tiny8 = block( 1024, 256, stride=1 )

        self.tiny9 = block( 1024, 512 )
        self.tiny10 = block( 2048, 512, stride=1 )

        self.linear = nn.Linear( 2048, num_classes )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, w_k, w_k_1 = self.boot(out)
        out, w_k, w_k_1 = self.tiny1(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny2(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny3(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny4(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny5(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny6(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny7(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny8(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny9(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny10(out, w_k, w_k_1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class TiniNetProfile89( nn.Module ):
    def __init__( self, boot_blocks, block, num_classes=10):
        super( TiniNetProfile89, self).__init__()

        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.boot = boot_blocks[0]( boot_blocks[1], 64, 64 )

        self.tiny1 = block( 256, 128 )
        self.tiny2 = block( 512, 128, stride=1 )
        self.tiny3 = block( 512, 128, stride=1 )
        self.tiny_1 = block( 512, 128, stride=1 )

        self.tiny4 = block( 512, 256 )
        self.tiny5 = block( 1024, 256, stride=1 )
        self.tiny6 = block( 1024, 256, stride=1 )
        self.tiny7 = block( 1024, 256, stride=1 )
        self.tiny8 = block( 1024, 256, stride=1 )
        self.tiny01 = block( 1024, 256, stride=1 )
        self.tiny02 = block( 1024, 256, stride=1 )
        self.tiny03 = block( 1024, 256, stride=1 )
        self.tiny04 = block( 1024, 256, stride=1 )
        self.tiny05 = block( 1024, 256, stride=1 )
        self.tiny06 = block( 1024, 256, stride=1 )
        self.tiny07 = block( 1024, 256, stride=1 )
        self.tiny08 = block( 1024, 256, stride=1 )
        self.tiny09 = block( 1024, 256, stride=1 )
        self.tiny010 = block( 1024, 256, stride=1 )
        self.tiny011 = block( 1024, 256, stride=1 )
        self.tiny012 = block( 1024, 256, stride=1 )
        self.tiny013 = block( 1024, 256, stride=1 )

        self.tiny9 = block( 1024, 512 )
        self.tiny10 = block( 2048, 512, stride=1 )
        self.tiny11 = block( 2048, 512, stride=1 )

        self.linear = nn.Linear( 2048, num_classes )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, w_k, w_k_1 = self.boot(out)
        out, w_k, w_k_1 = self.tiny1(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny2(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny3(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny_1(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny4(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny5(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny6(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny7(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny8(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny01(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny02(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny03(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny04(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny05(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny06(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny07(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny08(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny09(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny010(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny011(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny012(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny013(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny9(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny10(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny11(out, w_k, w_k_1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class TiniNetProfile140( nn.Module ):
    def __init__( self, boot_blocks, block, num_classes=10):
        super( TiniNetProfile140, self).__init__()

        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.boot = boot_blocks[0]( boot_blocks[1], 64, 64 )

        self.tiny1 = block( 256, 128 )
        self.tiny2 = block( 512, 128, stride=1 )
        self.tiny3 = block( 512, 128, stride=1 )
        self.tiny_1 = block( 512, 128, stride=1 )
        self.tiny_2 = block( 512, 128, stride=1 )
        self.tiny_3 = block( 512, 128, stride=1 )
        self.tiny_4 = block( 512, 128, stride=1 )
        self.tiny_5 = block( 512, 128, stride=1 )

        self.tiny4 = block( 512, 256 )
        self.tiny5 = block( 1024, 256, stride=1 )
        self.tiny6 = block( 1024, 256, stride=1 )
        self.tiny7 = block( 1024, 256, stride=1 )
        self.tiny8 = block( 1024, 256, stride=1 )
        self.tiny01 = block( 1024, 256, stride=1 )
        self.tiny02 = block( 1024, 256, stride=1 )
        self.tiny03 = block( 1024, 256, stride=1 )
        self.tiny04 = block( 1024, 256, stride=1 )
        self.tiny05 = block( 1024, 256, stride=1 )
        self.tiny06 = block( 1024, 256, stride=1 )
        self.tiny07 = block( 1024, 256, stride=1 )
        self.tiny08 = block( 1024, 256, stride=1 )
        self.tiny09 = block( 1024, 256, stride=1 )
        self.tiny010 = block( 1024, 256, stride=1 )
        self.tiny011 = block( 1024, 256, stride=1 )
        self.tiny012 = block( 1024, 256, stride=1 )
        self.tiny013 = block( 1024, 256, stride=1 )
        self.tiny014 = block( 1024, 256, stride=1 )
        self.tiny015 = block( 1024, 256, stride=1 )
        self.tiny016 = block( 1024, 256, stride=1 )
        self.tiny017 = block( 1024, 256, stride=1 )
        self.tiny018 = block( 1024, 256, stride=1 )
        self.tiny019 = block( 1024, 256, stride=1 )
        self.tiny020 = block( 1024, 256, stride=1 )
        self.tiny021 = block( 1024, 256, stride=1 )
        self.tiny022 = block( 1024, 256, stride=1 )
        self.tiny023 = block( 1024, 256, stride=1 )
        self.tiny024 = block( 1024, 256, stride=1 )
        self.tiny025 = block( 1024, 256, stride=1 )
        self.tiny026 = block( 1024, 256, stride=1 )

        self.tiny9 = block( 1024, 512 )
        self.tiny10 = block( 2048, 512, stride=1 )
        self.tiny11 = block( 2048, 512, stride=1 )

        self.linear = nn.Linear( 2048, num_classes )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, w_k, w_k_1 = self.boot(out)
        out, w_k, w_k_1 = self.tiny1(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny2(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny3(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny_1(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny_2(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny_3(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny_4(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny_5(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny4(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny5(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny6(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny7(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny8(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny01(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny02(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny03(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny04(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny05(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny06(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny07(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny08(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny09(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny010(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny011(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny012(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny013(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny014(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny015(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny016(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny017(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny018(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny019(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny020(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny021(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny022(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny023(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny024(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny025(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny026(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny9(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny10(out, w_k, w_k_1)
        out, w_k, w_k_1 = self.tiny11(out, w_k, w_k_1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def TiniResNet14_100():
    # return TiniNet( [TiniBoot, BasicBlock], TiniBlockV1, [ 1, 1, 1, 1 ], 10 )
    return TiniNetProfile14( [TiniBootProfile, BasicBlock], TiniBlockV1Profile, 100 )

def TiniResNet28_100():
    # return TiniNet( [TiniBoot, BasicBlock], TiniBlockV1, [ 1, 1, 1, 1 ], 10 )
    return TiniNetProfile28( [TiniBootProfile, BasicBlock], TiniBlockV1Profile, 100 )

def TiniResNet44_100():
    # return TiniNet( [TiniBoot, BasicBlock], TiniBlockV1, [ 1, 1, 1, 1 ], 10 )
    return TiniNetProfile44( [TiniBootProfile, Bottleneck], TiniBlockV2Profile, 100 )

def TiniResNet89_100():
    # return TiniNet( [TiniBoot, BasicBlock], TiniBlockV1, [ 1, 1, 1, 1 ], 10 )
    return TiniNetProfile89( [TiniBootProfile, Bottleneck], TiniBlockV2Profile, 100 )

def TiniResNet140_100():
    # return TiniNet( [TiniBoot, BasicBlock], TiniBlockV1, [ 1, 1, 1, 1 ], 10 )
    return TiniNetProfile140( [TiniBootProfile, Bottleneck], TiniBlockV2Profile, 100 )

# def per_analysis():
#     device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
#     net = TiniResNet14_100()
#     net.to( device )
#     model_name = 'TiniResNet14_100' + '_Batch'

#     # 模型性能瓶颈分析
#     with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
#         y = net(torch.ones(1,3,32,32).to( device ))
#     print(prof.table())
#     prof.export_chrome_trace('./profile/{}.json'.format(model_name))

#     print(y.cpu().detach().numpy().shape)

# def show_param():
#     model_name = "TiniResNet44_100"
#     net = TiniResNet44_100()
#     # stat( net, (3, 32, 32) )
#     # input = torch.randn(1,3,32,32)
#     # flops, params = profile( net, inputs=(input, ) )

#     print( model_name )

# def test():
#     net = TiniResNet140_100()
#     model_name = 'TiniResNet14_100'

#     y = net(torch.randn(1,3,32,32))

#     print( model_name )# -*- coding=utf-8 -*-
#     print(y.shape)

# test()
# show_param()
# per_analysis()


# %%