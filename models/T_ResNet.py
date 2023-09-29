# %% cifar100 phase1 phase2、cifar10 phase1
import torch
import torch.nn as nn
# from torchstat import stat
import torch.nn.functional as F

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

class TaylorBlock( nn.Module ):
    def __init__( self, block, in_planes, planes, stride=2, is_bootblock=False ):
        super( TaylorBlock, self).__init__()
        
        self.is_bootblock = is_bootblock
        self.is_need_four = (in_planes != planes*block.expansion)
        self.in_planes = in_planes
        
        if self.is_bootblock:
            if self.is_need_four:
                self.euler_v0 = self._make_layer( block, planes, 1, stride=1 )
            self.euler_v1 = self._make_layer( block, planes, 1, stride=1 )
            self.euler_v2 = self._make_layer( block, planes, 1, stride=1 )
            self.euler_v3 = self._make_layer( block, planes, 1, stride=1, is_normal=False )
        else:
            self.euler_h1 = self._make_layer( block, planes, 1, stride=stride )
            self.in_planes = in_planes
            self.euler_h2 = self._make_layer( block, planes, 1, stride=stride )
            self.in_planes = in_planes
            self.euler_h3 = self._make_layer( block, planes, 1, stride=stride, is_normal=False )

    def _make_layer(self, block, planes, num_blocks, stride, is_normal=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, is_normal=is_normal))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x0, x1=None, x2=None):
        if self.is_bootblock:
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
        else:
            w_k_2 = self.euler_h1( x0 )
            w_k_1 = self.euler_h2( x1 )
            grad = self.euler_h3( x2 )
            out = -w_k_1 + (0.5)*w_k_2 + grad
            out = F.relu( out )
            return w_k_2, w_k_1, out

class TaylorBlockV2( nn.Module ):

    def __init__( self, block, in_planes, planes, stride=2, is_bootblock=False ):
        super( TaylorBlockV2, self).__init__()
        
        self.is_bootblock = is_bootblock
        self.is_need_four = (in_planes != planes*block.expansion)
        self.in_planes = in_planes
        
        if self.is_bootblock:
            if self.is_need_four:
                self.euler_v0 = self._make_layer( block, planes, 1, stride=1 )
            self.euler_v1 = self._make_layer( block, planes, 1, stride=1 )
            self.euler_v2 = self._make_layer( block, planes, 1, stride=1 )
            self.euler_v3 = self._make_layer( block, planes, 1, stride=1, is_normal=False )
        else:
            self.euler_h1 = self._make_layer( block, planes, 1, stride=stride )
            self.in_planes = in_planes
            self.euler_h2 = self._make_layer( block, planes, 1, stride=stride )
            self.in_planes = in_planes
            self.euler_h3 = self._make_layer( block, planes, 1, stride=stride, is_normal=False )
        self.out_bn1 = nn.BatchNorm2d( planes )
        self.out_bn2 = nn.BatchNorm2d( planes )
        self.out_bn3 = nn.BatchNorm2d( planes )

    def _make_layer(self, block, planes, num_blocks, stride, is_normal=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, is_normal=is_normal))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x0, x1=None, x2=None):
        if self.is_bootblock:
            if self.is_need_four:
                w_k_2 = self.euler_v0( x0 )
            else:
                w_k_2 = x0
            w_k_1 = self.euler_v1(w_k_2)
            w_k = self.euler_v2(w_k_1)
            grad = self.euler_v3( w_k )
            out = self.out_bn1(-w_k_1) + self.out_bn2((0.5)*w_k_2) + self.out_bn3(grad)
            return w_k_1, w_k, out
        else:
            w_k_2 = self.euler_h1( x0 )
            w_k_1 = self.euler_h2( x1 )
            grad = self.euler_h3( x2 )
            out = self.out_bn1(-w_k_1) + self.out_bn2((0.5)*w_k_2) + self.out_bn3(grad)
            return w_k_2, w_k_1, out


'''纯宽网络'''
# 14 2泰勒2残差 对比res18
class TResNet6_2( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TResNet6_2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 64, 128 )
        
        self.in_planes = 128
        self.euler1 = self._make_layer( sub_block, 256, 1, stride=2 )
        self.euler2 = self._make_layer( sub_block, 512, 1, stride=2 )

        self.linear = nn.Linear( 512, num_classes )

    def _make_layer(self, block, planes, num_blocks, stride, is_normal=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, is_normal=is_normal))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        out = self.euler1( out )
        out = self.euler2( out )
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 14 3泰勒1残差 对比res18
class TResNet6_3( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TResNet6_3, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        self.taylor2 = block( sub_block, 64, 128 )
        self.taylor3 = block( sub_block, 128, 256 )
        
        self.in_planes = 256
        self.euler1 = self._make_layer( sub_block, 512, 1, stride=2 )

        self.linear = nn.Linear( 512, num_classes )

    def _make_layer(self, block, planes, num_blocks, stride, is_normal=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, is_normal=is_normal))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(w_k_1, w_k, out)
        out = self.euler1( out )
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 18 对比res34
class TResNet10( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TResNet10, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 64, 128 )
        self.taylor3 = block( sub_block, 128, 128, stride=1 )
        
        self.taylor4 = block( sub_block, 128, 256 )
        self.taylor5 = block( sub_block, 256, 256, stride=1 )
        
        self.taylor6 = block( sub_block, 256, 512 )

        self.linear = nn.Linear( 512, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 22 对比res50
class TResNet12( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TResNet12, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 64, 128 )
        self.taylor4 = block( sub_block, 128, 128, stride=1 )
        self.taylor3 = block( sub_block, 128, 128, stride=1 )
        
        self.taylor5 = block( sub_block, 128, 256 )
        self.taylor6 = block( sub_block, 256, 256, stride=1 )
        self.taylor7 = block( sub_block, 256, 256, stride=1 )
        
        self.taylor8 = block( sub_block, 256, 512 )

        self.linear = nn.Linear( 512, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor8(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 26 对比res101
class TResNet14( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TResNet14, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 64, 128 )
        self.taylor3 = block( sub_block, 128, 128, stride=1 )
        self.taylor4 = block( sub_block, 128, 128, stride=1 )
        
        self.taylor5 = block( sub_block, 128, 256 )
        self.taylor6 = block( sub_block, 256, 256, stride=1 )
        self.taylor7 = block( sub_block, 256, 256, stride=1 )
        self.taylor8 = block( sub_block, 256, 256, stride=1 )
        
        self.taylor9 = block( sub_block, 256, 512 )
        self.taylor10 = block( sub_block, 512, 512, stride=1 )
        
        self.linear = nn.Linear( 512, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor8(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor9(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor10(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 40 对比res152
class TResNet21( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TResNet21, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 64, 128 )
        self.taylor3 = block( sub_block, 128, 128, stride=1 )
        self.taylor4 = block( sub_block, 128, 128, stride=1 )
        self.taylor5 = block( sub_block, 128, 128, stride=1 )
        self.taylor6 = block( sub_block, 128, 128, stride=1 )
        self.taylor7 = block( sub_block, 128, 128, stride=1 )
        
        self.taylor8 = block( sub_block, 128, 256 )
        self.taylor9 = block( sub_block, 256, 256, stride=1 )
        self.taylor10 = block( sub_block, 256, 256, stride=1 )
        self.taylor11 = block( sub_block, 256, 256, stride=1 )
        self.taylor12 = block( sub_block, 256, 256, stride=1 )
        self.taylor13 = block( sub_block, 256, 256, stride=1 )
        self.taylor14 = block( sub_block, 256, 256, stride=1 )
        self.taylor15 = block( sub_block, 256, 256, stride=1 )
        
        self.taylor16 = block( sub_block, 256, 512 )
        self.taylor17 = block( sub_block, 512, 512, stride=1 )
        
        self.linear = nn.Linear( 512, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor8(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor9(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor10(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor11(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor12(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor13(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor14(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor15(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor16(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor17(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


''''肥瘦相间网络'''
# 26 对比res34
class TsResNet32( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TsResNet32, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 64, 128 )
        self.taylor3 = block( sub_block, 128, 128, is_bootblock=True )
        
        self.taylor4 = block( sub_block, 128, 256 )
        self.taylor5 = block( sub_block, 256, 256, is_bootblock=True )
        
        self.taylor6 = block( sub_block, 256, 512 )

        self.linear = nn.Linear( 512, num_classes )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor5(out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 28 对比res50
class TsResNet36( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TsResNet36, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 64, 128 )
        self.taylor3 = block( sub_block, 128, 128, is_bootblock=True )
        
        self.taylor4 = block( sub_block, 128, 256 )
        self.taylor5 = block( sub_block, 256, 256, stride=1 )
        self.taylor6 = block( sub_block, 256, 256, is_bootblock=True )
        
        self.taylor7 = block( sub_block, 256, 512 )

        self.linear = nn.Linear( 512, num_classes )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 非规律结构 对应101
class TsResNet58( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TsResNet58, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        self.taylor2 = block( sub_block, 64, 64, stride=1 )
        
        self.taylor3 = block( sub_block, 64, 128 )
        self.taylor4 = block( sub_block, 128, 128, is_bootblock=True )
        self.taylor5 = block( sub_block, 128, 128, is_bootblock=True )
        
        self.taylor6 = block( sub_block, 128, 256 )
        self.taylor7 = block( sub_block, 256, 256, stride=1 )
        self.taylor8 = block( sub_block, 256, 256, is_bootblock=True )
        self.taylor9 = block( sub_block, 256, 256, is_bootblock=True )
        
        self.taylor10 = block( sub_block, 256, 512 )
        self.taylor11 = block( sub_block, 512, 512,  is_bootblock=True )
        
        self.linear = nn.Linear( 512, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor4(out)
        w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor8(out)
        w_k_1, w_k, out = self.taylor9(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor10(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor11(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 60 规律结构 瘦肥瘦 对应101
class TsResNet70( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TsResNet70, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.taylor1 = block( sub_block, 64, 64, is_bootblock=True )
        self.taylor2 = block( sub_block, 64, 64, stride=1 )
        self.taylor3 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor4 = block( sub_block, 64, 64, is_bootblock=True )
        self.taylor5 = block( sub_block, 64, 128 ) 
        self.taylor6 = block( sub_block, 128, 128, is_bootblock=True )
        
        self.taylor9 = block( sub_block, 128, 128, is_bootblock=True )
        self.taylor10 = block( sub_block, 128, 256 )
        self.taylor11 = block( sub_block, 256, 256, is_bootblock=True )
        self.taylor12 = block( sub_block, 256, 256, stride=1 )
        
        self.taylor13 = block( sub_block, 256, 256, is_bootblock=True )
        self.taylor14 = block( sub_block, 256, 512 )
        self.taylor15 = block( sub_block, 512, 512, is_bootblock=True )

        self.linear = nn.Linear( 512, num_classes )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(out)
        w_k_1, w_k, out = self.taylor4(out)
        w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(out)
        # w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        # w_k_1, w_k, out = self.taylor8(out)
        w_k_1, w_k, out = self.taylor9(out)
        w_k_1, w_k, out = self.taylor10(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor11(out)
        w_k_1, w_k, out = self.taylor12(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor13(out)
        w_k_1, w_k, out = self.taylor14(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor15(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 84 对比res152
class TsResNet84( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TsResNet84, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.taylor1 = block( sub_block, 64, 64, is_bootblock=True )
        self.taylor2 = block( sub_block, 64, 64, stride=1 )
        self.taylor3 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor4 = block( sub_block, 64, 64, is_bootblock=True )
        self.taylor5 = block( sub_block, 64, 128 ) 
        self.taylor6 = block( sub_block, 128, 128, is_bootblock=True )
        self.taylor7 = block( sub_block, 128, 128, stride=1 ) 
        self.taylor8 = block( sub_block, 128, 128, is_bootblock=True )
        
        self.taylor9 = block( sub_block, 128, 128, is_bootblock=True )
        self.taylor10 = block( sub_block, 128, 256 )
        self.taylor11 = block( sub_block, 256, 256, is_bootblock=True )
        self.taylor12 = block( sub_block, 256, 256, stride=1 )
        self.taylor13 = block( sub_block, 256, 256, is_bootblock=True )
        self.taylor14 = block( sub_block, 256, 256, stride=1 )
        self.taylor15 = block( sub_block, 256, 256, is_bootblock=True )
        self.taylor16 = block( sub_block, 256, 256, stride=1 )
        
        self.taylor17 = block( sub_block, 256, 256, is_bootblock=True )
        self.taylor18 = block( sub_block, 256, 512 )
        self.taylor19 = block( sub_block, 512, 512, is_bootblock=True )

        self.linear = nn.Linear( 512, num_classes )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(out)
        w_k_1, w_k, out = self.taylor4(out)
        w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor8(out)
        w_k_1, w_k, out = self.taylor9(out)
        w_k_1, w_k, out = self.taylor10(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor11(out)
        w_k_1, w_k, out = self.taylor12(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor13(out)
        w_k_1, w_k, out = self.taylor14(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor15(out)
        w_k_1, w_k, out = self.taylor16(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor17(out)
        w_k_1, w_k, out = self.taylor18(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor19(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 暂时不用
class TResNet12B( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TResNet12B, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 256, 128 )
        self.taylor3 = block( sub_block, 512, 128, stride=1 )
        
        self.taylor4 = block( sub_block, 512, 256  )
        self.taylor5 = block( sub_block, 1024, 256, stride=1 )
        
        self.taylor6 = block( sub_block, 1024, 512 )
        self.taylor7 = block( sub_block, 2048, 512, stride=1 )

        self.linear = nn.Linear( 2048, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 暂时不用
class TResNet19B( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( TResNet19B, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 256, 128 )
        self.taylor3 = block( sub_block, 512, 128, stride=1 )
        self.taylor4 = block( sub_block, 512, 128, stride=1 )
        self.taylor5 = block( sub_block, 512, 128, stride=1 )
        self.taylor6 = block( sub_block, 512, 128, stride=1 )
        
        self.taylor7 = block( sub_block, 512, 256  )
        self.taylor8 = block( sub_block, 1024, 256, stride=1 )
        self.taylor9 = block( sub_block, 1024, 256, stride=1 )
        self.taylor10 = block( sub_block, 1024, 256, stride=1 )
        self.taylor11 = block( sub_block, 1024, 256, stride=1 )
        self.taylor12 = block( sub_block, 1024, 256, stride=1 )
        
        self.taylor13 = block( sub_block, 1024, 512 )
        self.taylor14 = block( sub_block, 2048, 512, stride=1 )

        self.linear = nn.Linear( 2048, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor8(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor9(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor10(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor11(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor12(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor13(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor14(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def T_ResNet6V2_2():
    return TResNet6_2( TaylorBlockV2, BasicBlockV2 )

def T_ResNet6V2_3():
    return TResNet6_3( TaylorBlockV2, BasicBlockV2 )

def T_ResNet10():
    return TResNet10( TaylorBlock, BasicBlock )

def T_ResNet10V2():
    return TResNet10( TaylorBlockV2, BasicBlockV2 )

def T_ResNet12():
    return TResNet12( TaylorBlock, BasicBlock )

# 增加batchnorm操作
def T_ResNet12V2():
    return TResNet12( TaylorBlockV2, BasicBlockV2 )

# 结构上改进算法
def Ts_ResNet32():
    return TsResNet32( TaylorBlockV2, BasicBlockV2 )

def Ts_ResNet36():
    return TsResNet36( TaylorBlockV2, BasicBlockV2 )

def Ts_ResNet58():
    return TsResNet58( TaylorBlockV2, BasicBlock )

def Ts_ResNet70():
    return TsResNet70( TaylorBlockV2, BasicBlock )

def Ts_ResNet84():
    return TsResNet84( TaylorBlockV2, BasicBlock )

def T_ResNet14():
    return TResNet14( TaylorBlock, BasicBlock )

def T_ResNet14V2():
    return TResNet14( TaylorBlockV2, BasicBlock )

def T_ResNet21():
    return TResNet21( TaylorBlock, BasicBlock )

def T_ResNet21V2():
    return TResNet21( TaylorBlockV2, BasicBlock )

def T_ResNet12B():
    return TResNet12B( TaylorBlock, Bottleneck )

def T_ResNet19B():
    return TResNet19B( TaylorBlock, Bottleneck )

def T100_ResNet6V2_2():
    return TResNet6_2( TaylorBlockV2, BasicBlockV2, 100 )

def T100_ResNet6V2_3():
    return TResNet6_3( TaylorBlockV2, BasicBlockV2, 100 )

def T100_ResNet10V2():
    return TResNet10( TaylorBlockV2, BasicBlockV2, 100 )

def T100_ResNet12V2():
    return TResNet12( TaylorBlockV2, BasicBlockV2, 100 )

def T100_ResNet14V2():
    return TResNet14( TaylorBlockV2, BasicBlock, 100 )

def T100_ResNet21V2():
    return TResNet21( TaylorBlockV2, BasicBlock, 100 )

def Ts100_ResNet32():
    return TsResNet32( TaylorBlockV2, BasicBlockV2, 100 )

def Ts100_ResNet36():
    return TsResNet36( TaylorBlockV2, BasicBlockV2, 100 )

def Ts100_ResNet70():
    return TsResNet70( TaylorBlockV2, BasicBlock, 100 )

def Ts100_ResNet84():
    return TsResNet84( TaylorBlockV2, BasicBlock, 100 )

def profile():
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    net = T_ResNet14()
    net.to( device )
    model_name = 'T_ResNet14' + '_Batch'

    # 模型性能瓶颈分析
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        y = net(torch.ones(1,3,32,32).to( device ))
    print(prof.table())
    prof.export_chrome_trace('./profile/{}.json'.format(model_name))

    print(y.cpu().detach().numpy().shape)

# def show_param():
#     model_name = "Ts_ResNet84"
#     net = Ts_ResNet84()
#     stat(net, (3,32,32))

#     print( model_name )

def test():
    net = Ts_ResNet64()
    model_name = 'Ts_ResNet64'

    y = net(torch.randn(1,3,32,32))

    print( model_name )
    print(y.shape)

# test()
# show_param()
# profile()


# %%