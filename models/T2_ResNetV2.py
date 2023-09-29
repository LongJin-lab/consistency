# %% cifar10 phase4
import torch
import torch.nn as nn
# from torchstat import stat
# from torchsummary import summary
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_normal=True):
        super(BasicBlock, self).__init__()
        self.is_normal = is_normal
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.is_normal:
            out += shortcut
        else:
            out += 1.5*shortcut
        return out, shortcut

class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_normal=True):
        super(Bottleneck, self).__init__()
        self.is_normal = is_normal
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        if self.is_normal:
            out += shortcut
        else:
            out += 1.5*shortcut
        return out, shortcut

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
            self.euler_v2  = self._make_layer( block, planes, 1, stride=1 )
            self.euler_v3 = self._make_layer( block, planes, 1, stride=1, is_normal=False )

        else:
            self.bnh1 = nn.BatchNorm2d(in_planes)
            self.bnh2 = nn.BatchNorm2d(in_planes)
            self.euler_h3 = self._make_layer( block, planes, 1, stride=stride, is_normal=False )

            if stride != 1 or in_planes != block.expansion*planes:
                self.map1 = nn.Sequential(
                    nn.Conv2d(in_planes, block.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                )

                self.map2 = nn.Sequential(
                    nn.Conv2d(in_planes, block.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                )
            
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
                w_k_2, _ = self.euler_v0( x0 )
            else:
                w_k_2 = x0
            w_k_1, w_k_2 = self.euler_v1(w_k_2)
            w_k, w_k_1 = self.euler_v2(w_k_1)
            mix, w_k = self.euler_v3( w_k )

        else:
            w_k_2 = self.map2(F.relu(self.bnh2(x2))) if hasattr(self, 'map2') else x2
            w_k_1 = self.map1(F.relu(self.bnh2(x1))) if hasattr(self, 'map1') else x1
            mix, w_k = self.euler_h3( x0 )

        out = mix - w_k_1 + 0.5*w_k_2
        return out, w_k, w_k_1

#pahse 4
# 18 对比res18
class T22ResNet18( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T22ResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 64, 128 )
        self.taylor3 = block( sub_block, 128, 128, stride=1 )

        self.taylor4 = block( sub_block, 128, 256 )

        self.taylor6 = block( sub_block, 256, 512 )
        self.taylor7 = block( sub_block, 512, 512, stride=1 )

        self.linear = nn.Linear( 512, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor01(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 22 对比preres18
class T22ResNet22( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T22ResNet22, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 64, 128 )
        self.taylor3 = block( sub_block, 128, 128, stride=1 )
        self.taylor = block( sub_block, 128, 128, stride=1 )
        self.taylor01 = block( sub_block, 128, 128, stride=1 )

        self.taylor4 = block( sub_block, 128, 256 )

        self.taylor6 = block( sub_block, 256, 512 )
        self.taylor7 = block( sub_block, 512, 512, stride=1 )

        self.linear = nn.Linear( 512, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor3(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor01(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 36 对比res34
class T22ResNet36( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T22ResNet36, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor4 = block( sub_block, 64, 128 )
        self.taylor5 = block( sub_block, 128, 128, stride=1 )
        self.taylor6 = block( sub_block, 128, 128, stride=1 )
        self.taylor7 = block( sub_block, 128, 128, stride=1 )
        self.taylor = block( sub_block, 128, 128, stride=1 )
        self.taylor01 = block( sub_block, 128, 128, stride=1 )
        
        self.taylor8 = block( sub_block, 128, 256 )
        self.taylor9 = block( sub_block, 256, 256, stride=1 )
        self.taylor10 = block( sub_block, 256, 256, stride=1 )
        self.taylor11 = block( sub_block, 256, 256, stride=1 )
        self.taylor12 = block( sub_block, 256, 256, stride=1 )
        
        self.taylor13 = block( sub_block, 256, 512 )
        self.taylor14 = block( sub_block, 512, 512, stride=1 )
        self.taylor15 = block( sub_block, 512, 512, stride=1 )

        self.linear = nn.Linear( 512, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor01(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor8(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor9(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor10(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor11(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor12(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor13(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor14(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor15(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 44 对比res50
class T22ResNet44( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T22ResNet44, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor4 = block( sub_block, 256, 128 )
        self.taylor5 = block( sub_block, 512, 128, stride=1 )
        self.taylor6 = block( sub_block, 512, 128, stride=1 )
        self.taylor7 = block( sub_block, 512, 128, stride=1 )
        
        self.taylor8 = block( sub_block, 512, 256 )
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

# 86 对比res101
class T22ResNet86( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T22ResNet86, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor4 = block( sub_block, 256, 128 )
        self.taylor5 = block( sub_block, 512, 128, stride=1 )
        self.taylor6 = block( sub_block, 512, 128, stride=1 )
        self.taylor7 = block( sub_block, 512, 128, stride=1 )
        
        self.taylor8 = block( sub_block, 512, 256 )
        self.taylor9 = block( sub_block, 1024, 256, stride=1 )
        self.taylor10 = block( sub_block, 1024, 256, stride=1 )
        self.taylor11 = block( sub_block, 1024, 256, stride=1 )
        self.taylor12 = block( sub_block, 1024, 256, stride=1 )
        self.taylor13 = block( sub_block, 1024, 256, stride=1 )
        self.taylor14 = block( sub_block, 1024, 256, stride=1 )
        self.taylor15 = block( sub_block, 1024, 256, stride=1 )
        self.taylor16 = block( sub_block, 1024, 256, stride=1 )
        self.taylor17 = block( sub_block, 1024, 256, stride=1 )
        self.taylor18 = block( sub_block, 1024, 256, stride=1 )
        self.taylor19 = block( sub_block, 1024, 256, stride=1 )
        self.taylor20 = block( sub_block, 1024, 256, stride=1 )
        self.taylor21 = block( sub_block, 1024, 256, stride=1 )
        self.taylor22 = block( sub_block, 1024, 256, stride=1 )
        self.taylor23 = block( sub_block, 1024, 256, stride=1 )
        self.taylor24 = block( sub_block, 1024, 256, stride=1 )
        self.taylor25 = block( sub_block, 1024, 256, stride=1 )
        
        self.taylor26 = block( sub_block, 1024, 512 )
        self.taylor27 = block( sub_block, 2048, 512, stride=1 )
        self.taylor28 = block( sub_block, 2048, 512, stride=1 )

        self.linear = nn.Linear( 2048, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
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
        w_k_1, w_k, out = self.taylor18(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor19(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor20(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor21(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor22(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor23(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor24(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor25(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor26(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor27(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor28(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 137 对比res152
class T22ResNet137( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T22ResNet137, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor4 = block( sub_block, 256, 128 )
        self.taylor5 = block( sub_block, 512, 128, stride=1 )
        self.taylor6 = block( sub_block, 512, 128, stride=1 )
        self.taylor7 = block( sub_block, 512, 128, stride=1 )
        self.taylor8 = block( sub_block, 512, 128, stride=1 )
        self.taylor9 = block( sub_block, 512, 128, stride=1 )
        self.taylor10 = block( sub_block, 512, 128, stride=1 )
        self.taylor11 = block( sub_block, 512, 128, stride=1 )
        
        self.taylor12 = block( sub_block, 512, 256 )
        self.taylor13 = block( sub_block, 1024, 256, stride=1 )
        self.taylor14 = block( sub_block, 1024, 256, stride=1 )
        self.taylor15 = block( sub_block, 1024, 256, stride=1 )
        self.taylor16 = block( sub_block, 1024, 256, stride=1 )
        self.taylor17 = block( sub_block, 1024, 256, stride=1 )
        self.taylor18 = block( sub_block, 1024, 256, stride=1 )
        self.taylor19 = block( sub_block, 1024, 256, stride=1 )
        self.taylor20 = block( sub_block, 1024, 256, stride=1 )
        self.taylor21 = block( sub_block, 1024, 256, stride=1 )
        self.taylor22 = block( sub_block, 1024, 256, stride=1 )
        self.taylor23 = block( sub_block, 1024, 256, stride=1 )
        self.taylor24 = block( sub_block, 1024, 256, stride=1 )
        self.taylor25 = block( sub_block, 1024, 256, stride=1 )
        self.taylor26 = block( sub_block, 1024, 256, stride=1 )
        self.taylor27 = block( sub_block, 1024, 256, stride=1 )
        self.taylor28 = block( sub_block, 1024, 256, stride=1 )
        self.taylor29 = block( sub_block, 1024, 256, stride=1 )
        self.taylor30 = block( sub_block, 1024, 256, stride=1 )
        self.taylor31 = block( sub_block, 1024, 256, stride=1 )
        self.taylor32 = block( sub_block, 1024, 256, stride=1 )
        self.taylor33 = block( sub_block, 1024, 256, stride=1 )
        self.taylor34 = block( sub_block, 1024, 256, stride=1 )
        self.taylor35 = block( sub_block, 1024, 256, stride=1 )
        self.taylor36 = block( sub_block, 1024, 256, stride=1 )
        self.taylor37 = block( sub_block, 1024, 256, stride=1 )
        self.taylor38 = block( sub_block, 1024, 256, stride=1 )
        self.taylor39 = block( sub_block, 1024, 256, stride=1 )
        self.taylor40 = block( sub_block, 1024, 256, stride=1 )
        self.taylor41 = block( sub_block, 1024, 256, stride=1 )
        self.taylor42 = block( sub_block, 1024, 256, stride=1 )
        
        self.taylor44 = block( sub_block, 1024, 512 )
        self.taylor45 = block( sub_block, 2048, 512, stride=1 )
        self.taylor46 = block( sub_block, 2048, 512, stride=1 )

        self.linear = nn.Linear( 2048, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
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
        w_k_1, w_k, out = self.taylor18(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor19(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor20(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor21(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor22(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor23(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor24(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor25(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor26(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor27(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor28(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor29(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor30(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor31(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor32(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor33(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor34(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor35(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor36(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor37(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor38(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor39(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor40(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor41(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor42(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor44(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor45(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor46(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 188 对比res200
class T22ResNet188( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T22ResNet188, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor4 = block( sub_block, 256, 128 )
        self.taylor5 = block( sub_block, 512, 128, stride=1 )
        self.taylor6 = block( sub_block, 512, 128, stride=1 )
        self.taylor7 = block( sub_block, 512, 128, stride=1 )
        self.taylor8 = block( sub_block, 512, 128, stride=1 )
        self.taylor9 = block( sub_block, 512, 128, stride=1 )
        self.taylor10 = block( sub_block, 512, 128, stride=1 )
        self.taylor11 = block( sub_block, 512, 128, stride=1 )
        self.taylor12 = block( sub_block, 512, 128, stride=1 )
        self.taylor13 = block( sub_block, 512, 128, stride=1 )
        self.taylor14 = block( sub_block, 512, 128, stride=1 )
        self.taylor15 = block( sub_block, 512, 128, stride=1 )
        
        self.taylor16 = block( sub_block, 512, 256 )
        self.taylor17 = block( sub_block, 1024, 256, stride=1 )
        self.taylor18 = block( sub_block, 1024, 256, stride=1 )
        self.taylor19 = block( sub_block, 1024, 256, stride=1 )
        self.taylor20 = block( sub_block, 1024, 256, stride=1 )
        self.taylor21 = block( sub_block, 1024, 256, stride=1 )
        self.taylor22 = block( sub_block, 1024, 256, stride=1 )
        self.taylor23 = block( sub_block, 1024, 256, stride=1 )
        self.taylor24 = block( sub_block, 1024, 256, stride=1 )
        self.taylor25 = block( sub_block, 1024, 256, stride=1 )
        self.taylor26 = block( sub_block, 1024, 256, stride=1 )
        self.taylor27 = block( sub_block, 1024, 256, stride=1 )
        self.taylor28 = block( sub_block, 1024, 256, stride=1 )
        self.taylor29 = block( sub_block, 1024, 256, stride=1 )
        self.taylor30 = block( sub_block, 1024, 256, stride=1 )
        self.taylor31 = block( sub_block, 1024, 256, stride=1 )
        self.taylor32 = block( sub_block, 1024, 256, stride=1 )
        self.taylor33 = block( sub_block, 1024, 256, stride=1 )
        self.taylor34 = block( sub_block, 1024, 256, stride=1 )
        self.taylor35 = block( sub_block, 1024, 256, stride=1 )
        self.taylor36 = block( sub_block, 1024, 256, stride=1 )
        self.taylor37 = block( sub_block, 1024, 256, stride=1 )
        self.taylor38 = block( sub_block, 1024, 256, stride=1 )
        self.taylor39 = block( sub_block, 1024, 256, stride=1 )
        self.taylor40 = block( sub_block, 1024, 256, stride=1 )
        self.taylor41 = block( sub_block, 1024, 256, stride=1 )
        self.taylor42 = block( sub_block, 1024, 256, stride=1 )
        self.taylor43 = block( sub_block, 1024, 256, stride=1 )
        self.taylor44 = block( sub_block, 1024, 256, stride=1 )
        self.taylor45 = block( sub_block, 1024, 256, stride=1 )
        self.taylor46 = block( sub_block, 1024, 256, stride=1 )
        self.taylor47 = block( sub_block, 1024, 256, stride=1 )
        self.taylor48 = block( sub_block, 1024, 256, stride=1 )
        self.taylor49 = block( sub_block, 1024, 256, stride=1 )
        self.taylor50 = block( sub_block, 1024, 256, stride=1 )
        self.taylor51 = block( sub_block, 1024, 256, stride=1 )
        self.taylor52 = block( sub_block, 1024, 256, stride=1 )
        self.taylor53 = block( sub_block, 1024, 256, stride=1 )
        self.taylor54 = block( sub_block, 1024, 256, stride=1 )
        self.taylor55 = block( sub_block, 1024, 256, stride=1 )
        self.taylor56 = block( sub_block, 1024, 256, stride=1 )
        self.taylor57 = block( sub_block, 1024, 256, stride=1 )
        self.taylor58 = block( sub_block, 1024, 256, stride=1 )
        self.taylor59 = block( sub_block, 1024, 256, stride=1 )
        
        self.taylor60 = block( sub_block, 1024, 512 )
        self.taylor61 = block( sub_block, 2048, 512, stride=1 )
        self.taylor62 = block( sub_block, 2048, 512, stride=1 )

        self.linear = nn.Linear( 2048, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
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
        w_k_1, w_k, out = self.taylor18(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor19(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor20(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor21(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor22(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor23(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor24(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor25(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor26(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor27(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor28(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor29(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor30(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor31(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor32(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor33(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor34(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor35(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor36(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor37(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor38(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor39(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor40(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor41(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor42(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor44(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor45(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor46(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor47(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor48(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor49(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor50(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor51(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor52(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor53(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor54(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor55(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor56(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor57(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor58(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor59(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor60(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor61(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor62(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class T22ResNet275( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T22ResNet275, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor4 = block( sub_block, 256, 128 )
        self.taylor5 = block( sub_block, 512, 128, stride=1 )
        self.taylor6 = block( sub_block, 512, 128, stride=1 )
        self.taylor7 = block( sub_block, 512, 128, stride=1 )
        self.taylor8 = block( sub_block, 512, 128, stride=1 )
        self.taylor9 = block( sub_block, 512, 128, stride=1 )
        self.taylor10 = block( sub_block, 512, 128, stride=1 )
        self.taylor11 = block( sub_block, 512, 128, stride=1 )
        self.taylor12 = block( sub_block, 512, 128, stride=1 )
        self.taylor13 = block( sub_block, 512, 128, stride=1 )
        self.taylor14 = block( sub_block, 512, 128, stride=1 )
        self.taylor15 = block( sub_block, 512, 128, stride=1 )
        self.taylor16 = block( sub_block, 512, 128, stride=1 )
        self.taylor17 = block( sub_block, 512, 128, stride=1 )
        self.taylor18 = block( sub_block, 512, 128, stride=1 )
        self.taylor19 = block( sub_block, 512, 128, stride=1 )
        self.taylor = block( sub_block, 512, 128, stride=1 )
        self.taylor01 = block( sub_block, 512, 128, stride=1 )
        
        self.taylor20 = block( sub_block, 512, 256 )
        self.taylor21 = block( sub_block, 1024, 256, stride=1 )
        self.taylor22 = block( sub_block, 1024, 256, stride=1 )
        self.taylor23 = block( sub_block, 1024, 256, stride=1 )
        self.taylor24 = block( sub_block, 1024, 256, stride=1 )
        self.taylor25 = block( sub_block, 1024, 256, stride=1 )
        self.taylor26 = block( sub_block, 1024, 256, stride=1 )
        self.taylor27 = block( sub_block, 1024, 256, stride=1 )
        self.taylor28 = block( sub_block, 1024, 256, stride=1 )
        self.taylor29 = block( sub_block, 1024, 256, stride=1 )
        self.taylor30 = block( sub_block, 1024, 256, stride=1 )
        self.taylor31 = block( sub_block, 1024, 256, stride=1 )
        self.taylor32 = block( sub_block, 1024, 256, stride=1 )
        self.taylor33 = block( sub_block, 1024, 256, stride=1 )
        self.taylor34 = block( sub_block, 1024, 256, stride=1 )
        self.taylor35 = block( sub_block, 1024, 256, stride=1 )
        self.taylor36 = block( sub_block, 1024, 256, stride=1 )
        self.taylor37 = block( sub_block, 1024, 256, stride=1 )
        self.taylor38 = block( sub_block, 1024, 256, stride=1 )
        self.taylor39 = block( sub_block, 1024, 256, stride=1 )
        self.taylor40 = block( sub_block, 1024, 256, stride=1 )
        self.taylor41 = block( sub_block, 1024, 256, stride=1 )
        self.taylor42 = block( sub_block, 1024, 256, stride=1 )
        self.taylor43 = block( sub_block, 1024, 256, stride=1 )
        self.taylor44 = block( sub_block, 1024, 256, stride=1 )
        self.taylor45 = block( sub_block, 1024, 256, stride=1 )
        self.taylor46 = block( sub_block, 1024, 256, stride=1 )
        self.taylor47 = block( sub_block, 1024, 256, stride=1 )
        self.taylor48 = block( sub_block, 1024, 256, stride=1 )
        self.taylor49 = block( sub_block, 1024, 256, stride=1 )
        self.taylor50 = block( sub_block, 1024, 256, stride=1 )
        self.taylor51 = block( sub_block, 1024, 256, stride=1 )
        self.taylor52 = block( sub_block, 1024, 256, stride=1 )
        self.taylor53 = block( sub_block, 1024, 256, stride=1 )
        self.taylor54 = block( sub_block, 1024, 256, stride=1 )
        self.taylor55 = block( sub_block, 1024, 256, stride=1 )
        self.taylor56 = block( sub_block, 1024, 256, stride=1 )
        self.taylor57 = block( sub_block, 1024, 256, stride=1 )
        self.taylor58 = block( sub_block, 1024, 256, stride=1 )
        self.taylor59 = block( sub_block, 1024, 256, stride=1 )
        self.taylor60 = block( sub_block, 1024, 256, stride=1 )
        self.taylor61 = block( sub_block, 1024, 256, stride=1 )
        self.taylor62 = block( sub_block, 1024, 256, stride=1 )
        self.taylor63 = block( sub_block, 1024, 256, stride=1 )
        self.taylor64 = block( sub_block, 1024, 256, stride=1 )
        self.taylor65 = block( sub_block, 1024, 256, stride=1 )
        self.taylor66 = block( sub_block, 1024, 256, stride=1 )
        self.taylor67 = block( sub_block, 1024, 256, stride=1 )
        self.taylor68 = block( sub_block, 1024, 256, stride=1 )
        self.taylor69 = block( sub_block, 1024, 256, stride=1 )
        self.taylor70 = block( sub_block, 1024, 256, stride=1 )
        self.taylor71 = block( sub_block, 1024, 256, stride=1 )
        self.taylor72 = block( sub_block, 1024, 256, stride=1 )
        self.taylor73 = block( sub_block, 1024, 256, stride=1 )
        self.taylor74 = block( sub_block, 1024, 256, stride=1 )
        self.taylor02 = block( sub_block, 1024, 256, stride=1 )
        self.taylor03 = block( sub_block, 1024, 256, stride=1 )
        self.taylor04 = block( sub_block, 1024, 256, stride=1 )
        self.taylor05 = block( sub_block, 1024, 256, stride=1 )
        self.taylor06 = block( sub_block, 1024, 256, stride=1 )
        self.taylor07 = block( sub_block, 1024, 256, stride=1 )
        self.taylor08 = block( sub_block, 1024, 256, stride=1 )
        self.taylor09 = block( sub_block, 1024, 256, stride=1 )
        self.taylor010 = block( sub_block, 1024, 256, stride=1 )
        self.taylor011 = block( sub_block, 1024, 256, stride=1 )
        self.taylor012 = block( sub_block, 1024, 256, stride=1 )
        
        self.taylor75 = block( sub_block, 1024, 512 )
        self.taylor76 = block( sub_block, 2048, 512, stride=1 )
        self.taylor77 = block( sub_block, 2048, 512, stride=1 )
        self.taylor78 = block( sub_block, 2048, 512, stride=1 )

        self.linear = nn.Linear( 2048, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
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
        w_k_1, w_k, out = self.taylor18(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor19(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor01(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor20(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor21(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor22(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor23(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor24(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor25(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor26(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor27(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor28(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor29(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor30(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor31(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor32(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor33(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor34(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor35(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor36(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor37(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor38(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor39(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor40(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor41(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor42(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor44(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor45(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor46(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor47(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor48(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor49(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor50(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor51(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor52(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor53(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor54(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor55(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor56(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor57(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor58(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor59(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor60(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor61(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor62(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor63(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor64(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor65(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor66(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor66(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor67(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor68(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor69(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor70(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor71(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor72(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor73(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor74(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor02(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor03(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor04(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor05(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor06(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor07(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor08(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor09(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor010(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor011(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor012(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor75(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor76(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor77(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor78(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def T22_ResNet18():
    return T22ResNet18( TaylorBlock, BasicBlock, num_classes=100 )

def T22_ResNet22():
    return T22ResNet22( TaylorBlock, BasicBlock, num_classes=100 )

def T22_ResNet36():
    return T22ResNet36( TaylorBlock, BasicBlock, num_classes=100 )

def T22_ResNet44():
    return T22ResNet44( TaylorBlock, Bottleneck,  num_classes=100 )

def T22_ResNet86():
    return T22ResNet86( TaylorBlock, Bottleneck,  num_classes=100 )

def T22_ResNet137():
    return T22ResNet137( TaylorBlock, Bottleneck,  num_classes=100 )

def T22_ResNet188():
    return T22ResNet188( TaylorBlock, Bottleneck,  num_classes=100 )

def T22_ResNet275():
    return T22ResNet275( TaylorBlock, Bottleneck,  num_classes=100 )


# def show_param():
#     model_name = "T22_ResNet275"
#     net = T22_ResNet275()
#     net = net.cuda()
#     # stat(net, (3,32,32))
#     summary(net, (3,32,32))

#     print( model_name )

# def test():
    # net = T22_ResNet137()
    # model_name = 'T22_ResNet137'

    # y = net(torch.randn(1,3,32,32))

    # print( model_name )
    # print(y.shape)

# test()
# show_param()
# profile()


# %%