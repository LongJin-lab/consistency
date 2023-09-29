# %% cifar100 phase4
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
        im = self.shortcut(x)
        if self.is_normal:
            out += im
        else:
            out += 1.5*im
        # pahse 4
        # out = F.relu(out)

        #phase 5
        return out, im

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
        im = self.shortcut(x)
        if self.is_normal:
            out += im
        else:
            out += 1.5*im
        # pahse 4
        #out = F.relu(out)

        # pahse 5
        return out, im

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

            #phase 4
            # self.out_bn1 = nn.BatchNorm2d( block.expansion*planes )
            # self.out_bn2 = nn.BatchNorm2d( block.expansion*planes )
            # self.out_bn3 = nn.BatchNorm2d( block.expansion*planes )

        else:
            self.euler_h3 = self._make_layer( block, planes, 1, stride=stride, is_normal=False )
            self.imap2 = nn.Sequential()
            self.imap1 = nn.Sequential()

            if stride != 1 or in_planes != block.expansion*planes:
                self.imap2 = nn.Sequential(
                    nn.Conv2d(in_planes, block.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.expansion*planes)
                )

                self.imap1 = nn.Sequential(
                    nn.Conv2d(in_planes, block.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.expansion*planes)
                )
            
            #phase 6
            # self.out_bn1 = nn.BatchNorm2d( block.expansion*planes )
            # self.out_bn2 = nn.BatchNorm2d( block.expansion*planes )
            # self.out_bn3 = nn.BatchNorm2d( block.expansion*planes )

        # pahse 6
        # self.out_conv1 = nn.Conv2d(block.expansion*planes, block.expansion*planes,
        #                     kernel_size=1, stride=1, bias=False)
        # self.out_conv2 = nn.Conv2d(block.expansion*planes, block.expansion*planes,
        #                 kernel_size=1, stride=1, bias=False)
        # self.out_conv3 = nn.Conv2d(block.expansion*planes, block.expansion*planes,
        #                     kernel_size=1, stride=1, bias=False)

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
                w_k_2 = F.relu( w_k_2 )
            else:
                w_k_2 = x0
            w_k_1, _ = self.euler_v1(w_k_2)
            w_k, _ = self.euler_v2(w_k_1)
            grad, _ = self.euler_v3( w_k )

            #phase 4
            # out = self.out_bn1(-w_k_1) + self.out_bn2((0.5)*w_k_2) + self.out_bn3(grad)

            # pahse 5
            out = -w_k + 0.5*w_k_1 + grad
            out = F.relu( out )

            #phase 6
            # w_k_1 = self.out_conv1(F.relu( w_k_1 ))
            # w_k = self.out_conv2(F.relu( w_k ))
            # grad = self.out_conv3(F.relu( grad ))
            # out = (-w_k_1) + ((0.5)*w_k_2) + (grad)
            return w_k_1, w_k, out
        else:
            w_k_1 = self.imap1( x0 )
            w_k = self.imap2( x1 )
            grad, im = self.euler_h3( x2 )

            # pahse 4
            # out = -w_k + 0.5*w_k_1 + grad

            # pahse 5
            out = -w_k + 0.5*w_k_1 + grad
            out = F.relu( out )

            #pahse 6
            # w_k_1 = self.out_conv1(F.relu( self.out_bn1(w_k_1) ))
            # w_k = self.out_conv2(F.relu( self.out_bn2(w_k) ))
            # grad = self.out_conv3(F.relu( self.out_bn3(grad) ))
            # out = -w_k + 0.5*w_k_1 + grad

            return w_k, im, out

#pahse 4
# 16 对比res18
class T2ResNet16( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T2ResNet16, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor2 = block( sub_block, 64, 128 )

        self.taylor4 = block( sub_block, 128, 256 )
        # self.taylor5 = block( sub_block, 256, 256, stride=1 )

        self.taylor6 = block( sub_block, 256, 512 )
        self.taylor7 = block( sub_block, 512, 512, stride=1 )

        self.linear = nn.Linear( 512, num_classes )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        w_k_1, w_k, out = self.taylor1(out)
        w_k_1, w_k, out = self.taylor2(w_k_1, w_k, out)
        # w_k_1, w_k, out = self.taylor3(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor4(w_k_1, w_k, out)
        # w_k_1, w_k, out = self.taylor5(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor6(w_k_1, w_k, out)
        w_k_1, w_k, out = self.taylor7(w_k_1, w_k, out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 32 对比res34
class T2ResNet34( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T2ResNet34, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.taylor1 = block( sub_block, 64, 64, stride=1, is_bootblock=True )
        
        self.taylor4 = block( sub_block, 64, 128 )
        self.taylor5 = block( sub_block, 128, 128, stride=1 )
        self.taylor6 = block( sub_block, 128, 128, stride=1 )
        self.taylor7 = block( sub_block, 128, 128, stride=1 )
        self.taylor = block( sub_block, 128, 128, stride=1 )
        
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

# 48 对比res50
class T2ResNet44( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T2ResNet44, self).__init__()
        
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
class T2ResNet86( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T2ResNet86, self).__init__()
        
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
class T2ResNet137( nn.Module ):
    def __init__( self, block, sub_block, num_classes=10):
        super( T2ResNet137, self).__init__()
        
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

# TODO pahse 6 V2 结构

def T2_ResNet16():
    return T2ResNet16( TaylorBlock, BasicBlock, num_classes=100 )

def T2_ResNet34():
    return T2ResNet34( TaylorBlock, BasicBlock, num_classes=100 )

def T2_ResNet44():
    return T2ResNet44( TaylorBlock, Bottleneck,  num_classes=100 )

def T2_ResNet86():
    return T2ResNet86( TaylorBlock, Bottleneck,  num_classes=100 )

def T2_ResNet137():
    return T2ResNet137( TaylorBlock, Bottleneck,  num_classes=100 )


# def show_param():
#     model_name = "T2_ResNet137"
#     net = T2_ResNet137()
#     net = net.cuda()
#     # stat(net, (3,32,32))
#     summary(net, (3,32,32))

#     print( model_name )

# def test():
#     net = T2_ResNet16()
#     model_name = 'T2_ResNet'

#     y = net(torch.randn(1,3,32,32))

#     print( model_name )
#     print(y.shape)

# test()
# show_param()
# profile()


# %%