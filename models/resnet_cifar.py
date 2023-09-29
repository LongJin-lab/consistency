import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init




class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class AB2BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k1=1.0, k2=1.0, option='B'):
        super(AB2BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.k1=k1
        self.k2=k2
        
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut1 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut2 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        yn,fyn_1 = x
        fyn = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(yn)))))
        yn1 = self.k1*fyn
        yn1 += self.shortcut1(yn)
        yn1 += self.k2*self.shortcut2(fyn_1)
        yn1 = F.relu(yn1)
        x=[yn1,fyn]
        return x

class ABL2BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(ABL2BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.h = nn.Parameter(torch.tensor([1.0], device='cuda'))
        self.k1= nn.Parameter(torch.tensor([0.05], device='cuda'))
        
        
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut1 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut2 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        yn,fyn_1 = x
        fyn = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(yn)))))
        yn1 = self.h*self.k1*fyn
        yn1 += self.shortcut1(yn)
        yn1 += self.h*(1-self.k1)*self.shortcut2(fyn_1)
        yn1 = F.relu(yn1)
        x=[yn1,fyn]
        return x

class ABLL2BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(ABLL2BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.h = nn.Parameter(torch.tensor([1.0], device='cuda'))
        self.k1= nn.Parameter(torch.tensor([0.05], device='cuda'))
        
        
        self.shortcut1 = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut1 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, x):
        yn,fyn_1,flag = x
        if flag==0:
            fyn = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(yn)))))
            yn1 = self.shortcut1(yn) + self.h*fyn 
            yn1 = F.relu(yn1)
            flag=1
            x=[yn1,fyn,flag]
            return x
            
        else:
            fyn = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(yn)))))
            yn1 = self.h*self.k1*fyn
            yn1 += self.shortcut1(yn)
            yn1 += self.h*(1-self.k1)*fyn_1
            yn1 = F.relu(yn1)
            x=[yn1,fyn,flag]
            return x


class AB2BasicBlock_A(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k1=1.0, k2=1.0, option='B'):
        super(AB2BasicBlock_A, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.k1=k1
        self.k2=k2
        
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut1 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut2 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        yn,fyn_1 = x
        fyn = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(yn))))))
        yn1 = self.k1*fyn
        yn1 += self.shortcut1(yn)
        yn1 += self.k2*self.shortcut2(fyn_1)
        x=[yn1,fyn]
        return x

class AB2BasicBlock_B(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k1=1.0, k2=1.0, option='B'):
        super(AB2BasicBlock_B, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.k1=k1
        self.k2=k2
        
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut1 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut2 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        yn,fyn_1 = x
        fyn = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(F.relu(yn))))))
        yn1 = self.k1*fyn
        yn1 += self.shortcut1(yn)
        yn1 += self.k2*self.shortcut2(fyn_1)
        x=[yn1,fyn]
        return x

class AB2BasicBlock_C(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k1=1.0, k2=1.0, option='B'):
        super(AB2BasicBlock_C, self).__init__()
        if stride == 1:
            self.bn1 = nn.BatchNorm2d(planes)
        if stride == 2:
            self.bn1 = nn.BatchNorm2d(int(planes/2))
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        
        self.k1=k1
        self.k2=k2
        
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut1 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut2 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        yn,fyn_1 = x
        
        fyn = self.conv2(F.relu(self.bn2(self.conv1(F.relu(self.bn1(yn))))))
        # shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        yn1 = self.k1*fyn
        yn1 += self.shortcut1(yn)
        yn1 += self.k2*self.shortcut2(fyn_1)
        x=[yn1,fyn]
        return x




class PREAB2ResNet(nn.Module):
    def __init__(self, block, num_blocks, h=1.0, b1=1.0, b2=1.0,num_classes=10):
        super(PREAB2ResNet, self).__init__()
        self.in_planes = 16
        self.k1=b1*h
        self.k2=b2*h
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        #self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.k1, self.k2))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        yn = F.relu(self.bn1(self.conv1(x)))
        out=[yn,0]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        yn = out[0]
        out = F.avg_pool2d(yn, yn.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class AB2ResNet(nn.Module):
    def __init__(self, block, num_blocks, h=1.0, b1=1.0, b2=1.0,num_classes=10):
        super(AB2ResNet, self).__init__()
        self.in_planes = 16
        self.k1=b1*h
        self.k2=b2*h
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        #self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.k1, self.k2))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        yn = F.relu(self.bn1(self.conv1(x)))
        out=[yn,0]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        yn = out[0]
        out = F.avg_pool2d(yn, yn.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class ABL2ResNet(nn.Module):
    def __init__(self, block, num_blocks,num_classes=10):
        super(ABL2ResNet, self).__init__()
        self.in_planes = 16
        
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        #self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        yn = F.relu(self.bn1(self.conv1(x)))
        out=[yn,0]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        yn = out[0]
        out = F.avg_pool2d(yn, yn.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ABLL2ResNet(nn.Module):
    def __init__(self, block, num_blocks,num_classes=10):
        super(ABLL2ResNet, self).__init__()
        self.in_planes = 16
        
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        #self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        yn = F.relu(self.bn1(self.conv1(x)))
        out=[yn,0,0]
        out = self.layer1(out)
        out=[out[0],0,0]
        out = self.layer2(out)
        out=[out[0],0,0]
        out = self.layer3(out)
        yn = out[0]
        out = F.avg_pool2d(yn, yn.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
     

class AB2BasicBlock_F(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k1=1.0, k2=1.0, option='B'):
        super(AB2BasicBlock_F, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.k1=k1
        self.k2=k2
        
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut1 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut2 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        yn,fyn_1,result = x
        fyn=self.conv1(yn)
        result.append(fyn)
        fyn=self.conv2(F.relu(self.bn1(fyn)))
        result.append(fyn)
        fyn = self.bn2(fyn)
        # fyn = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(yn)))))
        
        yn1 = self.k1*fyn
        yn1 += self.shortcut1(yn)
        yn1 += self.k2*self.shortcut2(fyn_1)
        yn1 = F.relu(yn1)
        x=[yn1,fyn,result]
        return x    
     
class AB2ResNet_Feather(nn.Module):
    def __init__(self, block, num_blocks, h=1.0, b1=1.0, b2=1.0,num_classes=10):
        super(AB2ResNet_Feather, self).__init__()
        self.in_planes = 16
        self.k1=b1*h
        self.k2=b2*h
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        
        self.result=[]

        #self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.k1, self.k2))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        result=[]
        yn=self.conv1(x)
        result.append(yn)
        yn = F.relu(self.bn1(yn))
        out=[yn,0,result]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        self.result = out[2]
        yn = out[0]
        out = F.avg_pool2d(yn, yn.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class AB2BasicBlock_F1(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k1=1.0, k2=1.0, option='B'):
        super(AB2BasicBlock_F1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.k1=k1
        self.k2=k2
        
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut1 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut2 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        yn,fyn_1,result = x
        fyn=self.conv1(yn)
        fyn=F.relu(self.bn1(fyn))
        fyn=self.conv2(fyn)
        fyn = self.bn2(fyn)
        # fyn = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(yn)))))
        
        yn1 = self.k1*fyn
        yn1 += self.shortcut1(yn)
        yn1 += self.k2*self.shortcut2(fyn_1)
        yn1 = F.relu(yn1)
        result.append(yn1)
        x=[yn1,fyn,result]
        return x    
     
class AB2ResNet_Feather1(nn.Module):
    def __init__(self, block, num_blocks, h=1.0, b1=1.0, b2=1.0,num_classes=10):
        super(AB2ResNet_Feather1, self).__init__()
        self.in_planes = 16
        self.k1=b1*h
        self.k2=b2*h
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        
        self.result=[]

        #self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.k1, self.k2))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        result=[]
        yn=self.conv1(x)
        yn = F.relu(self.bn1(yn))
        result.append(yn)
        out=[yn,0,result]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        self.result = out[2]
        yn = out[0]
        out = F.avg_pool2d(yn, yn.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def abl2resnet20():
    return ABL2ResNet(ABL2BasicBlock, [3, 3, 3])
def abl2resnet32():
    return ABL2ResNet(ABL2BasicBlock, [5, 5, 5])
def abl2resnet44():
    return ABL2ResNet(ABL2BasicBlock, [7, 7, 7])
def abl2resnet56():
    return ABL2ResNet(ABL2BasicBlock, [9, 9, 9])
def abl2resnet110():
    return ABL2ResNet(ABL2BasicBlock, [18, 18, 18])

def abll2resnet20():
    return ABLL2ResNet(ABLL2BasicBlock, [3, 3, 3])
def abll2resnet32():
    return ABLL2ResNet(ABLL2BasicBlock, [5, 5, 5])
def abll2resnet44():
    return ABLL2ResNet(ABLL2BasicBlock, [7, 7, 7])
def abll2resnet56():
    return ABLL2ResNet(ABLL2BasicBlock, [9, 9, 9])
def abll2resnet110():
    return ABLL2ResNet(ABLL2BasicBlock, [18, 18, 18])

  
def ab2resnet20(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [3, 3, 3], h, b1, b2)
def ab2resnet32(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [5, 5, 5], h, b1, b2)
def ab2resnet44(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [7, 7, 7], h, b1, b2)
def ab2resnet56(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [9, 9, 9], h, b1, b2)
def ab2resnet68(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [11, 11, 11], h, b1, b2)
def ab2resnet80(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [13, 13, 13], h, b1, b2)
def ab2resnet92(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [15, 15, 15], h, b1, b2)
def ab2resnet104(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [17, 17, 17], h, b1, b2)
def ab2resnet110(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [18, 18, 18], h, b1, b2)

def ab2resnet20_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [3, 3, 3], h, b1, b2)
def ab2resnet32_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [5, 5, 5], h, b1, b2)
def ab2resnet44_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [7, 7, 7], h, b1, b2)
def ab2resnet56_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [9, 9, 9], h, b1, b2)
def ab2resnet68_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [11, 11, 11], h, b1, b2)
def ab2resnet80_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [13, 13, 13], h, b1, b2)
def ab2resnet92_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [15, 15, 15], h, b1, b2)
def ab2resnet104_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [17, 17, 17], h, b1, b2)
def ab2resnet110_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [18, 18, 18], h, b1, b2)

def ab2resnet20_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [3, 3, 3], h, b1, b2)
def ab2resnet32_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [5, 5, 5], h, b1, b2)
def ab2resnet44_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [7, 7, 7], h, b1, b2)
def ab2resnet56_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [9, 9, 9], h, b1, b2)
def ab2resnet68_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [11, 11, 11], h, b1, b2)
def ab2resnet80_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [13, 13, 13], h, b1, b2)
def ab2resnet92_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [15, 15, 15], h, b1, b2)
def ab2resnet104_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [17, 17, 17], h, b1, b2)
def ab2resnet110_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [18, 18, 18], h, b1, b2)

def ab2resnet20_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [3, 3, 3], h, b1, b2)
def ab2resnet32_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [5, 5, 5], h, b1, b2)
def ab2resnet44_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [7, 7, 7], h, b1, b2)
def ab2resnet56_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [9, 9, 9], h, b1, b2)
def ab2resnet68_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [11, 11, 11], h, b1, b2)
def ab2resnet80_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [13, 13, 13], h, b1, b2)
def ab2resnet92_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [15, 15, 15], h, b1, b2)
def ab2resnet104_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [17, 17, 17], h, b1, b2)
def ab2resnet110_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [18, 18, 18], h, b1, b2)

def ab2resnet20_f(h, b1, b2):
    return AB2ResNet_Feather(AB2BasicBlock_F, [3, 3, 3], h, b1, b2)
def ab2resnet110_f(h, b1, b2):
    return AB2ResNet_Feather(AB2BasicBlock_F, [18, 18, 18], h, b1, b2)

def ab2resnet20_f1(h, b1, b2):
    return AB2ResNet_Feather1(AB2BasicBlock_F1, [3, 3, 3], h, b1, b2)
def ab2resnet110_f1(h, b1, b2):
    return AB2ResNet_Feather1(AB2BasicBlock_F1, [18, 18, 18], h, b1, b2)
'''----------------------------------------------------------------------------------------------------'''
def ab2res100net20(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [3, 3, 3], h, b1, b2,100)
def ab2res100net32(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [5, 5, 5], h, b1, b2,100)
def ab2res100net44(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [7, 7, 7], h, b1, b2,100)
def ab2res100net56(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [9, 9, 9], h, b1, b2,100)
def ab2res100net68(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [11, 11, 11], h, b1, b2,100)
def ab2res100net80(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [13, 13, 13], h, b1, b2,100)
def ab2res100net92(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [15, 15, 15], h, b1, b2,100)
def ab2res100net104(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [17, 17, 17], h, b1, b2,100)
def ab2res100net110(h, b1, b2):
    return AB2ResNet(AB2BasicBlock, [18, 18, 18], h, b1, b2,100)

def ab2res100net20_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [3, 3, 3], h, b1, b2,100)
def ab2res100net32_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [5, 5, 5], h, b1, b2,100)
def ab2res100net44_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [7, 7, 7], h, b1, b2,100)
def ab2res100net56_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [9, 9, 9], h, b1, b2,100)
def ab2res100net68_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [11, 11, 11], h, b1, b2,100)
def ab2res100net80_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [13, 13, 13], h, b1, b2,100)
def ab2res100net92_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [15, 15, 15], h, b1, b2,100)
def ab2res100net104_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [17, 17, 17], h, b1, b2,100)
def ab2res100net110_a(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_A, [18, 18, 18], h, b1, b2,100)

def ab2res100net20_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [3, 3, 3], h, b1, b2,100)
def ab2res100net32_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [5, 5, 5], h, b1, b2,100)
def ab2res100net44_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [7, 7, 7], h, b1, b2,100)
def ab2res100net56_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [9, 9, 9], h, b1, b2,100)
def ab2res100net68_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [11, 11, 11], h, b1, b2,100)
def ab2res100net80_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [13, 13, 13], h, b1, b2,100)
def ab2res100net92_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [15, 15, 15], h, b1, b2,100)
def ab2res100net104_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [17, 17, 17], h, b1, b2,100)
def ab2res100net110_b(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_B, [18, 18, 18], h, b1, b2,100)

def ab2res100net20_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [3, 3, 3], h, b1, b2,100)
def ab2res100net32_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [5, 5, 5], h, b1, b2,100)
def ab2res100net44_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [7, 7, 7], h, b1, b2,100)
def ab2res100net56_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [9, 9, 9], h, b1, b2,100)
def ab2res100net68_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [11, 11, 11], h, b1, b2,100)
def ab2res100net80_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [13, 13, 13], h, b1, b2,100)
def ab2res100net92_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [15, 15, 15], h, b1, b2,100)
def ab2res100net104_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [17, 17, 17], h, b1, b2,100)
def ab2res100net110_c(h, b1, b2):
    return AB2ResNet(AB2BasicBlock_C, [18, 18, 18], h, b1, b2,100)
'''def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)'''
        
class AB3BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k1=1.0, k2=1.0, k3=1,option='B'):
        super(AB3BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.k1=k1
        self.k2=k2
        self.k3=k3
        self.in_planes=in_planes
        self.planes=planes
        self.stride=stride
        
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        self.shortcut3 = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut1 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut2 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut3 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
        self.shortcut4 = nn.Sequential(
            #平均池化下采样
            # nn.AvgPool2d(2),
            # nn.Conv2d(int(in_planes/2), planes, kernel_size=1, stride=1, bias=False),
            #max下采样
            # nn.MaxPool2d(2),
            # nn.Conv2d(int(in_planes/2), planes, kernel_size=1, stride=1, bias=False),
            #conv下采样
            nn.Conv2d(int(in_planes/2), planes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        yn,fyn_1,fyn_2 = x
        fyn = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(yn)))))
        yn1 = self.k1*fyn
        yn1 += self.shortcut1(yn)
        yn1 += self.k2*self.shortcut2(fyn_1)
        if type(fyn_2)==type(yn1):
            if ((self.stride == 1 and self.in_planes == self.planes) and (yn1.size()!= fyn_2.size())):
                yn1 += self.k3*self.shortcut4(fyn_2)
                yn1 = F.relu(yn1)
                x=[yn1,fyn,fyn_1]
                return x
        yn1 += self.k3*self.shortcut3(fyn_2)
        yn1 = F.relu(yn1)
        x=[yn1,fyn,fyn_1]
        return x


class AB3ResNet(nn.Module):
    def __init__(self, block, num_blocks, h=1.0, b1=1.0, b2=1.0, b3=1.0, num_classes=10):
        super(AB3ResNet, self).__init__()
        self.in_planes = 16
        self.k1=b1*h
        self.k2=b2*h
        self.k3=b3*h
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        #self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.k1, self.k2, self.k3))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        yn = F.relu(self.bn1(self.conv1(x)))
        out=[yn,0,0]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        yn = out[0]
        out = F.avg_pool2d(yn, yn.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class AB3BasicBlock_F(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k1=1.0, k2=1.0, k3=1,option='B'):
        super(AB3BasicBlock_F, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.k1=k1
        self.k2=k2
        self.k3=k3
        self.in_planes=in_planes
        self.planes=planes
        self.stride=stride
        
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        self.shortcut3 = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut1 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut2 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut3 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
        self.shortcut4 = nn.Sequential(
            #平均池化下采样
            # nn.AvgPool2d(2),
            # nn.Conv2d(int(in_planes/2), planes, kernel_size=1, stride=1, bias=False),
            #max下采样
            # nn.MaxPool2d(2),
            # nn.Conv2d(int(in_planes/2), planes, kernel_size=1, stride=1, bias=False),
            #conv下采样
            nn.Conv2d(int(in_planes/2), planes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        yn,fyn_1,fyn_2,result = x
        fyn=self.conv1(yn)
        result.append(fyn)
        fyn=self.conv2(F.relu(self.bn1(fyn)))
        result.append(fyn)
        fyn = self.bn2(fyn)
        # fyn = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(yn)))))
        
        yn1 = self.k1*fyn
        yn1 += self.shortcut1(yn)
        yn1 += self.k2*self.shortcut2(fyn_1)
        if type(fyn_2)==type(yn1):
            if ((self.stride == 1 and self.in_planes == self.planes) and (yn1.size()!= fyn_2.size())):
                yn1 += self.k3*self.shortcut4(fyn_2)
                yn1 = F.relu(yn1)
                x=[yn1,fyn,fyn_1,result]
                return x
        yn1 += self.k3*self.shortcut3(fyn_2)
        yn1 = F.relu(yn1)
        x=[yn1,fyn,fyn_1,result]
        return x
    



class AB3ResNet_Feather(nn.Module):
    def __init__(self, block, num_blocks, h=1.0, b1=1.0, b2=1.0, b3=1.0, num_classes=10):
        super(AB3ResNet_Feather, self).__init__()
        self.in_planes = 16
        self.k1=b1*h
        self.k2=b2*h
        self.k3=b3*h
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        
        self.result=[]

        #self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.k1, self.k2, self.k3))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        result=[]
        yn=self.conv1(x)
        result.append(yn)
        yn = F.relu(self.bn1(yn))
        out=[yn,0,0,result]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        self.result = out[3]
        yn = out[0]
        out = F.avg_pool2d(yn, yn.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class AB3BasicBlock_F1(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k1=1.0, k2=1.0, k3=1,option='B'):
        super(AB3BasicBlock_F1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.k1=k1
        self.k2=k2
        self.k3=k3
        self.in_planes=in_planes
        self.planes=planes
        self.stride=stride
        
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        self.shortcut3 = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut1 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut2 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.shortcut3 = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
        self.shortcut4 = nn.Sequential(
            #平均池化下采样
            # nn.AvgPool2d(2),
            # nn.Conv2d(int(in_planes/2), planes, kernel_size=1, stride=1, bias=False),
            #max下采样
            # nn.MaxPool2d(2),
            # nn.Conv2d(int(in_planes/2), planes, kernel_size=1, stride=1, bias=False),
            #conv下采样
            nn.Conv2d(int(in_planes/2), planes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        yn,fyn_1,fyn_2,result = x
        fyn=self.conv1(yn)
        fyn=self.conv2(F.relu(self.bn1(fyn)))
        fyn = self.bn2(fyn)
        # fyn = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(yn)))))
        
        yn1 = self.k1*fyn
        yn1 += self.shortcut1(yn)
        yn1 += self.k2*self.shortcut2(fyn_1)
        if type(fyn_2)==type(yn1):
            if ((self.stride == 1 and self.in_planes == self.planes) and (yn1.size()!= fyn_2.size())):
                yn1 += self.k3*self.shortcut4(fyn_2)
                yn1 = F.relu(yn1)
                x=[yn1,fyn,fyn_1,result]
                return x
        yn1 += self.k3*self.shortcut3(fyn_2)
        yn1 = F.relu(yn1)
        result.append(yn1)
        x=[yn1,fyn,fyn_1,result]
        return x
    



class AB3ResNet_Feather1(nn.Module):
    def __init__(self, block, num_blocks, h=1.0, b1=1.0, b2=1.0, b3=1.0, num_classes=10):
        super(AB3ResNet_Feather1, self).__init__()
        self.in_planes = 16
        self.k1=b1*h
        self.k2=b2*h
        self.k3=b3*h
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        
        self.result=[]

        #self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.k1, self.k2, self.k3))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        result=[]
        yn=self.conv1(x)
        yn = F.relu(self.bn1(yn))
        result.append(yn)
        out=[yn,0,0,result]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        self.result = out[3]
        yn = out[0]
        out = F.avg_pool2d(yn, yn.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    
def ab3resnet20(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [3, 3, 3], h, b1, b2, b3)
def ab3resnet32(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [5, 5, 5], h, b1, b2, b3) 
def ab3resnet44(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [7, 7, 7], h, b1, b2, b3) 
def ab3resnet56(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [9, 9, 9], h, b1, b2, b3)
def ab3resnet68(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [11, 11, 11], h, b1, b2, b3)
def ab3resnet80(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [13, 13, 13], h, b1, b2, b3)
def ab3resnet92(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [15, 15, 15], h, b1, b2, b3)
def ab3resnet104(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [17, 17, 17], h, b1, b2, b3) 
def ab3resnet110(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [18, 18, 18], h, b1, b2, b3)

def ab3resnet20_f(h, b1, b2, b3):
    return AB3ResNet_Feather(AB3BasicBlock_F, [3, 3, 3], h, b1, b2, b3)
def ab3resnet110_f(h, b1, b2, b3):
    return AB3ResNet_Feather(AB3BasicBlock_F, [18, 18, 18], h, b1, b2, b3)

def ab3resnet20_f1(h, b1, b2, b3):
    return AB3ResNet_Feather1(AB3BasicBlock_F1, [3, 3, 3], h, b1, b2, b3)
def ab3resnet110_f1(h, b1, b2, b3):
    return AB3ResNet_Feather1(AB3BasicBlock_F1, [18, 18, 18], h, b1, b2, b3)

def ab3res100net20(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [3, 3, 3], h, b1, b2, b3,100)
def ab3res100net32(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [5, 5, 5], h, b1, b2, b3,100) 
def ab3res100net44(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [7, 7, 7], h, b1, b2, b3,100) 
def ab3res100net56(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [9, 9, 9], h, b1, b2, b3,100)
def ab3res100net68(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [11, 11, 11], h, b1, b2, b3,100)
def ab3res100net80(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [13, 13, 13], h, b1, b2, b3,100)
def ab3res100net92(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [15, 15, 15], h, b1, b2, b3,100)
def ab3res100net104(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [17, 17, 17], h, b1, b2, b3,100) 
def ab3res100net110(h, b1, b2, b3):
    return AB3ResNet(AB3BasicBlock, [18, 18, 18], h, b1, b2, b3,100) 


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class AB1BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k1=1.0, option='A'):
        super(AB1BasicBlock, self).__init__()
        self.k1=k1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.k1
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AB1ResNet(nn.Module):
    def __init__(self, block, num_blocks, h=1.0, b1=1.0, num_classes=10):
        super(AB1ResNet, self).__init__()
        self.in_planes = 16
        self.k1=b1*h
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        #self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.k1))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ab1resnet20(h,b1):
    return AB1ResNet(AB1BasicBlock, [3, 3, 3],h,b1)

def ab1resnet32(h,b1):
    return AB1ResNet(AB1BasicBlock, [5, 5, 5],h,b1)

def ab1resnet44(h,b1):
    return AB1ResNet(AB1BasicBlock, [7, 7, 7],h,b1)

def ab1resnet56(h,b1):
    return AB1ResNet(AB1BasicBlock, [9, 9, 9],h,b1)

def ab1resnet110(h,b1):
    return AB1ResNet(AB1BasicBlock, [18, 18, 18],h,b1)

'''----------------------------------------------------------------------------------------------------'''

def ab1res100net20(h,b1):
    return AB1ResNet(AB1BasicBlock, [3, 3, 3],h,b1,100)


def ab1res100net32(h,b1):
    return AB1ResNet(AB1BasicBlock, [5, 5, 5],h,b1,100)


def ab1res100net44(h,b1):
    return AB1ResNet(AB1BasicBlock, [7, 7, 7],h,b1,100)


def ab1res100net56(h,b1):
    return AB1ResNet(AB1BasicBlock, [9, 9, 9],h,b1,100)


def ab1res100net110(h,b1):
    return AB1ResNet(AB1BasicBlock, [18, 18, 18],h,b1,100)











class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        #self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])
'''----------------------------------------------------------------------------------------------------'''

def res100net20():
    return ResNet(BasicBlock, [3, 3, 3],100)


def res100net32():
    return ResNet(BasicBlock, [5, 5, 5],100)


def res100net44():
    return ResNet(BasicBlock, [7, 7, 7],100)


def res100net56():
    return ResNet(BasicBlock, [9, 9, 9],100)


def res100net110():
    return ResNet(BasicBlock, [18, 18, 18],100)




