'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
#%%
import torch
import torch.nn as nn
# from torchstat import stat
import torch.nn.functional as F







class AB2PreActResNet152(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(AB2PreActResNet152, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        
        #B1_1
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B1_2
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B1_3
        self.bn7 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B2_1
        self.bn10 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B2_2
        self.bn13 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B2_3
        self.bn16 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn17 = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(128)
        self.conv18 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B2_4
        self.bn19 = nn.BatchNorm2d(512)
        self.conv19 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn20 = nn.BatchNorm2d(128)
        self.conv20 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B2_5
        self.bn22 = nn.BatchNorm2d(512)
        self.conv22 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn23 = nn.BatchNorm2d(128)
        self.conv23 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24 = nn.BatchNorm2d(128)
        self.conv24 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B2_6
        self.bn25 = nn.BatchNorm2d(512)
        self.conv25 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn26 = nn.BatchNorm2d(128)
        self.conv26 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn27 = nn.BatchNorm2d(128)
        self.conv27 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B2_7
        self.bn28 = nn.BatchNorm2d(512)
        self.conv28 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn29 = nn.BatchNorm2d(128)
        self.conv29 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn30 = nn.BatchNorm2d(128)
        self.conv30 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B2_8
        self.bn31 = nn.BatchNorm2d(512)
        self.conv31= nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn32 = nn.BatchNorm2d(128)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(128)
        self.conv33 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_1
        self.bn34 = nn.BatchNorm2d(512)
        self.conv34 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn35 = nn.BatchNorm2d(256)
        self.conv35 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn36 = nn.BatchNorm2d(256)
        self.conv36 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_2
        self.bn37 = nn.BatchNorm2d(1024)
        self.conv37 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn38 = nn.BatchNorm2d(256)
        self.conv38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn39 = nn.BatchNorm2d(256)
        self.conv39 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_3
        self.bn40 = nn.BatchNorm2d(1024)
        self.conv40 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn41 = nn.BatchNorm2d(256)
        self.conv41 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(256)
        self.conv42 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_4
        self.bn43 = nn.BatchNorm2d(1024)
        self.conv43 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn44 = nn.BatchNorm2d(256)
        self.conv44 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn45 = nn.BatchNorm2d(256)
        self.conv45 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_5
        self.bn46 = nn.BatchNorm2d(1024)
        self.conv46 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn47 = nn.BatchNorm2d(256)
        self.conv47 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn48 = nn.BatchNorm2d(256)
        self.conv48 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_6
        self.bn49 = nn.BatchNorm2d(1024)
        self.conv49 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn50 = nn.BatchNorm2d(256)
        self.conv50 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(256)
        self.conv51 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_7
        self.bn52 = nn.BatchNorm2d(1024)
        self.conv52 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn53 = nn.BatchNorm2d(256)
        self.conv53 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn54 = nn.BatchNorm2d(256)
        self.conv54 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_8
        self.bn55 = nn.BatchNorm2d(1024)
        self.conv55 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn56 = nn.BatchNorm2d(256)
        self.conv56 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn57 = nn.BatchNorm2d(256)
        self.conv57 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_9
        self.bn58 = nn.BatchNorm2d(1024)
        self.conv58 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn59 = nn.BatchNorm2d(256)
        self.conv59 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn60 = nn.BatchNorm2d(256)
        self.conv60 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_10
        self.bn61 = nn.BatchNorm2d(1024)
        self.conv61 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn62 = nn.BatchNorm2d(256)
        self.conv62 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn63 = nn.BatchNorm2d(256)
        self.conv63 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_11
        self.bn64 = nn.BatchNorm2d(1024)
        self.conv64 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn65 = nn.BatchNorm2d(256)
        self.conv65 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn66 = nn.BatchNorm2d(256)
        self.conv66 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_12
        self.bn67 = nn.BatchNorm2d(1024)
        self.conv67 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn68 = nn.BatchNorm2d(256)
        self.conv68 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn69 = nn.BatchNorm2d(256)
        self.conv69 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_13
        self.bn70 = nn.BatchNorm2d(1024)
        self.conv70 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn71 = nn.BatchNorm2d(256)
        self.conv71 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(256)
        self.conv72 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_14
        self.bn73 = nn.BatchNorm2d(1024)
        self.conv73 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn74 = nn.BatchNorm2d(256)
        self.conv74 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn75 = nn.BatchNorm2d(256)
        self.conv75 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_15
        self.bn76 = nn.BatchNorm2d(1024)
        self.conv76 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn77 = nn.BatchNorm2d(256)
        self.conv77 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn78 = nn.BatchNorm2d(256)
        self.conv78 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_16
        self.bn79 = nn.BatchNorm2d(1024)
        self.conv79 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn80 = nn.BatchNorm2d(256)
        self.conv80 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn81 = nn.BatchNorm2d(256)
        self.conv81 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_17
        self.bn82 = nn.BatchNorm2d(1024)
        self.conv82 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn83 = nn.BatchNorm2d(256)
        self.conv83 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn84 = nn.BatchNorm2d(256)
        self.conv84 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_18
        self.bn85 = nn.BatchNorm2d(1024)
        self.conv85 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn86 = nn.BatchNorm2d(256)
        self.conv86 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn87 = nn.BatchNorm2d(256)
        self.conv87 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_19
        self.bn88 = nn.BatchNorm2d(1024)
        self.conv88 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn89 = nn.BatchNorm2d(256)
        self.conv89 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn90 = nn.BatchNorm2d(256)
        self.conv90 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_20
        self.bn91 = nn.BatchNorm2d(1024)
        self.conv91 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn92 = nn.BatchNorm2d(256)
        self.conv92 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(256)
        self.conv93 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_21
        self.bn94 = nn.BatchNorm2d(1024)
        self.conv94 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn95 = nn.BatchNorm2d(256)
        self.conv95 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn96 = nn.BatchNorm2d(256)
        self.conv96 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B3_22
        self.bn97 = nn.BatchNorm2d(1024)
        self.conv97 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn98 = nn.BatchNorm2d(256)
        self.conv98 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn99 = nn.BatchNorm2d(256)
        self.conv99 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_23
        self.bn100 = nn.BatchNorm2d(1024)
        self.conv100 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(256)
        self.conv101 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn102 = nn.BatchNorm2d(256)
        self.conv102 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B3_24
        self.bn103 = nn.BatchNorm2d(1024)
        self.conv103 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn104 = nn.BatchNorm2d(256)
        self.conv104 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn105 = nn.BatchNorm2d(256)
        self.conv105 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_25
        self.bn106 = nn.BatchNorm2d(1024)
        self.conv106 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn107 = nn.BatchNorm2d(256)
        self.conv107 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn108 = nn.BatchNorm2d(256)
        self.conv108 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_26
        self.bn109 = nn.BatchNorm2d(1024)
        self.conv109 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn110 = nn.BatchNorm2d(256)
        self.conv110 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.conv111 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_27
        self.bn112 = nn.BatchNorm2d(1024)
        self.conv112 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.conv113 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn114 = nn.BatchNorm2d(256)
        self.conv114 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_28
        self.bn115 = nn.BatchNorm2d(1024)
        self.conv115 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn116 = nn.BatchNorm2d(256)
        self.conv116 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn117 = nn.BatchNorm2d(256)
        self.conv117 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_29
        self.bn118 = nn.BatchNorm2d(1024)
        self.conv118 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn119 = nn.BatchNorm2d(256)
        self.conv119 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn120 = nn.BatchNorm2d(256)
        self.conv120 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_30
        self.bn121 = nn.BatchNorm2d(1024)
        self.conv121 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn122 = nn.BatchNorm2d(256)
        self.conv122 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn123 = nn.BatchNorm2d(256)
        self.conv123 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_31
        self.bn124 = nn.BatchNorm2d(1024)
        self.conv124 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn125 = nn.BatchNorm2d(256)
        self.conv125 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn126 = nn.BatchNorm2d(256)
        self.conv126 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_32
        self.bn127 = nn.BatchNorm2d(1024)
        self.conv127 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn128 = nn.BatchNorm2d(256)
        self.conv128 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn129 = nn.BatchNorm2d(256)
        self.conv129 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_33
        self.bn130 = nn.BatchNorm2d(1024)
        self.conv130 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn131 = nn.BatchNorm2d(256)
        self.conv131 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn132 = nn.BatchNorm2d(256)
        self.conv132 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_34
        self.bn133 = nn.BatchNorm2d(1024)
        self.conv133 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn134 = nn.BatchNorm2d(256)
        self.conv134 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn135 = nn.BatchNorm2d(256)
        self.conv135 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_35
        self.bn136 = nn.BatchNorm2d(1024)
        self.conv136 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn137 = nn.BatchNorm2d(256)
        self.conv137 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn138 = nn.BatchNorm2d(256)
        self.conv138 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_36
        self.bn139 = nn.BatchNorm2d(1024)
        self.conv139 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn140 = nn.BatchNorm2d(256)
        self.conv140 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn141 = nn.BatchNorm2d(256)
        self.conv141 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        #B4_1
        self.bn142 = nn.BatchNorm2d(1024)
        self.conv142 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn143 = nn.BatchNorm2d(512)
        self.conv143 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn144 = nn.BatchNorm2d(512)
        self.conv144 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B4_2
        self.bn145 = nn.BatchNorm2d(2048)
        self.conv145 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn146 = nn.BatchNorm2d(512)
        self.conv146 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn147 = nn.BatchNorm2d(512)
        self.conv147 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B4_3
        self.bn148 = nn.BatchNorm2d(2048)
        self.conv148 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn149 = nn.BatchNorm2d(512)
        self.conv149 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn150 = nn.BatchNorm2d(512)
        self.conv150 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        
        
        
        self.conva = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(256)
        
        self.convb = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bnc = nn.BatchNorm2d(256)
        self.convc = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        self.convd = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bne = nn.BatchNorm2d(512)
        self.conve = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        self.convf = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bng = nn.BatchNorm2d(1024)
        self.convg = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(2048, out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv0(x)))
        yn = out
        out = self.conv3(F.relu(self.bn3(self.conv2(F.relu(self.bn2(self.conv1(out)))))))
        fyn_1 = out
        out = self.conva(yn) + fyn_1
        yn = out
        out = self.conv6(F.relu(self.bn6(self.conv5(F.relu(self.bn5(self.conv4(F.relu(self.bn4(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv9(F.relu(self.bn9(self.conv8(F.relu(self.bn8(self.conv7(F.relu(self.bn7(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn10(out))
        yn = out
        out = self.conv12(F.relu(self.bn12(self.conv11(F.relu(self.bn11(self.conv10(out)))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convc(self.bnc(F.avg_pool2d(fyn_1,2))))
        out = out + self.convb(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv15(F.relu(self.bn15(self.conv14(F.relu(self.bn14(self.conv13(F.relu(self.bn13(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv18(F.relu(self.bn18(self.conv17(F.relu(self.bn17(self.conv16(F.relu(self.bn16(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv21(F.relu(self.bn21(self.conv20(F.relu(self.bn20(self.conv19(F.relu(self.bn19(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv24(F.relu(self.bn24(self.conv23(F.relu(self.bn23(self.conv22(F.relu(self.bn22(out)))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv27(F.relu(self.bn27(self.conv26(F.relu(self.bn26(self.conv25(F.relu(self.bn25(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv30(F.relu(self.bn30(self.conv29(F.relu(self.bn29(self.conv28(F.relu(self.bn28(out)))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv33(F.relu(self.bn33(self.conv32(F.relu(self.bn32(self.conv31(F.relu(self.bn31(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn34(out))
        yn = out
        out = self.conv36(F.relu(self.bn36(self.conv35(F.relu(self.bn35(self.conv34(out)))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.conve(self.bne(F.avg_pool2d(fyn_1,2))))
        out = out + self.convd(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv39(F.relu(self.bn39(self.conv38(F.relu(self.bn38(self.conv37(F.relu(self.bn37(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv42(F.relu(self.bn42(self.conv41(F.relu(self.bn41(self.conv40(F.relu(self.bn40(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv45(F.relu(self.bn45(self.conv44(F.relu(self.bn44(self.conv43(F.relu(self.bn43(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv48(F.relu(self.bn48(self.conv47(F.relu(self.bn47(self.conv46(F.relu(self.bn46(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv51(F.relu(self.bn51(self.conv50(F.relu(self.bn50(self.conv49(F.relu(self.bn49(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv54(F.relu(self.bn54(self.conv53(F.relu(self.bn53(self.conv52(F.relu(self.bn52(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv57(F.relu(self.bn57(self.conv56(F.relu(self.bn56(self.conv55(F.relu(self.bn55(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv60(F.relu(self.bn60(self.conv59(F.relu(self.bn59(self.conv58(F.relu(self.bn58(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv63(F.relu(self.bn63(self.conv62(F.relu(self.bn62(self.conv61(F.relu(self.bn61(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv66(F.relu(self.bn66(self.conv65(F.relu(self.bn65(self.conv64(F.relu(self.bn64(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv69(F.relu(self.bn69(self.conv68(F.relu(self.bn68(self.conv67(F.relu(self.bn67(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv72(F.relu(self.bn72(self.conv71(F.relu(self.bn71(self.conv70(F.relu(self.bn70(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv75(F.relu(self.bn75(self.conv74(F.relu(self.bn74(self.conv73(F.relu(self.bn73(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv78(F.relu(self.bn78(self.conv77(F.relu(self.bn77(self.conv76(F.relu(self.bn76(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv81(F.relu(self.bn81(self.conv80(F.relu(self.bn80(self.conv79(F.relu(self.bn79(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv84(F.relu(self.bn84(self.conv83(F.relu(self.bn83(self.conv82(F.relu(self.bn82(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv87(F.relu(self.bn87(self.conv86(F.relu(self.bn86(self.conv85(F.relu(self.bn85(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv90(F.relu(self.bn90(self.conv89(F.relu(self.bn89(self.conv88(F.relu(self.bn88(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv93(F.relu(self.bn93(self.conv92(F.relu(self.bn92(self.conv91(F.relu(self.bn91(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv96(F.relu(self.bn96(self.conv95(F.relu(self.bn95(self.conv94(F.relu(self.bn94(out)))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv99(F.relu(self.bn99(self.conv98(F.relu(self.bn98(self.conv97(F.relu(self.bn97(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv102(F.relu(self.bn102(self.conv101(F.relu(self.bn101(self.conv100(F.relu(self.bn100(out)))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv105(F.relu(self.bn105(self.conv104(F.relu(self.bn104(self.conv103(F.relu(self.bn103(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv108(F.relu(self.bn108(self.conv107(F.relu(self.bn107(self.conv106(F.relu(self.bn106(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv111(F.relu(self.bn111(self.conv110(F.relu(self.bn110(self.conv109(F.relu(self.bn109(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv114(F.relu(self.bn114(self.conv113(F.relu(self.bn113(self.conv112(F.relu(self.bn112(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv117(F.relu(self.bn117(self.conv116(F.relu(self.bn116(self.conv115(F.relu(self.bn115(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv120(F.relu(self.bn120(self.conv119(F.relu(self.bn119(self.conv118(F.relu(self.bn118(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv123(F.relu(self.bn123(self.conv122(F.relu(self.bn122(self.conv121(F.relu(self.bn121(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv126(F.relu(self.bn126(self.conv125(F.relu(self.bn125(self.conv124(F.relu(self.bn124(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv129(F.relu(self.bn129(self.conv128(F.relu(self.bn128(self.conv127(F.relu(self.bn127(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv132(F.relu(self.bn132(self.conv131(F.relu(self.bn131(self.conv130(F.relu(self.bn130(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv135(F.relu(self.bn135(self.conv134(F.relu(self.bn134(self.conv133(F.relu(self.bn133(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv138(F.relu(self.bn138(self.conv137(F.relu(self.bn137(self.conv136(F.relu(self.bn136(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv141(F.relu(self.bn141(self.conv140(F.relu(self.bn140(self.conv139(F.relu(self.bn139(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn142(out))
        yn = out
        out = self.conv144(F.relu(self.bn144(self.conv143(F.relu(self.bn143(self.conv142(out)))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convg(self.bng(F.avg_pool2d(fyn_1,2))))
        out = out + self.convf(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv147(F.relu(self.bn147(self.conv146(F.relu(self.bn146(self.conv145(F.relu(self.bn145(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv150(F.relu(self.bn150(self.conv149(F.relu(self.bn149(self.conv148(F.relu(self.bn148(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class AB2PreActResNet101(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(AB2PreActResNet101, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        
        #B1_1
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B1_2
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B1_3
        self.bn7 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B2_1
        self.bn10 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B2_2
        self.bn13 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B2_3
        self.bn16 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn17 = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(128)
        self.conv18 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B2_8
        self.bn31 = nn.BatchNorm2d(512)
        self.conv31= nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn32 = nn.BatchNorm2d(128)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(128)
        self.conv33 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_1
        self.bn34 = nn.BatchNorm2d(512)
        self.conv34 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn35 = nn.BatchNorm2d(256)
        self.conv35 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn36 = nn.BatchNorm2d(256)
        self.conv36 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_2
        self.bn37 = nn.BatchNorm2d(1024)
        self.conv37 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn38 = nn.BatchNorm2d(256)
        self.conv38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn39 = nn.BatchNorm2d(256)
        self.conv39 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_3
        self.bn40 = nn.BatchNorm2d(1024)
        self.conv40 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn41 = nn.BatchNorm2d(256)
        self.conv41 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(256)
        self.conv42 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_4
        self.bn43 = nn.BatchNorm2d(1024)
        self.conv43 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn44 = nn.BatchNorm2d(256)
        self.conv44 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn45 = nn.BatchNorm2d(256)
        self.conv45 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_5
        self.bn46 = nn.BatchNorm2d(1024)
        self.conv46 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn47 = nn.BatchNorm2d(256)
        self.conv47 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn48 = nn.BatchNorm2d(256)
        self.conv48 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_6
        self.bn49 = nn.BatchNorm2d(1024)
        self.conv49 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn50 = nn.BatchNorm2d(256)
        self.conv50 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(256)
        self.conv51 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_7
        self.bn52 = nn.BatchNorm2d(1024)
        self.conv52 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn53 = nn.BatchNorm2d(256)
        self.conv53 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn54 = nn.BatchNorm2d(256)
        self.conv54 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_8
        self.bn55 = nn.BatchNorm2d(1024)
        self.conv55 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn56 = nn.BatchNorm2d(256)
        self.conv56 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn57 = nn.BatchNorm2d(256)
        self.conv57 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_9
        self.bn58 = nn.BatchNorm2d(1024)
        self.conv58 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn59 = nn.BatchNorm2d(256)
        self.conv59 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn60 = nn.BatchNorm2d(256)
        self.conv60 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_10
        self.bn61 = nn.BatchNorm2d(1024)
        self.conv61 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn62 = nn.BatchNorm2d(256)
        self.conv62 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn63 = nn.BatchNorm2d(256)
        self.conv63 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_11
        self.bn64 = nn.BatchNorm2d(1024)
        self.conv64 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn65 = nn.BatchNorm2d(256)
        self.conv65 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn66 = nn.BatchNorm2d(256)
        self.conv66 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_12
        self.bn67 = nn.BatchNorm2d(1024)
        self.conv67 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn68 = nn.BatchNorm2d(256)
        self.conv68 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn69 = nn.BatchNorm2d(256)
        self.conv69 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_13
        self.bn70 = nn.BatchNorm2d(1024)
        self.conv70 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn71 = nn.BatchNorm2d(256)
        self.conv71 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(256)
        self.conv72 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_14
        self.bn73 = nn.BatchNorm2d(1024)
        self.conv73 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn74 = nn.BatchNorm2d(256)
        self.conv74 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn75 = nn.BatchNorm2d(256)
        self.conv75 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_15
        self.bn76 = nn.BatchNorm2d(1024)
        self.conv76 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn77 = nn.BatchNorm2d(256)
        self.conv77 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn78 = nn.BatchNorm2d(256)
        self.conv78 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_16
        self.bn79 = nn.BatchNorm2d(1024)
        self.conv79 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn80 = nn.BatchNorm2d(256)
        self.conv80 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn81 = nn.BatchNorm2d(256)
        self.conv81 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_17
        self.bn82 = nn.BatchNorm2d(1024)
        self.conv82 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn83 = nn.BatchNorm2d(256)
        self.conv83 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn84 = nn.BatchNorm2d(256)
        self.conv84 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_18
        self.bn85 = nn.BatchNorm2d(1024)
        self.conv85 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn86 = nn.BatchNorm2d(256)
        self.conv86 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn87 = nn.BatchNorm2d(256)
        self.conv87 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_19
        self.bn88 = nn.BatchNorm2d(1024)
        self.conv88 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn89 = nn.BatchNorm2d(256)
        self.conv89 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn90 = nn.BatchNorm2d(256)
        self.conv90 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_20
        self.bn91 = nn.BatchNorm2d(1024)
        self.conv91 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn92 = nn.BatchNorm2d(256)
        self.conv92 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(256)
        self.conv93 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B3_21
        self.bn94 = nn.BatchNorm2d(1024)
        self.conv94 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn95 = nn.BatchNorm2d(256)
        self.conv95 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn96 = nn.BatchNorm2d(256)
        self.conv96 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_22
        self.bn97 = nn.BatchNorm2d(1024)
        self.conv97 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn98 = nn.BatchNorm2d(256)
        self.conv98 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn99 = nn.BatchNorm2d(256)
        self.conv99 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B3_36
        self.bn139 = nn.BatchNorm2d(1024)
        self.conv139 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn140 = nn.BatchNorm2d(256)
        self.conv140 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn141 = nn.BatchNorm2d(256)
        self.conv141 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        #B4_1
        self.bn142 = nn.BatchNorm2d(1024)
        self.conv142 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn143 = nn.BatchNorm2d(512)
        self.conv143 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn144 = nn.BatchNorm2d(512)
        self.conv144 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B4_2
        self.bn145 = nn.BatchNorm2d(2048)
        self.conv145 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn146 = nn.BatchNorm2d(512)
        self.conv146 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn147 = nn.BatchNorm2d(512)
        self.conv147 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B4_3
        self.bn148 = nn.BatchNorm2d(2048)
        self.conv148 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn149 = nn.BatchNorm2d(512)
        self.conv149 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn150 = nn.BatchNorm2d(512)
        self.conv150 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        
        
        
        self.conva = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(256)
        
        self.convb = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bnc = nn.BatchNorm2d(256)
        self.convc = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        self.convd = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bne = nn.BatchNorm2d(512)
        self.conve = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        self.convf = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bng = nn.BatchNorm2d(1024)
        self.convg = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(2048, out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv0(x)))
        yn = out
        out = self.conv3(F.relu(self.bn3(self.conv2(F.relu(self.bn2(self.conv1(out)))))))
        fyn_1 = out
        out = self.conva(yn) + fyn_1
        yn = out
        out = self.conv6(F.relu(self.bn6(self.conv5(F.relu(self.bn5(self.conv4(F.relu(self.bn4(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv9(F.relu(self.bn9(self.conv8(F.relu(self.bn8(self.conv7(F.relu(self.bn7(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn10(out))
        yn = out
        out = self.conv12(F.relu(self.bn12(self.conv11(F.relu(self.bn11(self.conv10(out)))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convc(self.bnc(F.avg_pool2d(fyn_1,2))))
        out = out + self.convb(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv15(F.relu(self.bn15(self.conv14(F.relu(self.bn14(self.conv13(F.relu(self.bn13(out)))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv18(F.relu(self.bn18(self.conv17(F.relu(self.bn17(self.conv16(F.relu(self.bn16(out)))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv33(F.relu(self.bn33(self.conv32(F.relu(self.bn32(self.conv31(F.relu(self.bn31(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn34(out))
        yn = out
        out = self.conv36(F.relu(self.bn36(self.conv35(F.relu(self.bn35(self.conv34(out)))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.conve(self.bne(F.avg_pool2d(fyn_1,2))))
        out = out + self.convd(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv39(F.relu(self.bn39(self.conv38(F.relu(self.bn38(self.conv37(F.relu(self.bn37(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv42(F.relu(self.bn42(self.conv41(F.relu(self.bn41(self.conv40(F.relu(self.bn40(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv45(F.relu(self.bn45(self.conv44(F.relu(self.bn44(self.conv43(F.relu(self.bn43(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv48(F.relu(self.bn48(self.conv47(F.relu(self.bn47(self.conv46(F.relu(self.bn46(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv51(F.relu(self.bn51(self.conv50(F.relu(self.bn50(self.conv49(F.relu(self.bn49(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv54(F.relu(self.bn54(self.conv53(F.relu(self.bn53(self.conv52(F.relu(self.bn52(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv57(F.relu(self.bn57(self.conv56(F.relu(self.bn56(self.conv55(F.relu(self.bn55(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv60(F.relu(self.bn60(self.conv59(F.relu(self.bn59(self.conv58(F.relu(self.bn58(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv63(F.relu(self.bn63(self.conv62(F.relu(self.bn62(self.conv61(F.relu(self.bn61(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv66(F.relu(self.bn66(self.conv65(F.relu(self.bn65(self.conv64(F.relu(self.bn64(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv69(F.relu(self.bn69(self.conv68(F.relu(self.bn68(self.conv67(F.relu(self.bn67(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv72(F.relu(self.bn72(self.conv71(F.relu(self.bn71(self.conv70(F.relu(self.bn70(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv75(F.relu(self.bn75(self.conv74(F.relu(self.bn74(self.conv73(F.relu(self.bn73(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv78(F.relu(self.bn78(self.conv77(F.relu(self.bn77(self.conv76(F.relu(self.bn76(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv81(F.relu(self.bn81(self.conv80(F.relu(self.bn80(self.conv79(F.relu(self.bn79(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv84(F.relu(self.bn84(self.conv83(F.relu(self.bn83(self.conv82(F.relu(self.bn82(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv87(F.relu(self.bn87(self.conv86(F.relu(self.bn86(self.conv85(F.relu(self.bn85(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv90(F.relu(self.bn90(self.conv89(F.relu(self.bn89(self.conv88(F.relu(self.bn88(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv93(F.relu(self.bn93(self.conv92(F.relu(self.bn92(self.conv91(F.relu(self.bn91(out)))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv96(F.relu(self.bn96(self.conv95(F.relu(self.bn95(self.conv94(F.relu(self.bn94(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv99(F.relu(self.bn99(self.conv98(F.relu(self.bn98(self.conv97(F.relu(self.bn97(out)))))))))
        '''
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv141(F.relu(self.bn141(self.conv140(F.relu(self.bn140(self.conv139(F.relu(self.bn139(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn142(out))
        yn = out
        out = self.conv144(F.relu(self.bn144(self.conv143(F.relu(self.bn143(self.conv142(out)))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convg(self.bng(F.avg_pool2d(fyn_1,2))))
        out = out + self.convf(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv147(F.relu(self.bn147(self.conv146(F.relu(self.bn146(self.conv145(F.relu(self.bn145(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv150(F.relu(self.bn150(self.conv149(F.relu(self.bn149(self.conv148(F.relu(self.bn148(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class AB2PreActResNet50(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(AB2PreActResNet50, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        
        #B1_1
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B1_2
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B1_3
        self.bn7 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B2_1
        self.bn10 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B2_2
        self.bn13 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B2_3
        self.bn16 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn17 = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(128)
        self.conv18 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B2_4
        self.bn31 = nn.BatchNorm2d(512)
        self.conv31= nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn32 = nn.BatchNorm2d(128)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(128)
        self.conv33 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_1
        self.bn34 = nn.BatchNorm2d(512)
        self.conv34 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn35 = nn.BatchNorm2d(256)
        self.conv35 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn36 = nn.BatchNorm2d(256)
        self.conv36 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_2
        self.bn37 = nn.BatchNorm2d(1024)
        self.conv37 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn38 = nn.BatchNorm2d(256)
        self.conv38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn39 = nn.BatchNorm2d(256)
        self.conv39 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B3_3
        self.bn40 = nn.BatchNorm2d(1024)
        self.conv40 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn41 = nn.BatchNorm2d(256)
        self.conv41 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(256)
        self.conv42 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B3_4
        self.bn43 = nn.BatchNorm2d(1024)
        self.conv43 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn44 = nn.BatchNorm2d(256)
        self.conv44 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn45 = nn.BatchNorm2d(256)
        self.conv45 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        '''
        #B3_5
        self.bn46 = nn.BatchNorm2d(1024)
        self.conv46 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn47 = nn.BatchNorm2d(256)
        self.conv47 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn48 = nn.BatchNorm2d(256)
        self.conv48 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        #B3_36
        self.bn139 = nn.BatchNorm2d(1024)
        self.conv139 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn140 = nn.BatchNorm2d(256)
        self.conv140 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn141 = nn.BatchNorm2d(256)
        self.conv141 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        #B4_1
        self.bn142 = nn.BatchNorm2d(1024)
        self.conv142 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn143 = nn.BatchNorm2d(512)
        self.conv143 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn144 = nn.BatchNorm2d(512)
        self.conv144 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B4_2
        self.bn145 = nn.BatchNorm2d(2048)
        self.conv145 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn146 = nn.BatchNorm2d(512)
        self.conv146 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn147 = nn.BatchNorm2d(512)
        self.conv147 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        #B4_3
        self.bn148 = nn.BatchNorm2d(2048)
        self.conv148 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn149 = nn.BatchNorm2d(512)
        self.conv149 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn150 = nn.BatchNorm2d(512)
        self.conv150 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        
        
        
        self.conva = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(256)
        
        self.convb = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bnc = nn.BatchNorm2d(256)
        self.convc = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        self.convd = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bne = nn.BatchNorm2d(512)
        self.conve = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        self.convf = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bng = nn.BatchNorm2d(1024)
        self.convg = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(2048, out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv0(x)))
        yn = out
        out = self.conv3(F.relu(self.bn3(self.conv2(F.relu(self.bn2(self.conv1(out)))))))
        fyn_1 = out
        out = self.conva(yn) + fyn_1
        yn = out
        out = self.conv6(F.relu(self.bn6(self.conv5(F.relu(self.bn5(self.conv4(F.relu(self.bn4(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv9(F.relu(self.bn9(self.conv8(F.relu(self.bn8(self.conv7(F.relu(self.bn7(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn10(out))
        yn = out
        out = self.conv12(F.relu(self.bn12(self.conv11(F.relu(self.bn11(self.conv10(out)))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convc(self.bnc(F.avg_pool2d(fyn_1,2))))
        out = out + self.convb(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv15(F.relu(self.bn15(self.conv14(F.relu(self.bn14(self.conv13(F.relu(self.bn13(out)))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv18(F.relu(self.bn18(self.conv17(F.relu(self.bn17(self.conv16(F.relu(self.bn16(out)))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv33(F.relu(self.bn33(self.conv32(F.relu(self.bn32(self.conv31(F.relu(self.bn31(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn34(out))
        yn = out
        out = self.conv36(F.relu(self.bn36(self.conv35(F.relu(self.bn35(self.conv34(out)))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.conve(self.bne(F.avg_pool2d(fyn_1,2))))
        out = out + self.convd(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv39(F.relu(self.bn39(self.conv38(F.relu(self.bn38(self.conv37(F.relu(self.bn37(out)))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv42(F.relu(self.bn42(self.conv41(F.relu(self.bn41(self.conv40(F.relu(self.bn40(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv45(F.relu(self.bn45(self.conv44(F.relu(self.bn44(self.conv43(F.relu(self.bn43(out)))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv48(F.relu(self.bn48(self.conv47(F.relu(self.bn47(self.conv46(F.relu(self.bn46(out)))))))))
        
        
       
        
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv141(F.relu(self.bn141(self.conv140(F.relu(self.bn140(self.conv139(F.relu(self.bn139(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn142(out))
        yn = out
        out = self.conv144(F.relu(self.bn144(self.conv143(F.relu(self.bn143(self.conv142(out)))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convg(self.bng(F.avg_pool2d(fyn_1,2))))
        out = out + self.convf(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv147(F.relu(self.bn147(self.conv146(F.relu(self.bn146(self.conv145(F.relu(self.bn145(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv150(F.relu(self.bn150(self.conv149(F.relu(self.bn149(self.conv148(F.relu(self.bn148(out)))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class AB2PreActResNet34(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(AB2PreActResNet34, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        
        #B1_1
        
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B1_2
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B1_3
        self.bn5 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B2_1
        self.bn7 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B2_2
        self.bn9 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B2_3
        self.bn11 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B2_4
        self.bn13 = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B3_1
        self.bn15 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(256)
        self.conv16 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B3_2
        self.bn17 = nn.BatchNorm2d(256)
        self.conv17 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(256)
        self.conv18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B3_3
        self.bn19 = nn.BatchNorm2d(256)
        self.conv19 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(256)
        self.conv20 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B3_4
        self.bn21 = nn.BatchNorm2d(256)
        self.conv21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(256)
        self.conv22 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B3_5
        self.bn23 = nn.BatchNorm2d(256)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24 = nn.BatchNorm2d(256)
        self.conv24 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        
        #B3_6
        self.bn25 = nn.BatchNorm2d(256)
        self.conv25 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(256)
        self.conv26 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        
        #B4_1
        self.bn27 = nn.BatchNorm2d(256)
        self.conv27 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn28 = nn.BatchNorm2d(512)
        self.conv28 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B4_2
        self.bn29 = nn.BatchNorm2d(512)
        self.conv29 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn30 = nn.BatchNorm2d(512)
        self.conv30 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        
        
        #B4_3
        self.bn31 = nn.BatchNorm2d(512)
        self.conv31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(512)
        self.conv32 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        
        
        self.bna = nn.BatchNorm2d(64)
        self.conva = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.convb = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bnc = nn.BatchNorm2d(128)
        self.convc = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.convd = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bne = nn.BatchNorm2d(256)
        self.conve = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.convf = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        
        

        
        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(512, out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv0(x)))
        yn = out
        out = self.conv2(F.relu(self.bn2(self.conv1(out))))
        fyn_1 = out
        out = yn + fyn_1
        yn = out
        out = self.conv4(F.relu(self.bn4(self.conv3(F.relu(self.bn3(out))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv6(F.relu(self.bn6(self.conv5(F.relu(self.bn5(out))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn7(out))
        yn=out
        out = self.conv8(F.relu(self.bn8(self.conv7(out))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convb(F.avg_pool2d(fyn_1,2)))
        out = out + self.conva(self.bna(F.avg_pool2d(yn,2)))
        fyn_1 = fyn
        yn = out
        out = self.conv10(F.relu(self.bn10(self.conv9(F.relu(self.bn9(out))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv12(F.relu(self.bn12(self.conv11(F.relu(self.bn11(out))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv14(F.relu(self.bn14(self.conv13(F.relu(self.bn13(out))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn15(out))
        yn=out
        out = self.conv16(F.relu(self.bn16(self.conv15(out))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convd(F.avg_pool2d(fyn_1,2)))
        out = out + self.convc(self.bnc(F.avg_pool2d(yn,2)))
        fyn_1 = fyn
        yn = out
        out = self.conv18(F.relu(self.bn18(self.conv17(F.relu(self.bn17(out))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv20(F.relu(self.bn20(self.conv19(F.relu(self.bn19(out))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv22(F.relu(self.bn22(self.conv21(F.relu(self.bn21(out))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv24(F.relu(self.bn24(self.conv23(F.relu(self.bn23(out))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv26(F.relu(self.bn26(self.conv25(F.relu(self.bn25(out))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn27(out))
        yn=out
        out = self.conv28(F.relu(self.bn28(self.conv27(out))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convf(F.avg_pool2d(fyn_1,2)))
        out = out + 3/2. * self.conve(self.bne(F.avg_pool2d(yn,2)))
        fyn_1 = fyn
        yn = out
        out = self.conv30(F.relu(self.bn30(self.conv29(F.relu(self.bn29(out))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        yn = out
        out = self.conv32(F.relu(self.bn32(self.conv31(F.relu(self.bn31(out))))))
        

        
       
        
        
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class AB2PreActResNet18(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(AB2PreActResNet18, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        
        #B1_1
        
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B1_2
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        #B2_1
        self.bn5 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B2_2
        self.bn7 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)

        #B3_1
        self.bn9 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B3_2
        self.bn11 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)

        
        #B4_1
        self.bn13 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        
        #B4_2
        self.bn15 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        

        
        
        self.conva = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.convb = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb = nn.BatchNorm2d(64)
        
        self.convc = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.convd = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(128)
        
        self.conve = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.convf = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(256)
        

        
        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(512, out_planes)

    def forward(self, x):
        out = self.conv0(x)
        yn = out
        out = F.relu(self.bn1(out))
        out = self.conv2(F.relu(self.bn2(self.conv1(out))))
        fyn_1 = out
        out = yn + fyn_1
        yn = out
        out = self.conv4(F.relu(self.bn4(self.conv3(F.relu(self.bn3(out))))))

        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn5(out))
        yn=out
        out = self.conv6(F.relu(self.bn6(self.conv5(out))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convb(F.relu(self.bnb(F.avg_pool2d(fyn_1,2)))))
        out = out + self.conva(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv8(F.relu(self.bn8(self.conv7(F.relu(self.bn7(out))))))

        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn9(out))
        yn=out
        out = self.conv10(F.relu(self.bn10(self.conv9(out))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convd(F.relu(self.bnd(F.avg_pool2d(fyn_1,2)))))
        out = out + self.convc(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv12(F.relu(self.bn12(self.conv11(F.relu(self.bn11(out))))))

    
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        fyn_1 = fyn
        out = F.relu(self.bn13(out))
        yn=out
        out = self.conv14(F.relu(self.bn14(self.conv13(out))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.convf(F.relu(self.bnf(F.avg_pool2d(fyn_1,2)))))
        out = out + 3/2. * self.conve(F.avg_pool2d(yn,2))
        fyn_1 = fyn
        yn = out
        out = self.conv16(F.relu(self.bn16(self.conv15(F.relu(self.bn15(out))))))

        
       
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def AB2_PreAct100ResNet152():
    return AB2PreActResNet152(3,100)

def AB2_PreAct100ResNet101():
    return AB2PreActResNet101(3,100)

def AB2_PreAct100ResNet50():
    return AB2PreActResNet50(3,100)

def AB2_PreAct100ResNet34():
    return AB2PreActResNet34(3,100)

def AB2_PreAct100ResNet18():
    return AB2PreActResNet18(3,100)




def AB2_PreActResNet152():
    return AB2PreActResNet152(3,10)

def AB2_PreActResNet101():
    return AB2PreActResNet101(3,10)

def AB2_PreActResNet50():
    return AB2PreActResNet50(3,10)

def AB2_PreActResNet34():
    return AB2PreActResNet34(3,10)

def AB2_PreActResNet18():
    return AB2PreActResNet18(3,10)

class AB2ResNet152(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(AB2ResNet152, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        #B1_1
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        #B1_2
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        #B1_3
        self.conv7 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        #B2_1
        self.conv10 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        #B2_2
        self.conv13 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        #B2_3
        self.conv16 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn16 = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(128)
        self.conv18 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn18 = nn.BatchNorm2d(512)
        #B2_4
        self.conv19 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn19 = nn.BatchNorm2d(128)
        self.conv20 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(512)
        #B2_5
        self.conv22 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(128)
        self.conv23 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(128)
        self.conv24 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn24 = nn.BatchNorm2d(512)
        #B2_6
        self.conv25 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn25 = nn.BatchNorm2d(128)
        self.conv26 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(128)
        self.conv27 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn27 = nn.BatchNorm2d(512)
        '''#B2_7
        self.conv28 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn28 = nn.BatchNorm2d(128)
        self.conv29 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(128)
        self.conv30 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn30 = nn.BatchNorm2d(512)'''
        #B2_8
        self.conv31= nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn31 = nn.BatchNorm2d(128)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(128)
        self.conv33 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn33 = nn.BatchNorm2d(512)
        #B3_1
        self.conv34 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn34 = nn.BatchNorm2d(256)
        self.conv35 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn35 = nn.BatchNorm2d(256)
        self.conv36 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn36 = nn.BatchNorm2d(1024)
        #B3_2
        self.conv37 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn37 = nn.BatchNorm2d(256)
        self.conv38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn38 = nn.BatchNorm2d(256)
        self.conv39 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn39 = nn.BatchNorm2d(1024)
        #B3_3
        self.conv40 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn40 = nn.BatchNorm2d(256)
        self.conv41 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(256)
        self.conv42 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn42 = nn.BatchNorm2d(1024)
        #B3_4
        self.conv43 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn43 = nn.BatchNorm2d(256)
        self.conv44 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn44 = nn.BatchNorm2d(256)
        self.conv45 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn45 = nn.BatchNorm2d(1024)
        #B3_5
        self.conv46 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn46 = nn.BatchNorm2d(256)
        self.conv47 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn47 = nn.BatchNorm2d(256)
        self.conv48 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn48 = nn.BatchNorm2d(1024)
        #B3_6
        self.conv49 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn49 = nn.BatchNorm2d(256)
        self.conv50 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn50 = nn.BatchNorm2d(256)
        self.conv51 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn51 = nn.BatchNorm2d(1024)
        #B3_7
        self.conv52 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn52 = nn.BatchNorm2d(256)
        self.conv53 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn53 = nn.BatchNorm2d(256)
        self.conv54 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn54 = nn.BatchNorm2d(1024)
        #B3_8
        self.conv55 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn55 = nn.BatchNorm2d(256)
        self.conv56 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn56 = nn.BatchNorm2d(256)
        self.conv57 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn57 = nn.BatchNorm2d(1024)
        #B3_9
        self.conv58 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn58 = nn.BatchNorm2d(256)
        self.conv59 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn59 = nn.BatchNorm2d(256)
        self.conv60 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn60 = nn.BatchNorm2d(1024)
        #B3_10
        self.conv61 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn61 = nn.BatchNorm2d(256)
        self.conv62 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(256)
        self.conv63 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn63 = nn.BatchNorm2d(1024)
        #B3_11
        self.conv64 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn64 = nn.BatchNorm2d(256)
        self.conv65 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn65 = nn.BatchNorm2d(256)
        self.conv66 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn66 = nn.BatchNorm2d(1024)
        #B3_12
        self.conv67 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn67 = nn.BatchNorm2d(256)
        self.conv68 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn68 = nn.BatchNorm2d(256)
        self.conv69 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn69 = nn.BatchNorm2d(1024)
        #B3_13
        self.conv70 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn70 = nn.BatchNorm2d(256)
        self.conv71 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(256)
        self.conv72 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn72 = nn.BatchNorm2d(1024)
        #B3_14
        self.conv73 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn73 = nn.BatchNorm2d(256)
        self.conv74 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn74 = nn.BatchNorm2d(256)
        self.conv75 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn75 = nn.BatchNorm2d(1024)
        #B3_15
        self.conv76 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn76 = nn.BatchNorm2d(256)
        self.conv77 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn77 = nn.BatchNorm2d(256)
        self.conv78 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn78 = nn.BatchNorm2d(1024)
        #B3_16
        self.conv79 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn79 = nn.BatchNorm2d(256)
        self.conv80 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn80 = nn.BatchNorm2d(256)
        self.conv81 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(1024)
        #B3_17
        self.conv82 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn82 = nn.BatchNorm2d(256)
        self.conv83 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(256)
        self.conv84 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn84 = nn.BatchNorm2d(1024)
        #B3_18
        self.conv85 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn85 = nn.BatchNorm2d(256)
        self.conv86 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn86 = nn.BatchNorm2d(256)
        self.conv87 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn87 = nn.BatchNorm2d(1024)
        #B3_19
        self.conv88 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn88 = nn.BatchNorm2d(256)
        self.conv89 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn89 = nn.BatchNorm2d(256)
        self.conv90 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn90 = nn.BatchNorm2d(1024)
        #B3_20
        self.conv91 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(256)
        self.conv92 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(256)
        self.conv93 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn93 = nn.BatchNorm2d(1024)
        #B3_21
        self.conv94 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn94 = nn.BatchNorm2d(256)
        self.conv95 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn95 = nn.BatchNorm2d(256)
        self.conv96 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn96 = nn.BatchNorm2d(1024)
        #B3_22
        self.conv97 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn97 = nn.BatchNorm2d(256)
        self.conv98 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn98 = nn.BatchNorm2d(256)
        self.conv99 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn99 = nn.BatchNorm2d(1024)
        #B3_23
        self.conv100 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn100 = nn.BatchNorm2d(256)
        self.conv101 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn101 = nn.BatchNorm2d(256)
        self.conv102 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn102 = nn.BatchNorm2d(1024)
        #B3_24
        self.conv103 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn103 = nn.BatchNorm2d(256)
        self.conv104 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn104 = nn.BatchNorm2d(256)
        self.conv105 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn105 = nn.BatchNorm2d(1024)
        #B3_25
        self.conv106 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn106 = nn.BatchNorm2d(256)
        self.conv107 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn107 = nn.BatchNorm2d(256)
        self.conv108 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn108 = nn.BatchNorm2d(1024)
        #B3_26
        self.conv109 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn109 = nn.BatchNorm2d(256)
        self.conv110 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn110 = nn.BatchNorm2d(256)
        self.conv111 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(1024)
        #B3_27
        self.conv112 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.conv113 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.conv114 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn114 = nn.BatchNorm2d(1024)
        #B3_28
        self.conv115 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn115 = nn.BatchNorm2d(256)
        self.conv116 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn116 = nn.BatchNorm2d(256)
        self.conv117 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn117 = nn.BatchNorm2d(1024)
        #B3_29
        self.conv118 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn118 = nn.BatchNorm2d(256)
        self.conv119 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn119 = nn.BatchNorm2d(256)
        self.conv120 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn120 = nn.BatchNorm2d(1024)
        #B3_30
        self.conv121 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn121 = nn.BatchNorm2d(256)
        self.conv122 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122 = nn.BatchNorm2d(256)
        self.conv123 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn123 = nn.BatchNorm2d(1024)
        #B3_31
        self.conv124 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn124 = nn.BatchNorm2d(256)
        self.conv125 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn125 = nn.BatchNorm2d(256)
        self.conv126 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn126 = nn.BatchNorm2d(1024)
        #B3_32
        self.conv127 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn127 = nn.BatchNorm2d(256)
        self.conv128 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn128 = nn.BatchNorm2d(256)
        self.conv129 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn129 = nn.BatchNorm2d(1024)
        #B3_33
        self.conv130 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn130 = nn.BatchNorm2d(256)
        self.conv131 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn131 = nn.BatchNorm2d(256)
        self.conv132 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn132 = nn.BatchNorm2d(1024)
        '''#B3_34
        self.conv133 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn133 = nn.BatchNorm2d(256)
        self.conv134 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn134 = nn.BatchNorm2d(256)
        self.conv135 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn135 = nn.BatchNorm2d(1024)
        #B3_35
        self.conv136 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn136 = nn.BatchNorm2d(256)
        self.conv137 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn137 = nn.BatchNorm2d(256)
        self.conv138 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn138 = nn.BatchNorm2d(1024)'''
        #B3_36
        self.conv139 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn139 = nn.BatchNorm2d(256)
        self.conv140 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn140 = nn.BatchNorm2d(256)
        self.conv141 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn141 = nn.BatchNorm2d(1024)
        
        #B4_1
        self.conv142 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn142 = nn.BatchNorm2d(512)
        self.conv143 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn143 = nn.BatchNorm2d(512)
        self.conv144 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn144 = nn.BatchNorm2d(2048)
        #B4_2
        self.conv145 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn145 = nn.BatchNorm2d(512)
        self.conv146 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn146 = nn.BatchNorm2d(512)
        self.conv147 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn147 = nn.BatchNorm2d(2048)
        #B4_3
        self.conv148 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn148 = nn.BatchNorm2d(512)
        self.conv149 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn149 = nn.BatchNorm2d(512)
        self.conv150 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn150 = nn.BatchNorm2d(2048)
        
        
        
        
        self.conva = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(256)
        
        self.convb = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb = nn.BatchNorm2d(512)
        self.convc = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnc = nn.BatchNorm2d(512)
        
        self.convd = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(1024)
        self.conve = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne = nn.BatchNorm2d(1024)
        
        self.convf = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(2048)
        self.convg = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bng = nn.BatchNorm2d(2048)
        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(2048, out_planes)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        yn = out
        out = self.bn3(self.conv3(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(out))))))))
        fyn_1 = out
        out = self.bna(self.conva(yn)) + fyn_1
        out = F.relu(out)
        yn = out
        out = self.bn6(self.conv6(F.relu(self.bn5(self.conv5(F.relu(self.bn4(self.conv4(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn9(self.conv9(F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn12(self.conv12(F.relu(self.bn11(self.conv11(F.relu(self.bn10(self.conv10(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bnb(self.convb(F.avg_pool2d(fyn_1,2))))
        out = out + self.bnc(self.convc(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn15(self.conv15(F.relu(self.bn14(self.conv14(F.relu(self.bn13(self.conv13(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn18(self.conv18(F.relu(self.bn17(self.conv17(F.relu(self.bn16(self.conv16(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn21(self.conv21(F.relu(self.bn20(self.conv20(F.relu(self.bn19(self.conv19(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn24(self.conv24(F.relu(self.bn23(self.conv23(F.relu(self.bn22(self.conv22(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn27(self.conv27(F.relu(self.bn26(self.conv26(F.relu(self.bn25(self.conv25(out))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn30(self.conv30(F.relu(self.bn29(self.conv29(F.relu(self.bn28(self.conv28(out))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn33(self.conv33(F.relu(self.bn32(self.conv32(F.relu(self.bn31(self.conv31(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn36(self.conv36(F.relu(self.bn35(self.conv35(F.relu(self.bn34(self.conv34(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bnd(self.convd(F.avg_pool2d(fyn_1,2))))
        out = out + self.bne(self.conve(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn39(self.conv39(F.relu(self.bn38(self.conv38(F.relu(self.bn37(self.conv37(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn42(self.conv42(F.relu(self.bn41(self.conv41(F.relu(self.bn40(self.conv40(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn45(self.conv45(F.relu(self.bn44(self.conv44(F.relu(self.bn43(self.conv43(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn48(self.conv48(F.relu(self.bn47(self.conv47(F.relu(self.bn46(self.conv46(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn51(self.conv51(F.relu(self.bn50(self.conv50(F.relu(self.bn49(self.conv49(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn54(self.conv54(F.relu(self.bn53(self.conv53(F.relu(self.bn52(self.conv52(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn57(self.conv57(F.relu(self.bn56(self.conv56(F.relu(self.bn55(self.conv55(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn60(self.conv60(F.relu(self.bn59(self.conv59(F.relu(self.bn58(self.conv58(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn63(self.conv63(F.relu(self.bn62(self.conv62(F.relu(self.bn61(self.conv61(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn66(self.conv66(F.relu(self.bn65(self.conv65(F.relu(self.bn64(self.conv64(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn69(self.conv69(F.relu(self.bn68(self.conv68(F.relu(self.bn67(self.conv67(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn72(self.conv72(F.relu(self.bn71(self.conv71(F.relu(self.bn70(self.conv70(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn75(self.conv75(F.relu(self.bn74(self.conv74(F.relu(self.bn73(self.conv73(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn78(self.conv78(F.relu(self.bn77(self.conv77(F.relu(self.bn76(self.conv76(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn81(self.conv81(F.relu(self.bn80(self.conv80(F.relu(self.bn79(self.conv79(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn84(self.conv84(F.relu(self.bn83(self.conv83(F.relu(self.bn82(self.conv82(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn87(self.conv87(F.relu(self.bn86(self.conv86(F.relu(self.bn85(self.conv85(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn90(self.conv90(F.relu(self.bn89(self.conv89(F.relu(self.bn88(self.conv88(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn93(self.conv93(F.relu(self.bn92(self.conv92(F.relu(self.bn91(self.conv91(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn96(self.conv96(F.relu(self.bn95(self.conv95(F.relu(self.bn94(self.conv94(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn99(self.conv99(F.relu(self.bn98(self.conv98(F.relu(self.bn97(self.conv97(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn102(self.conv102(F.relu(self.bn101(self.conv101(F.relu(self.bn100(self.conv100(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn105(self.conv105(F.relu(self.bn104(self.conv104(F.relu(self.bn103(self.conv103(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn108(self.conv108(F.relu(self.bn107(self.conv107(F.relu(self.bn106(self.conv106(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn111(self.conv111(F.relu(self.bn110(self.conv110(F.relu(self.bn109(self.conv109(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn114(self.conv114(F.relu(self.bn113(self.conv113(F.relu(self.bn112(self.conv112(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn117(self.conv117(F.relu(self.bn116(self.conv116(F.relu(self.bn115(self.conv115(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn120(self.conv120(F.relu(self.bn119(self.conv119(F.relu(self.bn118(self.conv118(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn123(self.conv123(F.relu(self.bn122(self.conv122(F.relu(self.bn121(self.conv121(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn126(self.conv126(F.relu(self.bn125(self.conv125(F.relu(self.bn124(self.conv124(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn129(self.conv129(F.relu(self.bn128(self.conv128(F.relu(self.bn127(self.conv127(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn132(self.conv132(F.relu(self.bn131(self.conv131(F.relu(self.bn130(self.conv130(out))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn135(self.conv135(F.relu(self.bn134(self.conv134(F.relu(self.bn133(self.conv133(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn138(self.conv138(F.relu(self.bn137(self.conv137(F.relu(self.bn136(self.conv136(out))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn141(self.conv141(F.relu(self.bn140(self.conv140(F.relu(self.bn139(self.conv139(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn144(self.conv144(F.relu(self.bn143(self.conv143(F.relu(self.bn142(self.conv142(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bnf(self.convf(F.avg_pool2d(fyn_1,2))))
        out = out + self.bng(self.convg(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn147(self.conv147(F.relu(self.bn146(self.conv146(F.relu(self.bn145(self.conv145(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn150(self.conv150(F.relu(self.bn149(self.conv149(F.relu(self.bn148(self.conv148(out))))))))
        
        
        
        
        
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class AB2ResNet101(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(AB2ResNet101, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        #B1_1
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        #B1_2
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        #B1_3
        self.conv7 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        #B2_1
        self.conv10 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        #B2_2
        self.conv13 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        '''#B2_3
        self.conv16 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn16 = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(128)
        self.conv18 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn18 = nn.BatchNorm2d(512)'''
        #B2_4
        self.conv19 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn19 = nn.BatchNorm2d(128)
        self.conv20 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(512)
        #B3_1
        self.conv22 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(256)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(256)
        self.conv24 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn24 = nn.BatchNorm2d(1024)
        #B3_2
        self.conv25 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn25 = nn.BatchNorm2d(256)
        self.conv26 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(256)
        self.conv27 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn27 = nn.BatchNorm2d(1024)
        #B3_3
        self.conv28 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn28 = nn.BatchNorm2d(256)
        self.conv29 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(256)
        self.conv30 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn30 = nn.BatchNorm2d(1024)
        #B3_4
        self.conv31 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn33 = nn.BatchNorm2d(1024)
        #B3_5
        self.conv34 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn34 = nn.BatchNorm2d(256)
        self.conv35 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn35 = nn.BatchNorm2d(256)
        self.conv36 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn36 = nn.BatchNorm2d(1024)
        #B3_6
        self.conv37 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn37 = nn.BatchNorm2d(256)
        self.conv38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn38 = nn.BatchNorm2d(256)
        self.conv39 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn39 = nn.BatchNorm2d(1024)
        #B3_7
        self.conv40 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn40 = nn.BatchNorm2d(256)
        self.conv41 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(256)
        self.conv42 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn42 = nn.BatchNorm2d(1024)
        #B3_8
        self.conv43 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn43 = nn.BatchNorm2d(256)
        self.conv44 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn44 = nn.BatchNorm2d(256)
        self.conv45 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn45 = nn.BatchNorm2d(1024)
        #B3_9
        self.conv46 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn46 = nn.BatchNorm2d(256)
        self.conv47 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn47 = nn.BatchNorm2d(256)
        self.conv48 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn48 = nn.BatchNorm2d(1024)
        #B3_10
        self.conv49 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn49 = nn.BatchNorm2d(256)
        self.conv50 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn50 = nn.BatchNorm2d(256)
        self.conv51 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn51 = nn.BatchNorm2d(1024)
        #B3_11
        self.conv52 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn52 = nn.BatchNorm2d(256)
        self.conv53 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn53 = nn.BatchNorm2d(256)
        self.conv54 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn54 = nn.BatchNorm2d(1024)
        #B3_12
        self.conv55 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn55 = nn.BatchNorm2d(256)
        self.conv56 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn56 = nn.BatchNorm2d(256)
        self.conv57 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn57 = nn.BatchNorm2d(1024)
        #B3_13
        self.conv58 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn58 = nn.BatchNorm2d(256)
        self.conv59 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn59 = nn.BatchNorm2d(256)
        self.conv60 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn60 = nn.BatchNorm2d(1024)
        #B3_14
        self.conv61 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn61 = nn.BatchNorm2d(256)
        self.conv62 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(256)
        self.conv63 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn63 = nn.BatchNorm2d(1024)
        #B3_15
        self.conv64 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn64 = nn.BatchNorm2d(256)
        self.conv65 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn65 = nn.BatchNorm2d(256)
        self.conv66 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn66 = nn.BatchNorm2d(1024)
        #B3_16
        self.conv67 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn67 = nn.BatchNorm2d(256)
        self.conv68 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn68 = nn.BatchNorm2d(256)
        self.conv69 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn69 = nn.BatchNorm2d(1024)
        #B3_17
        self.conv70 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn70 = nn.BatchNorm2d(256)
        self.conv71 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(256)
        self.conv72 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn72 = nn.BatchNorm2d(1024)
        #B3_18
        self.conv73 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn73 = nn.BatchNorm2d(256)
        self.conv74 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn74 = nn.BatchNorm2d(256)
        self.conv75 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn75 = nn.BatchNorm2d(1024)
        #B3_19
        self.conv76 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn76 = nn.BatchNorm2d(256)
        self.conv77 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn77 = nn.BatchNorm2d(256)
        self.conv78 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn78 = nn.BatchNorm2d(1024)
        '''#B3_20
        self.conv79 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn79 = nn.BatchNorm2d(256)
        self.conv80 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn80 = nn.BatchNorm2d(256)
        self.conv81 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(1024)
        #B3_21
        self.conv82 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn82 = nn.BatchNorm2d(256)
        self.conv83 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(256)
        self.conv84 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn84 = nn.BatchNorm2d(1024)'''
        #B3_22
        self.conv85 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn85 = nn.BatchNorm2d(256)
        self.conv86 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn86 = nn.BatchNorm2d(256)
        self.conv87 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn87 = nn.BatchNorm2d(1024)
        #B3_23
        self.conv88 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn88 = nn.BatchNorm2d(256)
        self.conv89 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn89 = nn.BatchNorm2d(256)
        self.conv90 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn90 = nn.BatchNorm2d(1024)
        #B4_1
        self.conv91 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(512)
        self.conv92 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(512)
        self.conv93 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn93 = nn.BatchNorm2d(2048)
        #B4_2
        self.conv94 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn94 = nn.BatchNorm2d(512)
        self.conv95 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn95 = nn.BatchNorm2d(512)
        self.conv96 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn96 = nn.BatchNorm2d(2048)
        #B4_3
        self.conv97 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn97 = nn.BatchNorm2d(512)
        self.conv98 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn98 = nn.BatchNorm2d(512)
        self.conv99 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn99 = nn.BatchNorm2d(2048)
        
        
        
        
        self.conva = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(256)
        
        self.convb = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb = nn.BatchNorm2d(512)
        self.convc = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnc = nn.BatchNorm2d(512)
        
        self.convd = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(1024)
        self.conve = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne = nn.BatchNorm2d(1024)
        
        self.convf = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(2048)
        self.convg = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bng = nn.BatchNorm2d(2048)
        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(2048, out_planes)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        yn = out
        out = self.bn3(self.conv3(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(out))))))))
        fyn_1 = out
        out = self.bna(self.conva(yn)) + fyn_1
        out = F.relu(out)
        yn = out
        out = self.bn6(self.conv6(F.relu(self.bn5(self.conv5(F.relu(self.bn4(self.conv4(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn9(self.conv9(F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn12(self.conv12(F.relu(self.bn11(self.conv11(F.relu(self.bn10(self.conv10(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bnb(self.convb(F.avg_pool2d(fyn_1,2))))
        out = out + self.bnc(self.convc(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn15(self.conv15(F.relu(self.bn14(self.conv14(F.relu(self.bn13(self.conv13(out))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn18(self.conv18(F.relu(self.bn17(self.conv17(F.relu(self.bn16(self.conv16(out))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn21(self.conv21(F.relu(self.bn20(self.conv20(F.relu(self.bn19(self.conv19(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn24(self.conv24(F.relu(self.bn23(self.conv23(F.relu(self.bn22(self.conv22(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bnd(self.convd(F.avg_pool2d(fyn_1,2))))
        out = out + self.bne(self.conve(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn27(self.conv27(F.relu(self.bn26(self.conv26(F.relu(self.bn25(self.conv25(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn30(self.conv30(F.relu(self.bn29(self.conv29(F.relu(self.bn28(self.conv28(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn33(self.conv33(F.relu(self.bn32(self.conv32(F.relu(self.bn31(self.conv31(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn36(self.conv36(F.relu(self.bn35(self.conv35(F.relu(self.bn34(self.conv34(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn39(self.conv39(F.relu(self.bn38(self.conv38(F.relu(self.bn37(self.conv37(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn42(self.conv42(F.relu(self.bn41(self.conv41(F.relu(self.bn40(self.conv40(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn45(self.conv45(F.relu(self.bn44(self.conv44(F.relu(self.bn43(self.conv43(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn48(self.conv48(F.relu(self.bn47(self.conv47(F.relu(self.bn46(self.conv46(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn51(self.conv51(F.relu(self.bn50(self.conv50(F.relu(self.bn49(self.conv49(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn54(self.conv54(F.relu(self.bn53(self.conv53(F.relu(self.bn52(self.conv52(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn57(self.conv57(F.relu(self.bn56(self.conv56(F.relu(self.bn55(self.conv55(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn60(self.conv60(F.relu(self.bn59(self.conv59(F.relu(self.bn58(self.conv58(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn63(self.conv63(F.relu(self.bn62(self.conv62(F.relu(self.bn61(self.conv61(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn66(self.conv66(F.relu(self.bn65(self.conv65(F.relu(self.bn64(self.conv64(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn69(self.conv69(F.relu(self.bn68(self.conv68(F.relu(self.bn67(self.conv67(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn72(self.conv72(F.relu(self.bn71(self.conv71(F.relu(self.bn70(self.conv70(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn75(self.conv75(F.relu(self.bn74(self.conv74(F.relu(self.bn73(self.conv73(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn78(self.conv78(F.relu(self.bn77(self.conv77(F.relu(self.bn76(self.conv76(out))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn81(self.conv81(F.relu(self.bn80(self.conv80(F.relu(self.bn79(self.conv79(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn84(self.conv84(F.relu(self.bn83(self.conv83(F.relu(self.bn82(self.conv82(out))))))))
        '''
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn87(self.conv87(F.relu(self.bn86(self.conv86(F.relu(self.bn85(self.conv85(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn90(self.conv90(F.relu(self.bn89(self.conv89(F.relu(self.bn88(self.conv88(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn93(self.conv93(F.relu(self.bn92(self.conv92(F.relu(self.bn91(self.conv91(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bnf(self.convf(F.avg_pool2d(fyn_1,2))))
        out = out + self.bng(self.convg(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn96(self.conv96(F.relu(self.bn95(self.conv95(F.relu(self.bn94(self.conv94(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn99(self.conv99(F.relu(self.bn98(self.conv98(F.relu(self.bn97(self.conv97(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class AB2ResNet50(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(AB2ResNet50, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        #B1_1
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        #B1_2
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        #B1_3
        self.conv7 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        #B2_1
        self.conv10 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        #B2_2
        self.conv13 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        #B2_3
        self.conv16 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn16 = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(128)
        self.conv18 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn18 = nn.BatchNorm2d(512)
        #B2_4
        self.conv19 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn19 = nn.BatchNorm2d(128)
        self.conv20 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(512)
        #B3_1
        self.conv22 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(256)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(256)
        self.conv24 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn24 = nn.BatchNorm2d(1024)
        #B3_2
        self.conv25 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn25 = nn.BatchNorm2d(256)
        self.conv26 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(256)
        self.conv27 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn27 = nn.BatchNorm2d(1024)
        '''#B3_3
        self.conv28 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn28 = nn.BatchNorm2d(256)
        self.conv29 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(256)
        self.conv30 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn30 = nn.BatchNorm2d(1024)
        #B3_4
        self.conv31 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn33 = nn.BatchNorm2d(1024)'''
        #B3_5
        self.conv34 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn34 = nn.BatchNorm2d(256)
        self.conv35 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn35 = nn.BatchNorm2d(256)
        self.conv36 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn36 = nn.BatchNorm2d(1024)
        #B3_6
        self.conv37 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn37 = nn.BatchNorm2d(256)
        self.conv38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn38 = nn.BatchNorm2d(256)
        self.conv39 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn39 = nn.BatchNorm2d(1024)
        #B4_1
        self.conv40 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn40 = nn.BatchNorm2d(512)
        self.conv41 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn42 = nn.BatchNorm2d(2048)
        #B4_2
        self.conv43 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn43 = nn.BatchNorm2d(512)
        self.conv44 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn44 = nn.BatchNorm2d(512)
        self.conv45 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn45 = nn.BatchNorm2d(2048)
        #B4_3
        self.conv46 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn46 = nn.BatchNorm2d(512)
        self.conv47 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn47 = nn.BatchNorm2d(512)
        self.conv48 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn48 = nn.BatchNorm2d(2048)
        
        
        
        
        self.conva = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(256)
        
        self.convb = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb = nn.BatchNorm2d(512)
        self.convc = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnc = nn.BatchNorm2d(512)
        
        self.convd = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(1024)
        self.conve = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne = nn.BatchNorm2d(1024)
        
        self.convf = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(2048)
        self.convg = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bng = nn.BatchNorm2d(2048)
        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(2048, out_planes)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        yn = out
        out = self.bn3(self.conv3(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(out))))))))
        fyn_1 = out
        out = self.bna(self.conva(yn)) + fyn_1
        out = F.relu(out)
        yn = out
        out = self.bn6(self.conv6(F.relu(self.bn5(self.conv5(F.relu(self.bn4(self.conv4(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn9(self.conv9(F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn12(self.conv12(F.relu(self.bn11(self.conv11(F.relu(self.bn10(self.conv10(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bnb(self.convb(F.avg_pool2d(fyn_1,2))))
        out = out + self.bnc(self.convc(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn15(self.conv15(F.relu(self.bn14(self.conv14(F.relu(self.bn13(self.conv13(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn18(self.conv18(F.relu(self.bn17(self.conv17(F.relu(self.bn16(self.conv16(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn21(self.conv21(F.relu(self.bn20(self.conv20(F.relu(self.bn19(self.conv19(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn24(self.conv24(F.relu(self.bn23(self.conv23(F.relu(self.bn22(self.conv22(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bnd(self.convd(F.avg_pool2d(fyn_1,2))))
        out = out + self.bne(self.conve(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn27(self.conv27(F.relu(self.bn26(self.conv26(F.relu(self.bn25(self.conv25(out))))))))
        
        '''fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn30(self.conv30(F.relu(self.bn29(self.conv29(F.relu(self.bn28(self.conv28(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn33(self.conv33(F.relu(self.bn32(self.conv32(F.relu(self.bn31(self.conv31(out))))))))'''
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn36(self.conv36(F.relu(self.bn35(self.conv35(F.relu(self.bn34(self.conv34(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn39(self.conv39(F.relu(self.bn38(self.conv38(F.relu(self.bn37(self.conv37(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn42(self.conv42(F.relu(self.bn41(self.conv41(F.relu(self.bn40(self.conv40(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bnf(self.convf(F.avg_pool2d(fyn_1,2))))
        out = out + self.bng(self.convg(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn45(self.conv45(F.relu(self.bn44(self.conv44(F.relu(self.bn43(self.conv43(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn48(self.conv48(F.relu(self.bn47(self.conv47(F.relu(self.bn46(self.conv46(out))))))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class AB2ResNet34(nn.Module):
    def __init__(self, in_planes, planes, out_planes, stride=1):
        super(AB2ResNet34, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        
        self.conv15 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(256)
        self.conv16 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(256)
        self.conv17 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(256)
        self.conv18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(256)
        self.conv19 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19 = nn.BatchNorm2d(256)
        self.conv20 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(256)
        self.conv21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(256)
        self.conv22 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(256)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(256)
        self.conv24 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24 = nn.BatchNorm2d(256)
        self.conv25 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn25 = nn.BatchNorm2d(256)
        self.conv26 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(256)
        
        self.conv27 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn27 = nn.BatchNorm2d(512)
        self.conv28 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn28 = nn.BatchNorm2d(512)
        self.conv29 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(512)
        self.conv30 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn30 = nn.BatchNorm2d(512)
        self.conv31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(512)
        self.conv32 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(512)
        
        
        self.conva = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(128)
        self.convb = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb = nn.BatchNorm2d(128)
        
        self.convc = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnc = nn.BatchNorm2d(256)
        self.convd = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(256)
        
        self.conve = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne = nn.BatchNorm2d(512)
        self.convf = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(512, out_planes)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        yn = out
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(out)))))
        fyn_1 = out
        out = out + yn
        out = F.relu(out)
        yn = out
        out = self.bn4(self.conv4(F.relu(self.bn3(self.conv3(out))))) 
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn6(self.conv6(F.relu(self.bn5(self.conv5(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn8(self.conv8(F.relu(self.bn7(self.conv7(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bna(self.conva(F.avg_pool2d(fyn_1,2))))
        out = out + self.bnb(self.convb(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn10(self.conv10(F.relu(self.bn9(self.conv9(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn12(self.conv12(F.relu(self.bn11(self.conv11(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn14(self.conv14(F.relu(self.bn13(self.conv13(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn16(self.conv16(F.relu(self.bn15(self.conv15(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bnc(self.convc(F.avg_pool2d(fyn_1,2))))
        out = out + self.bnd(self.convd(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn18(self.conv18(F.relu(self.bn17(self.conv17(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn20(self.conv20(F.relu(self.bn19(self.conv19(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn22(self.conv22(F.relu(self.bn21(self.conv21(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn24(self.conv24(F.relu(self.bn23(self.conv23(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn26(self.conv26(F.relu(self.bn25(self.conv25(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn28(self.conv28(F.relu(self.bn27(self.conv27(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * self.bne(self.conve(F.avg_pool2d(fyn_1,2))))
        out = out + self.bnf(self.convf(F.avg_pool2d(yn,2)))
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn30(self.conv30(F.relu(self.bn29(self.conv29(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        fyn_1 = fyn
        yn = out
        out = self.bn32(self.conv32(F.relu(self.bn31(self.conv31(out)))))
        
        fyn = out
        out = 3/2. * fyn
        out = out + (-1/2. * fyn_1)
        out = out + yn
        out = F.relu(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

class AB2ResNet18(nn.Module):
    def __init__(self, in_planes, planes, out_planes, stride=1):
        super(AB2ResNet18, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(512)
        
        self.conva = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(128)
        self.convb = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb = nn.BatchNorm2d(128)
        
        self.convc = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnc = nn.BatchNorm2d(256)
        self.convd = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(256)
        
        self.conve = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne = nn.BatchNorm2d(512)
        self.convf = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(512, out_planes)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        y1 = out
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(out)))))
        fy1 = out
        out = out + y1
        out = F.relu(out)
        y2 = out
        out = self.bn4(self.conv4(F.relu(self.bn3(self.conv3(out))))) 
        fy2 = out
        out = 3/2. * out
        out = out + (-1/2. * fy1)
        out = out + y2
        out = F.relu(out)
        y3 = out
        out = self.bn6(self.conv6(F.relu(self.bn5(self.conv5(out)))))
        fy3 = out
        out = 3/2. * out
        out = out + (-1/2. * self.bna(self.conva(F.avg_pool2d(fy2,2))))
        out = out + self.bnb(self.convb(F.avg_pool2d(y3,2)))
        out = F.relu(out)
        y4 = out
        out = self.bn8(self.conv8(F.relu(self.bn7(self.conv7(out)))))
        fy4 = out
        out = 3/2. * out
        out = out + (-1/2. * fy3)
        out = out + y4
        out = F.relu(out)
        y5 = out
        out = self.bn10(self.conv10(F.relu(self.bn9(self.conv9(out)))))
        fy5 = out
        out = 3/2. * out
        out = out + (-1/2. * self.bnc(self.convc(F.avg_pool2d(fy4,2))))
        out = out + self.bnd(self.convd(F.avg_pool2d(y5,2)))
        out = F.relu(out)
        y6 = out
        out = self.bn12(self.conv12(F.relu(self.bn11(self.conv11(out)))))
        fy6 = out 
        out = 3/2. * out
        out = out + (-1/2. * fy5)
        out = out + y6
        out = F.relu(out)
        y7 = out
        out = self.bn14(self.conv14(F.relu(self.bn13(self.conv13(out)))))
        fy7 = out
        out = 3/2. * out
        out = out + (-1/2. * self.bne(self.conve(F.avg_pool2d(fy6,2))))
        out = out + self.bnf(self.convf(F.avg_pool2d(y7,2)))
        out = F.relu(out) 
        y8 = out
        out = self.bn16(self.conv16(F.relu(self.bn15(self.conv15(out)))))
        fy8 = out 
        out = 3/2. * out
        out = out + (-1/2. * fy7)
        out = out + y8
        out = F.relu(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

def AB2_ResNet152():
    return AB2ResNet152(3,10)

def AB2_ResNet101():
    return AB2ResNet101(3,10)

def AB2_ResNet50():
    return AB2ResNet50(3,10)

def AB2_ResNet34():
    return AB2ResNet34(3,64,10)
    
def AB2_ResNet18():
    return AB2ResNet18(3,64,10)

def AB2_Res100Net152():
    return AB2ResNet152(3,100)

def AB2_Res100Net101():
    return AB2ResNet101(3,100)

def AB2_Res100Net50():
    return AB2ResNet50(3,100)

def AB2_Res100Net34():
    return AB2ResNet34(3,64,100)
    
def AB2_Res100Net18():
    return AB2ResNet18(3,64,100)



class AB3ResNet152(nn.Module):
    def __init__(self, in_planes,out_planes, stride=1):
        super(AB3ResNet152, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        #B1_1
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        #B1_2
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        #B1_3
        self.conv7 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        
        #B2_1
        self.conv10 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        #B2_2
        self.conv13 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        #B2_3
        self.conv16 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn16 = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(128)
        self.conv18 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn18 = nn.BatchNorm2d(512)
        #B2_4
        self.conv19 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn19 = nn.BatchNorm2d(128)
        self.conv20 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(512)
        #B2_5
        self.conv22 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(128)
        self.conv23 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(128)
        self.conv24 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn24 = nn.BatchNorm2d(512)
        #B2_6
        self.conv25 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn25 = nn.BatchNorm2d(128)
        self.conv26 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(128)
        self.conv27 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn27 = nn.BatchNorm2d(512)
        '''#B2_7
        self.conv28 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn28 = nn.BatchNorm2d(128)
        self.conv29 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(128)
        self.conv30 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn30 = nn.BatchNorm2d(512)'''
        #B2_8
        self.conv31= nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn31 = nn.BatchNorm2d(128)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(128)
        self.conv33 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn33 = nn.BatchNorm2d(512)
        
        #B3_1
        self.conv34 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn34 = nn.BatchNorm2d(256)
        self.conv35 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn35 = nn.BatchNorm2d(256)
        self.conv36 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn36 = nn.BatchNorm2d(1024)
        #B3_2
        self.conv37 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn37 = nn.BatchNorm2d(256)
        self.conv38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn38 = nn.BatchNorm2d(256)
        self.conv39 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn39 = nn.BatchNorm2d(1024)
        #B3_3
        self.conv40 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn40 = nn.BatchNorm2d(256)
        self.conv41 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(256)
        self.conv42 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn42 = nn.BatchNorm2d(1024)
        #B3_4
        self.conv43 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn43 = nn.BatchNorm2d(256)
        self.conv44 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn44 = nn.BatchNorm2d(256)
        self.conv45 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn45 = nn.BatchNorm2d(1024)
        #B3_5
        self.conv46 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn46 = nn.BatchNorm2d(256)
        self.conv47 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn47 = nn.BatchNorm2d(256)
        self.conv48 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn48 = nn.BatchNorm2d(1024)
        #B3_6
        self.conv49 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn49 = nn.BatchNorm2d(256)
        self.conv50 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn50 = nn.BatchNorm2d(256)
        self.conv51 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn51 = nn.BatchNorm2d(1024)
        #B3_7
        self.conv52 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn52 = nn.BatchNorm2d(256)
        self.conv53 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn53 = nn.BatchNorm2d(256)
        self.conv54 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn54 = nn.BatchNorm2d(1024)
        #B3_8
        self.conv55 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn55 = nn.BatchNorm2d(256)
        self.conv56 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn56 = nn.BatchNorm2d(256)
        self.conv57 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn57 = nn.BatchNorm2d(1024)
        #B3_9
        self.conv58 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn58 = nn.BatchNorm2d(256)
        self.conv59 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn59 = nn.BatchNorm2d(256)
        self.conv60 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn60 = nn.BatchNorm2d(1024)
        #B3_10
        self.conv61 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn61 = nn.BatchNorm2d(256)
        self.conv62 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(256)
        self.conv63 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn63 = nn.BatchNorm2d(1024)
        #B3_11
        self.conv64 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn64 = nn.BatchNorm2d(256)
        self.conv65 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn65 = nn.BatchNorm2d(256)
        self.conv66 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn66 = nn.BatchNorm2d(1024)
        #B3_12
        self.conv67 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn67 = nn.BatchNorm2d(256)
        self.conv68 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn68 = nn.BatchNorm2d(256)
        self.conv69 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn69 = nn.BatchNorm2d(1024)
        #B3_13
        self.conv70 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn70 = nn.BatchNorm2d(256)
        self.conv71 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(256)
        self.conv72 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn72 = nn.BatchNorm2d(1024)
        #B3_14
        self.conv73 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn73 = nn.BatchNorm2d(256)
        self.conv74 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn74 = nn.BatchNorm2d(256)
        self.conv75 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn75 = nn.BatchNorm2d(1024)
        #B3_15
        self.conv76 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn76 = nn.BatchNorm2d(256)
        self.conv77 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn77 = nn.BatchNorm2d(256)
        self.conv78 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn78 = nn.BatchNorm2d(1024)
        #B3_16
        self.conv79 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn79 = nn.BatchNorm2d(256)
        self.conv80 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn80 = nn.BatchNorm2d(256)
        self.conv81 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(1024)
        #B3_17
        self.conv82 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn82 = nn.BatchNorm2d(256)
        self.conv83 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(256)
        self.conv84 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn84 = nn.BatchNorm2d(1024)
        #B3_18
        self.conv85 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn85 = nn.BatchNorm2d(256)
        self.conv86 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn86 = nn.BatchNorm2d(256)
        self.conv87 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn87 = nn.BatchNorm2d(1024)
        #B3_19
        self.conv88 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn88 = nn.BatchNorm2d(256)
        self.conv89 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn89 = nn.BatchNorm2d(256)
        self.conv90 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn90 = nn.BatchNorm2d(1024)
        #B3_20
        self.conv91 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(256)
        self.conv92 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(256)
        self.conv93 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn93 = nn.BatchNorm2d(1024)
        #B3_21
        self.conv94 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn94 = nn.BatchNorm2d(256)
        self.conv95 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn95 = nn.BatchNorm2d(256)
        self.conv96 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn96 = nn.BatchNorm2d(1024)
        #B3_22
        self.conv97 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn97 = nn.BatchNorm2d(256)
        self.conv98 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn98 = nn.BatchNorm2d(256)
        self.conv99 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn99 = nn.BatchNorm2d(1024)
        #B3_23
        self.conv100 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn100 = nn.BatchNorm2d(256)
        self.conv101 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn101 = nn.BatchNorm2d(256)
        self.conv102 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn102 = nn.BatchNorm2d(1024)
        #B3_24
        self.conv103 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn103 = nn.BatchNorm2d(256)
        self.conv104 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn104 = nn.BatchNorm2d(256)
        self.conv105 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn105 = nn.BatchNorm2d(1024)
        #B3_25
        self.conv106 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn106 = nn.BatchNorm2d(256)
        self.conv107 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn107 = nn.BatchNorm2d(256)
        self.conv108 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn108 = nn.BatchNorm2d(1024)
        #B3_26
        self.conv109 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn109 = nn.BatchNorm2d(256)
        self.conv110 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn110 = nn.BatchNorm2d(256)
        self.conv111 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(1024)
        #B3_27
        self.conv112 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.conv113 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.conv114 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn114 = nn.BatchNorm2d(1024)
        #B3_28
        self.conv115 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn115 = nn.BatchNorm2d(256)
        self.conv116 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn116 = nn.BatchNorm2d(256)
        self.conv117 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn117 = nn.BatchNorm2d(1024)
        '''#B3_29
        self.conv118 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn118 = nn.BatchNorm2d(256)
        self.conv119 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn119 = nn.BatchNorm2d(256)
        self.conv120 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn120 = nn.BatchNorm2d(1024)
        #B3_30
        self.conv121 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn121 = nn.BatchNorm2d(256)
        self.conv122 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122 = nn.BatchNorm2d(256)
        self.conv123 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn123 = nn.BatchNorm2d(1024)
        #B3_31
        self.conv124 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn124 = nn.BatchNorm2d(256)
        self.conv125 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn125 = nn.BatchNorm2d(256)
        self.conv126 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn126 = nn.BatchNorm2d(1024)
        #B3_32
        self.conv127 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn127 = nn.BatchNorm2d(256)
        self.conv128 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn128 = nn.BatchNorm2d(256)
        self.conv129 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn129 = nn.BatchNorm2d(1024)
        #B3_33
        self.conv130 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn130 = nn.BatchNorm2d(256)
        self.conv131 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn131 = nn.BatchNorm2d(256)
        self.conv132 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn132 = nn.BatchNorm2d(1024)
        #B3_34
        self.conv133 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn133 = nn.BatchNorm2d(256)
        self.conv134 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn134 = nn.BatchNorm2d(256)
        self.conv135 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn135 = nn.BatchNorm2d(1024)
        #B3_35
        self.conv136 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn136 = nn.BatchNorm2d(256)
        self.conv137 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn137 = nn.BatchNorm2d(256)
        self.conv138 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn138 = nn.BatchNorm2d(1024)'''
        #B3_36
        self.conv139 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn139 = nn.BatchNorm2d(256)
        self.conv140 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn140 = nn.BatchNorm2d(256)
        self.conv141 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn141 = nn.BatchNorm2d(1024)
        
        #B4_1
        self.conv142 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn142 = nn.BatchNorm2d(512)
        self.conv143 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn143 = nn.BatchNorm2d(512)
        self.conv144 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn144 = nn.BatchNorm2d(2048)
        #B4_2
        self.conv145 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn145 = nn.BatchNorm2d(512)
        self.conv146 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn146 = nn.BatchNorm2d(512)
        self.conv147 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn147 = nn.BatchNorm2d(2048)
        #B4_3
        self.conv148 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn148 = nn.BatchNorm2d(512)
        self.conv149 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn149 = nn.BatchNorm2d(512)
        self.conv150 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn150 = nn.BatchNorm2d(2048)
        
        
        
        
        self.conva = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(256)
        
        self.convb = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb = nn.BatchNorm2d(512)
        self.convb1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb1 = nn.BatchNorm2d(512)
        self.convc = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnc = nn.BatchNorm2d(512)
        self.convd = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(512)
        
        
        self.conve = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne = nn.BatchNorm2d(1024)
        self.conve1 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne1 = nn.BatchNorm2d(1024)
        self.convf = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(1024)
        self.convg = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bng = nn.BatchNorm2d(1024)
        
        self.convh = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnh = nn.BatchNorm2d(2048)
        self.convh1 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnh1 = nn.BatchNorm2d(2048)
        self.convi = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bni = nn.BatchNorm2d(2048)
        self.convj = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnj = nn.BatchNorm2d(2048)
        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(2048, out_planes)

    def forward(self, x):
        k2 = 23/12
        k1 = -16/12
        k0 = 5/12 
        
        out = F.relu(self.bn0(self.conv0(x)))
        y = out
        out = self.bn3(self.conv3(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(out))))))))
        fy0 = out
        out = self.bna(self.conva(y)) + fy0
        out = F.relu(out)
        y = out
        out = self.bn6(self.conv6(F.relu(self.bn5(self.conv5(F.relu(self.bn4(self.conv4(out))))))))
        
        fy1 = out
        out = fy1 + y
        out = F.relu(out)
        y = out
        out = self.bn9(self.conv9(F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn12(self.conv12(F.relu(self.bn11(self.conv11(F.relu(self.bn10(self.conv10(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bnb(self.convb(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bnc(self.convc(F.avg_pool2d(fy0,2))))
        out = out + self.bnd(self.convd(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn15(self.conv15(F.relu(self.bn14(self.conv14(F.relu(self.bn13(self.conv13(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bnb1(self.convb1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn18(self.conv18(F.relu(self.bn17(self.conv17(F.relu(self.bn16(self.conv16(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn21(self.conv21(F.relu(self.bn20(self.conv20(F.relu(self.bn19(self.conv19(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn24(self.conv24(F.relu(self.bn23(self.conv23(F.relu(self.bn22(self.conv22(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn27(self.conv27(F.relu(self.bn26(self.conv26(F.relu(self.bn25(self.conv25(out))))))))
        '''
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn30(self.conv30(F.relu(self.bn29(self.conv29(F.relu(self.bn28(self.conv28(out))))))))
        '''
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn33(self.conv33(F.relu(self.bn32(self.conv32(F.relu(self.bn31(self.conv31(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn36(self.conv36(F.relu(self.bn35(self.conv35(F.relu(self.bn34(self.conv34(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bne(self.conve(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bnf(self.convf(F.avg_pool2d(fy0,2))))
        out = out + self.bng(self.convg(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn39(self.conv39(F.relu(self.bn38(self.conv38(F.relu(self.bn37(self.conv37(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bne1(self.conve1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn42(self.conv42(F.relu(self.bn41(self.conv41(F.relu(self.bn40(self.conv40(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn45(self.conv45(F.relu(self.bn44(self.conv44(F.relu(self.bn43(self.conv43(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn48(self.conv48(F.relu(self.bn47(self.conv47(F.relu(self.bn46(self.conv46(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn51(self.conv51(F.relu(self.bn50(self.conv50(F.relu(self.bn49(self.conv49(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn54(self.conv54(F.relu(self.bn53(self.conv53(F.relu(self.bn52(self.conv52(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn57(self.conv57(F.relu(self.bn56(self.conv56(F.relu(self.bn55(self.conv55(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn60(self.conv60(F.relu(self.bn59(self.conv59(F.relu(self.bn58(self.conv58(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn63(self.conv63(F.relu(self.bn62(self.conv62(F.relu(self.bn61(self.conv61(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn66(self.conv66(F.relu(self.bn65(self.conv65(F.relu(self.bn64(self.conv64(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn69(self.conv69(F.relu(self.bn68(self.conv68(F.relu(self.bn67(self.conv67(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn72(self.conv72(F.relu(self.bn71(self.conv71(F.relu(self.bn70(self.conv70(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn75(self.conv75(F.relu(self.bn74(self.conv74(F.relu(self.bn73(self.conv73(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn78(self.conv78(F.relu(self.bn77(self.conv77(F.relu(self.bn76(self.conv76(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn81(self.conv81(F.relu(self.bn80(self.conv80(F.relu(self.bn79(self.conv79(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn84(self.conv84(F.relu(self.bn83(self.conv83(F.relu(self.bn82(self.conv82(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn87(self.conv87(F.relu(self.bn86(self.conv86(F.relu(self.bn85(self.conv85(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn90(self.conv90(F.relu(self.bn89(self.conv89(F.relu(self.bn88(self.conv88(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn93(self.conv93(F.relu(self.bn92(self.conv92(F.relu(self.bn91(self.conv91(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn96(self.conv96(F.relu(self.bn95(self.conv95(F.relu(self.bn94(self.conv94(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn99(self.conv99(F.relu(self.bn98(self.conv98(F.relu(self.bn97(self.conv97(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn102(self.conv102(F.relu(self.bn101(self.conv101(F.relu(self.bn100(self.conv100(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn105(self.conv105(F.relu(self.bn104(self.conv104(F.relu(self.bn103(self.conv103(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn108(self.conv108(F.relu(self.bn107(self.conv107(F.relu(self.bn106(self.conv106(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn111(self.conv111(F.relu(self.bn110(self.conv110(F.relu(self.bn109(self.conv109(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn114(self.conv114(F.relu(self.bn113(self.conv113(F.relu(self.bn112(self.conv112(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn117(self.conv117(F.relu(self.bn116(self.conv116(F.relu(self.bn115(self.conv115(out))))))))
        
        '''fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn120(self.conv120(F.relu(self.bn119(self.conv119(F.relu(self.bn118(self.conv118(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn123(self.conv123(F.relu(self.bn122(self.conv122(F.relu(self.bn121(self.conv121(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn126(self.conv126(F.relu(self.bn125(self.conv125(F.relu(self.bn124(self.conv124(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn129(self.conv129(F.relu(self.bn128(self.conv128(F.relu(self.bn127(self.conv127(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn132(self.conv132(F.relu(self.bn131(self.conv131(F.relu(self.bn130(self.conv130(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn135(self.conv135(F.relu(self.bn134(self.conv134(F.relu(self.bn133(self.conv133(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn138(self.conv138(F.relu(self.bn137(self.conv137(F.relu(self.bn136(self.conv136(out))))))))
        '''
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn141(self.conv141(F.relu(self.bn140(self.conv140(F.relu(self.bn139(self.conv139(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn144(self.conv144(F.relu(self.bn143(self.conv143(F.relu(self.bn142(self.conv142(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bnh(self.convh(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bni(self.convi(F.avg_pool2d(fy0,2))))
        out = out + self.bnj(self.convj(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn147(self.conv147(F.relu(self.bn146(self.conv146(F.relu(self.bn145(self.conv145(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bnh1(self.convh1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn150(self.conv150(F.relu(self.bn149(self.conv149(F.relu(self.bn148(self.conv148(out))))))))
        
        
        
        
        
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class AB3ResNet101(nn.Module):
    def __init__(self, in_planes,out_planes, stride=1):
        super(AB3ResNet101, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        #B1_1
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        #B1_2
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        #B1_3
        self.conv7 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        
        #B2_1
        self.conv10 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        #B2_2
        self.conv13 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        #B2_3
        self.conv16 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn16 = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(128)
        self.conv18 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn18 = nn.BatchNorm2d(512)
        #B2_4
        self.conv19 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn19 = nn.BatchNorm2d(128)
        self.conv20 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(512)
        
        #B3_1
        self.conv22 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(256)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(256)
        self.conv24 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn24 = nn.BatchNorm2d(1024)
        #B3_2
        self.conv25 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn25 = nn.BatchNorm2d(256)
        self.conv26 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(256)
        self.conv27 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn27 = nn.BatchNorm2d(1024)
        #B3_3
        self.conv28 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn28 = nn.BatchNorm2d(256)
        self.conv29 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(256)
        self.conv30 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn30 = nn.BatchNorm2d(1024)
        #B3_4
        self.conv31 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn33 = nn.BatchNorm2d(1024)
        #B3_5
        self.conv34 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn34 = nn.BatchNorm2d(256)
        self.conv35 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn35 = nn.BatchNorm2d(256)
        self.conv36 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn36 = nn.BatchNorm2d(1024)
        #B3_6
        self.conv37 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn37 = nn.BatchNorm2d(256)
        self.conv38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn38 = nn.BatchNorm2d(256)
        self.conv39 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn39 = nn.BatchNorm2d(1024)
        #B3_7
        self.conv40 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn40 = nn.BatchNorm2d(256)
        self.conv41 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(256)
        self.conv42 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn42 = nn.BatchNorm2d(1024)
        #B3_8
        self.conv43 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn43 = nn.BatchNorm2d(256)
        self.conv44 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn44 = nn.BatchNorm2d(256)
        self.conv45 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn45 = nn.BatchNorm2d(1024)
        #B3_9
        self.conv46 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn46 = nn.BatchNorm2d(256)
        self.conv47 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn47 = nn.BatchNorm2d(256)
        self.conv48 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn48 = nn.BatchNorm2d(1024)
        #B3_10
        self.conv49 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn49 = nn.BatchNorm2d(256)
        self.conv50 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn50 = nn.BatchNorm2d(256)
        self.conv51 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn51 = nn.BatchNorm2d(1024)
        #B3_11
        self.conv52 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn52 = nn.BatchNorm2d(256)
        self.conv53 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn53 = nn.BatchNorm2d(256)
        self.conv54 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn54 = nn.BatchNorm2d(1024)
        #B3_12
        self.conv55 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn55 = nn.BatchNorm2d(256)
        self.conv56 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn56 = nn.BatchNorm2d(256)
        self.conv57 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn57 = nn.BatchNorm2d(1024)
        #B3_13
        self.conv58 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn58 = nn.BatchNorm2d(256)
        self.conv59 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn59 = nn.BatchNorm2d(256)
        self.conv60 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn60 = nn.BatchNorm2d(1024)
        #B3_14
        self.conv61 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn61 = nn.BatchNorm2d(256)
        self.conv62 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(256)
        self.conv63 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn63 = nn.BatchNorm2d(1024)
        '''#B3_15
        self.conv64 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn64 = nn.BatchNorm2d(256)
        self.conv65 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn65 = nn.BatchNorm2d(256)
        self.conv66 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn66 = nn.BatchNorm2d(1024)
        #B3_16
        self.conv67 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn67 = nn.BatchNorm2d(256)
        self.conv68 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn68 = nn.BatchNorm2d(256)
        self.conv69 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn69 = nn.BatchNorm2d(1024)
        #B3_17
        self.conv70 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn70 = nn.BatchNorm2d(256)
        self.conv71 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(256)
        self.conv72 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn72 = nn.BatchNorm2d(1024)
        #B3_18
        self.conv73 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn73 = nn.BatchNorm2d(256)
        self.conv74 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn74 = nn.BatchNorm2d(256)
        self.conv75 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn75 = nn.BatchNorm2d(1024)
        #B3_19
        self.conv76 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn76 = nn.BatchNorm2d(256)
        self.conv77 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn77 = nn.BatchNorm2d(256)
        self.conv78 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn78 = nn.BatchNorm2d(1024)
        #B3_20
        self.conv79 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn79 = nn.BatchNorm2d(256)
        self.conv80 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn80 = nn.BatchNorm2d(256)
        self.conv81 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(1024)
        #B3_21
        self.conv82 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn82 = nn.BatchNorm2d(256)
        self.conv83 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(256)
        self.conv84 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn84 = nn.BatchNorm2d(1024)'''
        #B3_22
        self.conv85 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn85 = nn.BatchNorm2d(256)
        self.conv86 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn86 = nn.BatchNorm2d(256)
        self.conv87 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn87 = nn.BatchNorm2d(1024)
        #B3_23
        self.conv88 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn88 = nn.BatchNorm2d(256)
        self.conv89 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn89 = nn.BatchNorm2d(256)
        self.conv90 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn90 = nn.BatchNorm2d(1024)
        
        #B4_1
        self.conv91 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(512)
        self.conv92 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(512)
        self.conv93 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn93 = nn.BatchNorm2d(2048)
        #B4_2
        self.conv94 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn94 = nn.BatchNorm2d(512)
        self.conv95 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn95 = nn.BatchNorm2d(512)
        self.conv96 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn96 = nn.BatchNorm2d(2048)
        #B4_3
        self.conv97 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn97 = nn.BatchNorm2d(512)
        self.conv98 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn98 = nn.BatchNorm2d(512)
        self.conv99 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn99 = nn.BatchNorm2d(2048)
        
        
        
        
        self.conva = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(256)
        
        self.convb = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb = nn.BatchNorm2d(512)
        self.convb1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb1 = nn.BatchNorm2d(512)
        self.convc = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnc = nn.BatchNorm2d(512)
        self.convd = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(512)
        
        
        self.conve = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne = nn.BatchNorm2d(1024)
        self.conve1 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne1 = nn.BatchNorm2d(1024)
        self.convf = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(1024)
        self.convg = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bng = nn.BatchNorm2d(1024)
        
        self.convh = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnh = nn.BatchNorm2d(2048)
        self.convh1 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnh1 = nn.BatchNorm2d(2048)
        self.convi = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bni = nn.BatchNorm2d(2048)
        self.convj = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnj = nn.BatchNorm2d(2048)
        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(2048, out_planes)

    def forward(self, x):
        k2 = 23/12
        k1 = -16/12
        k0 = 5/12 
        
        out = F.relu(self.bn0(self.conv0(x)))
        y = out
        out = self.bn3(self.conv3(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(out))))))))
        fy0 = out
        out = self.bna(self.conva(y)) + fy0
        out = F.relu(out)
        y = out
        out = self.bn6(self.conv6(F.relu(self.bn5(self.conv5(F.relu(self.bn4(self.conv4(out))))))))
        
        fy1 = out
        out = fy1 + y
        out = F.relu(out)
        y = out
        out = self.bn9(self.conv9(F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn12(self.conv12(F.relu(self.bn11(self.conv11(F.relu(self.bn10(self.conv10(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bnb(self.convb(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bnc(self.convc(F.avg_pool2d(fy0,2))))
        out = out + self.bnd(self.convd(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn15(self.conv15(F.relu(self.bn14(self.conv14(F.relu(self.bn13(self.conv13(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bnb1(self.convb1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn18(self.conv18(F.relu(self.bn17(self.conv17(F.relu(self.bn16(self.conv16(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn21(self.conv21(F.relu(self.bn20(self.conv20(F.relu(self.bn19(self.conv19(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn24(self.conv24(F.relu(self.bn23(self.conv23(F.relu(self.bn22(self.conv22(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bne(self.conve(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bnf(self.convf(F.avg_pool2d(fy0,2))))
        out = out + self.bng(self.convg(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn27(self.conv27(F.relu(self.bn26(self.conv26(F.relu(self.bn25(self.conv25(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bne1(self.conve1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn30(self.conv30(F.relu(self.bn29(self.conv29(F.relu(self.bn28(self.conv28(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn33(self.conv33(F.relu(self.bn32(self.conv32(F.relu(self.bn31(self.conv31(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn36(self.conv36(F.relu(self.bn35(self.conv35(F.relu(self.bn34(self.conv34(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn39(self.conv39(F.relu(self.bn38(self.conv38(F.relu(self.bn37(self.conv37(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn42(self.conv42(F.relu(self.bn41(self.conv41(F.relu(self.bn40(self.conv40(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn45(self.conv45(F.relu(self.bn44(self.conv44(F.relu(self.bn43(self.conv43(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn48(self.conv48(F.relu(self.bn47(self.conv47(F.relu(self.bn46(self.conv46(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn51(self.conv51(F.relu(self.bn50(self.conv50(F.relu(self.bn49(self.conv49(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn54(self.conv54(F.relu(self.bn53(self.conv53(F.relu(self.bn52(self.conv52(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn57(self.conv57(F.relu(self.bn56(self.conv56(F.relu(self.bn55(self.conv55(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn60(self.conv60(F.relu(self.bn59(self.conv59(F.relu(self.bn58(self.conv58(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn63(self.conv63(F.relu(self.bn62(self.conv62(F.relu(self.bn61(self.conv61(out))))))))
        
        '''fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn66(self.conv66(F.relu(self.bn65(self.conv65(F.relu(self.bn64(self.conv64(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn69(self.conv69(F.relu(self.bn68(self.conv68(F.relu(self.bn67(self.conv67(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn72(self.conv72(F.relu(self.bn71(self.conv71(F.relu(self.bn70(self.conv70(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn75(self.conv75(F.relu(self.bn74(self.conv74(F.relu(self.bn73(self.conv73(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn78(self.conv78(F.relu(self.bn77(self.conv77(F.relu(self.bn76(self.conv76(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn81(self.conv81(F.relu(self.bn80(self.conv80(F.relu(self.bn79(self.conv79(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn84(self.conv84(F.relu(self.bn83(self.conv83(F.relu(self.bn82(self.conv82(out))))))))
        '''
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn87(self.conv87(F.relu(self.bn86(self.conv86(F.relu(self.bn85(self.conv85(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn90(self.conv90(F.relu(self.bn89(self.conv89(F.relu(self.bn88(self.conv88(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn93(self.conv93(F.relu(self.bn92(self.conv92(F.relu(self.bn91(self.conv91(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bnh(self.convh(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bni(self.convi(F.avg_pool2d(fy0,2))))
        out = out + self.bnj(self.convj(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn96(self.conv96(F.relu(self.bn95(self.conv95(F.relu(self.bn94(self.conv94(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bnh1(self.convh1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn99(self.conv99(F.relu(self.bn98(self.conv98(F.relu(self.bn97(self.conv97(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class AB3ResNet50(nn.Module):
    def __init__(self, in_planes,out_planes, stride=1):
        super(AB3ResNet50, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        #B1_1
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        #B1_2
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        #B1_3
        self.conv7 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        
        #B2_1
        self.conv10 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn12 = nn.BatchNorm2d(512)
        #B2_2
        self.conv13 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        #B2_3
        self.conv16 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn16 = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(128)
        self.conv18 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn18 = nn.BatchNorm2d(512)
        #B2_4
        self.conv19 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn19 = nn.BatchNorm2d(128)
        self.conv20 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(128)
        self.conv21 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(512)
        
        #B3_1
        self.conv22 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(256)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(256)
        self.conv24 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn24 = nn.BatchNorm2d(1024)
        #B3_2
        self.conv25 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn25 = nn.BatchNorm2d(256)
        self.conv26 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(256)
        self.conv27 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn27 = nn.BatchNorm2d(1024)
        #B3_3
        self.conv28 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn28 = nn.BatchNorm2d(256)
        self.conv29 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(256)
        self.conv30 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn30 = nn.BatchNorm2d(1024)
        '''#B3_4
        self.conv31 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn33 = nn.BatchNorm2d(1024)
        #B3_5
        self.conv34 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn34 = nn.BatchNorm2d(256)
        self.conv35 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn35 = nn.BatchNorm2d(256)
        self.conv36 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn36 = nn.BatchNorm2d(1024)'''
        #B3_6
        self.conv37 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn37 = nn.BatchNorm2d(256)
        self.conv38 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn38 = nn.BatchNorm2d(256)
        self.conv39 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn39 = nn.BatchNorm2d(1024)
        
        #B4_1
        self.conv40 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn40 = nn.BatchNorm2d(512)
        self.conv41 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn42 = nn.BatchNorm2d(2048)
        #B4_2
        self.conv43 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn43 = nn.BatchNorm2d(512)
        self.conv44 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn44 = nn.BatchNorm2d(512)
        self.conv45 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn45 = nn.BatchNorm2d(2048)
        #B4_3
        self.conv46 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn46 = nn.BatchNorm2d(512)
        self.conv47 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn47 = nn.BatchNorm2d(512)
        self.conv48 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn48 = nn.BatchNorm2d(2048)
        
        
        
        
        self.conva = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(256)
        
        self.convb = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb = nn.BatchNorm2d(512)
        self.convb1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb1 = nn.BatchNorm2d(512)
        self.convc = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnc = nn.BatchNorm2d(512)
        self.convd = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(512)
        
        
        self.conve = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne = nn.BatchNorm2d(1024)
        self.conve1 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne1 = nn.BatchNorm2d(1024)
        self.convf = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(1024)
        self.convg = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bng = nn.BatchNorm2d(1024)
        
        self.convh = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnh = nn.BatchNorm2d(2048)
        self.convh1 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnh1 = nn.BatchNorm2d(2048)
        self.convi = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bni = nn.BatchNorm2d(2048)
        self.convj = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnj = nn.BatchNorm2d(2048)
        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(2048, out_planes)

    def forward(self, x):
        k2 = 23/12
        k1 = -16/12
        k0 = 5/12 
        
        out = F.relu(self.bn0(self.conv0(x)))
        y = out
        out = self.bn3(self.conv3(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(out))))))))
        fy0 = out
        out = self.bna(self.conva(y)) + fy0
        out = F.relu(out)
        y = out
        out = self.bn6(self.conv6(F.relu(self.bn5(self.conv5(F.relu(self.bn4(self.conv4(out))))))))
        
        fy1 = out
        out = fy1 + y
        out = F.relu(out)
        y = out
        out = self.bn9(self.conv9(F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn12(self.conv12(F.relu(self.bn11(self.conv11(F.relu(self.bn10(self.conv10(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bnb(self.convb(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bnc(self.convc(F.avg_pool2d(fy0,2))))
        out = out + self.bnd(self.convd(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn15(self.conv15(F.relu(self.bn14(self.conv14(F.relu(self.bn13(self.conv13(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bnb1(self.convb1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn18(self.conv18(F.relu(self.bn17(self.conv17(F.relu(self.bn16(self.conv16(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn21(self.conv21(F.relu(self.bn20(self.conv20(F.relu(self.bn19(self.conv19(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn24(self.conv24(F.relu(self.bn23(self.conv23(F.relu(self.bn22(self.conv22(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bne(self.conve(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bnf(self.convf(F.avg_pool2d(fy0,2))))
        out = out + self.bng(self.convg(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn27(self.conv27(F.relu(self.bn26(self.conv26(F.relu(self.bn25(self.conv25(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bne1(self.conve1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn30(self.conv30(F.relu(self.bn29(self.conv29(F.relu(self.bn28(self.conv28(out))))))))
        
        '''fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn33(self.conv33(F.relu(self.bn32(self.conv32(F.relu(self.bn31(self.conv31(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn36(self.conv36(F.relu(self.bn35(self.conv35(F.relu(self.bn34(self.conv34(out))))))))'''
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn39(self.conv39(F.relu(self.bn38(self.conv38(F.relu(self.bn37(self.conv37(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn42(self.conv42(F.relu(self.bn41(self.conv41(F.relu(self.bn40(self.conv40(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bnh(self.convh(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bni(self.convi(F.avg_pool2d(fy0,2))))
        out = out + self.bnj(self.convj(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn45(self.conv45(F.relu(self.bn44(self.conv44(F.relu(self.bn43(self.conv43(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bnh1(self.convh1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn48(self.conv48(F.relu(self.bn47(self.conv47(F.relu(self.bn46(self.conv46(out))))))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class AB3ResNet34(nn.Module):
    def __init__(self, in_planes, planes,out_planes, stride=1):
        super(AB3ResNet34, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv6 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)
        
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(128)
        
        self.conv15 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(256)
        self.conv16 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(256)
        self.conv17 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(256)
        self.conv18 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(256)
        self.conv19 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19 = nn.BatchNorm2d(256)
        self.conv20 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(256)
        self.conv21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(256)
        self.conv22 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(256)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(256)
        self.conv24 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24 = nn.BatchNorm2d(256)
        self.conv25 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn25 = nn.BatchNorm2d(256)
        self.conv26 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(256)
        
        self.conv27 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn27 = nn.BatchNorm2d(512)
        self.conv28 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn28 = nn.BatchNorm2d(512)
        self.conv29 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29 = nn.BatchNorm2d(512)
        self.conv30 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn30 = nn.BatchNorm2d(512)
        self.conv31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(512)
        self.conv32 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(512)
        
        
        self.conva = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(128)
        self.conva1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna1 = nn.BatchNorm2d(128)
        self.convb = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb = nn.BatchNorm2d(128)
        self.convc = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnc = nn.BatchNorm2d(128)
        
        self.convd = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(256)
        self.convd1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd1 = nn.BatchNorm2d(256)
        self.conve = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne = nn.BatchNorm2d(256)
        self.convf = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(256)
        
        self.convg = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bng = nn.BatchNorm2d(512)
        self.convg1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bng1 = nn.BatchNorm2d(512)
        self.convh = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnh = nn.BatchNorm2d(512)
        self.convi = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bni = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(512, out_planes)

    def forward(self, x):
        k2 = 23/12
        k1 = -16/12
        k0 = 5/12 
        out = F.relu(self.bn0(self.conv0(x)))
        y = out
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(out)))))
        fy0 = out
        out = out + y
        out = F.relu(out)
        y = out
        out = self.bn4(self.conv4(F.relu(self.bn3(self.conv3(out))))) 
        
        fy1 = out
        out = out + y
        out = F.relu(out)
        y = out
        out = self.bn6(self.conv6(F.relu(self.bn5(self.conv5(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn8(self.conv8(F.relu(self.bn7(self.conv7(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bna(self.conva(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bnb(self.convb(F.avg_pool2d(fy0,2))))
        out = out + self.bnc(self.convc(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn10(self.conv10(F.relu(self.bn9(self.conv9(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bna1(self.conva1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn12(self.conv12(F.relu(self.bn11(self.conv11(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn14(self.conv14(F.relu(self.bn13(self.conv13(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn16(self.conv16(F.relu(self.bn15(self.conv15(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bnd(self.convd(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bne(self.conve(F.avg_pool2d(fy0,2))))
        out = out + self.bnf(self.convf(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn18(self.conv18(F.relu(self.bn17(self.conv17(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bnd1(self.convd1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn20(self.conv20(F.relu(self.bn19(self.conv19(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn22(self.conv22(F.relu(self.bn21(self.conv21(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn24(self.conv24(F.relu(self.bn23(self.conv23(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn26(self.conv26(F.relu(self.bn25(self.conv25(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn28(self.conv28(F.relu(self.bn27(self.conv27(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bng(self.convg(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bnh(self.convh(F.avg_pool2d(fy0,2))))
        out = out + self.bni(self.convi(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn30(self.conv30(F.relu(self.bn29(self.conv29(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bng1(self.convg1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn32(self.conv32(F.relu(self.bn31(self.conv31(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * fy0)
        out = out + y
        out = F.relu(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

class AB3ResNet18(nn.Module):
    def __init__(self, in_planes, planes,out_planes, stride=1):
        super(AB3ResNet18, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        
        self.conv9 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        
        self.conv13 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(512)
        
        self.conva = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna = nn.BatchNorm2d(128)
        self.conva1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bna1 = nn.BatchNorm2d(128)
        self.convb = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnb = nn.BatchNorm2d(128)
        self.convc = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnc = nn.BatchNorm2d(128)
        
        self.convd = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(256)
        self.convd1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnd1 = nn.BatchNorm2d(256)
        self.conve = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bne = nn.BatchNorm2d(256)
        self.convf = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(256)
        
        self.convg = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bng = nn.BatchNorm2d(512)
        self.convg1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bng1 = nn.BatchNorm2d(512)
        self.convh = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnh = nn.BatchNorm2d(512)
        self.convi = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bni = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)
        
        self.linear = nn.Linear(512, out_planes)

    def forward(self, x):
        k2 = 23/12
        k1 = -16/12
        k0 = 5/12 
        out = F.relu(self.bn0(self.conv0(x)))
        y = out
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(out)))))
        fy0 = out
        out = out + y
        out = F.relu(out)
        y = out
        out = self.bn4(self.conv4(F.relu(self.bn3(self.conv3(out))))) 
        
        fy1 = out
        out = out + y
        out = F.relu(out)
        y = out
        out = self.bn6(self.conv6(F.relu(self.bn5(self.conv5(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bna(self.conva(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bnb(self.convb(F.avg_pool2d(fy0,2))))
        out = out + self.bnc(self.convc(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn8(self.conv8(F.relu(self.bn7(self.conv7(out)))))
        
        
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bna1(self.conva1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn10(self.conv10(F.relu(self.bn9(self.conv9(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bnd(self.convd(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bne(self.conve(F.avg_pool2d(fy0,2))))
        out = out + self.bnf(self.convf(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn12(self.conv12(F.relu(self.bn11(self.conv11(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bnd1(self.convd1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn14(self.conv14(F.relu(self.bn13(self.conv13(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * self.bng(self.convg(F.avg_pool2d(fy1,2))))
        out = out + (k0 * self.bnh(self.convh(F.avg_pool2d(fy0,2))))
        out = out + self.bni(self.convi(F.avg_pool2d(y,2)))
        out = F.relu(out)
        fy0 = fy1
        fy1 = fy2
        y = out
        out = self.bn16(self.conv16(F.relu(self.bn15(self.conv15(out)))))
        
        fy2 = out
        out = k2 * fy2
        out = out + (k1 * fy1)
        out = out + (k0 * self.bng1(self.convg1(F.avg_pool2d(fy0,2))))
        out = out + y
        out = F.relu(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

def AB3_ResNet18():
    return AB3ResNet18(3,64,10)

def AB3_ResNet34():
    return AB3ResNet34(3,64,10)

def AB3_ResNet50():
    return AB3ResNet50(3,10)

def AB3_ResNet101():
    return AB3ResNet101(3,10)

def AB3_ResNet152():
    return AB3ResNet152(3,10)

def AB3_Res100Net18():
    return AB3ResNet18(3,64,100)

def AB3_Res100Net34():
    return AB3ResNet34(3,64,100)

def AB3_Res100Net50():
    return AB3ResNet50(3,100)

def AB3_Res100Net101():
    return AB3ResNet101(3,100)

def AB3_Res100Net152():
    return AB3ResNet152(3,100)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
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
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
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
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

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
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])



def Res100Net18():
    return ResNet(BasicBlock, [2, 2, 2, 2], 100)


def Res100Net34():
    return ResNet(BasicBlock, [3, 4, 6, 3], 100)


def Res100Net50():
    return ResNet(Bottleneck, [3, 4, 6, 3], 100)


def Res100Net101():
    return ResNet(Bottleneck, [3, 4, 23, 3], 100)


def Res100Net152():
    return ResNet(Bottleneck, [3, 8, 36, 3], 100)