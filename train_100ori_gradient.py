#%%
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler
import time
import datetime as dt

import torchvision
import torchvision.transforms as transforms
# from torchstat import stat

import os
import argparse

from tensorboardX import SummaryWriter

from models import *
from datetime import datetime

from torchsummary import summary
# Training

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--tao', default=1, type=int)
parser.add_argument('--nnlayer', default=1, type=int)
parser.add_argument('--repnum', default=1, type=int)
parser.add_argument('--h', default=1, type=float, help='h')
parser.add_argument('--b1', default=1.5, type=float, help='learning rate')
parser.add_argument('--b2', default=-0.5, type=float, help='learning rate')
parser.add_argument('--b3', default=1.0, type=float, help='learning rate')
parser.add_argument('--a0', default=1.0, type=float, help='learning rate')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
parser.add_argument('--epoch', type=int, default=200, help='training epoch')
parser.add_argument('--warm', type=int, default=1, help='warm up training phase')

                                    
parser.add_argument('-d', '--data', default='./data', type=str)
parser.add_argument('--arch', '-a', default='ResNet18', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-bs', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize (default: 200)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=250, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# Optimization options
parser.add_argument('--opt_level', default='O2', type=str,
                    help='O2 is fast mixed FP16/32 training, O0 (FP32 training) and O3 (FP16 training), O1 ("conservative mixed precision"), O2 ("fast mixed precision").--opt_level O1 and O2 both use dynamic loss scaling by default unless manually overridden. --opt-level O0 and O3 (the "pure" training modes) do not use loss scaling by default. See more in https://github.com/NVIDIA/apex/tree/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet')
parser.add_argument('--keep-batchnorm-fp32', default=True, action='store_true',
                    help='keeping cudnn bn leads to fast training')
parser.add_argument('--loss-scale', type=float, default=None)
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('-t', '--test', action='store_true',
                    help='Launch test mode with preset arguments')
parser.add_argument('--warmup', '--wp', default=5, type=int,
                    help='number of epochs to warmup')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 4e-5 for mobile models)')
parser.add_argument('--wd-all', dest = 'wdall', action='store_true',
                    help='weight decay on all parameters')

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--ex", default=0, type=int)
parser.add_argument("--alpha", default=0.1, type=float)
parser.add_argument("--beta", default=15.0, type=float)
parser.add_argument("--notes", default='', type=str)

args = parser.parse_args()
# args.save_path = 'runs/cifar100/'+'cifar100_' + args.arch+'_/Ori'+\
#                                     '_BS'+str(args.batch_size)+'_LR'+\
#                                    str(args.lr)+'epoch_'+\
#                                    str(args.epoch)+'warmup'+str(args.warm)+\
#                                    args.notes+\
#                                     "{0:%Y-%m-%dT%H-%M/}".format(datetime.now())

#2-step
args.save_path='log/10.29/'+str(args.arch)+'b1_'+str(args.b1)+'b2_'+str(args.b2)+"{0:%Y-%m-%dT%H-%M/}".format(datetime.now())
# #3-step
# args.save_path='log/10.29/'+str(args.arch)+'b1_'+str(args.b1)+'b2_'+str(args.b2)+'b3_'+str(args.b3)+"{0:%Y-%m-%dT%H-%M/}".format(datetime.now())


ite_num=1

def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()
    net.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # if epoch <= args.warm:
            # warmup_scheduler.step()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        loss.backward()
        optimizer.step()
        
        #########################
        
        global ite_num
        grads = []
        for param in net.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        # print(torch.max(torch.abs(grads)).item())  #打印网络所有梯度中最大的梯度
        writer.add_scalar('gradient_max',torch.max(torch.abs(grads)).item(),ite_num)
        writer.add_scalar('gradient_mean',torch.mean(torch.abs(grads)).item(),ite_num)
        ite_num=ite_num+1
        #########################
        

        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: {:.1f}, Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, losses.avg, top1.avg))
    writer.add_scalar('Train/Average loss', losses.avg, epoch)
    writer.add_scalar('Train/Accuracy-top1', top1.avg, epoch)
    writer.add_scalar('Train/Accuracy-top5', top5.avg, epoch)
    writer.add_scalar('Train/Time', batch_time.sum, epoch)
    #for name, param in net.named_parameters():
        #layer, attr = os.path.splitext(name)
        #attr = attr[1:]
        #writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    return top1.avg, losses.avg, batch_time.sum


def test(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()
    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            # test_loss += loss.item()
            # _, preds = outputs.max(1)
            # correct += preds.eq(old_labels).sum()
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', losses.avg, epoch)
    writer.add_scalar('Test/Accuracy-top1', top1.avg, epoch)
    writer.add_scalar('Test/Accuracy-top5', top5.avg, epoch)
    writer.add_scalar('Test/Time', batch_time.sum, epoch)
    
    return top1.avg, losses.avg, batch_time.sum
            


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
               
if __name__ == '__main__':
  

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, args.batch_size, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, args.batch_size, shuffle=False, num_workers=args.workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Model
    print('==> Building model {}'.format( args.arch ))
    print('==> b1 {}'.format( args.b1 ))
    print('==> b2 {}'.format( args.b2 ))
    print('==> b3 {}'.format( args.b3 ))
    print('==> a0 {}'.format( args.a0 ))
    print('==> Train info:')
    print('==> Start DateTime: {}'.format( dt.datetime.now() ))
    print('==> Device: {}'.format( args.notes ))
    #net = VGG('VGG16')
    #net_name = "ResNet50"
    #model_name = "ResNet50"
    net_name = args.arch
    model_name = args.arch
    if args.arch == "r18" :
        net = R100esNet18()
    elif args.arch=="r34" :
        net = R100esNet34()
    elif args.arch=="r50" :
        net = R100esNet50()
    elif args.arch == "r18v2" :
        net = R100esNet18V2()
    elif args.arch=="r34v2" :
        net = R100esNet34V2()
    elif args.arch=="r50v2" :
        net = R100esNet50V2()
    elif args.arch=="r101" :
        net = R100esNet101()
    elif args.arch=="r152" :
        net = R100esNet152()
    elif args.arch == "p100r18" :
        net = PreAct100ResNet18()
    elif args.arch=="p100r34" :
        net = PreAct100ResNet34()
    elif args.arch=="p100r50" :
        net = PreAct100ResNet50()
    elif args.arch=="p100r101" :
        net = PreAct100ResNet101()
    elif args.arch=="p100r152" :
        net = PreAct100ResNet152()
    elif args.arch=="pr200" :
        net = PreAct100ResNet200()
    elif args.arch=="pr299" :
        net = PreAct100ResNet299()
    elif args.arch=='tr6v22':
        net = T100_ResNet6V2_2()
    elif args.arch=='tr6v23':
        net = T100_ResNet6V2_3()
    elif args.arch=='tr10v2':
        net = T100_ResNet10V2()
    elif args.arch=='tr12v2':
        net = T100_ResNet12V2()
    elif args.arch=='tr14v2':
        net = T100_ResNet14V2()
    elif args.arch=='tr21v2':
        net = T100_ResNet21V2()
    elif args.arch=='trs32':
        net = Ts100_ResNet32()
    elif args.arch=='trs36':
        net = Ts100_ResNet36()
    elif args.arch=='trs70':
        net = Ts100_ResNet70()
    elif args.arch=='trs84':
        net = Ts100_ResNet84()
    elif args.arch=='tini14':
        net = TiniResNet14_100()
    elif args.arch=='tini28':
        net = TiniResNet28_100()
    elif args.arch=='tini44':
        net = TiniResNet44_100()
    elif args.arch=='tini89':
        net = TiniResNet89_100()
    elif args.arch=='tini140':
        net = TiniResNet140_100()
    elif args.arch == "t2218" :
        net = T22_ResNet18()
    elif args.arch == "t2222" :
        net = T22_ResNet22()
    elif args.arch=="t2236" :
        net = T22_ResNet36()
    elif args.arch=="t2244" :
        net = T22_ResNet44()
    elif args.arch == "t2286" :
        net = T22_ResNet86()
    elif args.arch == "t22137" :
        net = T22_ResNet137()
    elif args.arch == "t22188" :
        net = T22_ResNet188()
    elif args.arch == "t22275" :
        net = T22_ResNet275()
    elif args.arch=="ab2p100r152" :
        net = AB2_PreAct100ResNet152()
    elif args.arch=="ab2p100r101" :
        net = AB2_PreAct100ResNet101()
    elif args.arch=="ab2p100r50" :
        net = AB2_PreAct100ResNet50()
    elif args.arch=="ab2p100r34" :
        net = AB2_PreAct100ResNet34()
    elif args.arch=="ab2p100r18" :
        net = AB2_PreAct100ResNet18()
    
    elif args.arch=="100ab2r18" :
        net = AB2_Res100Net18()
    elif args.arch=="100ab2r34" :
        net = AB2_Res100Net34()
    elif args.arch=="100ab2r50" :
        net = AB2_Res100Net50()
    elif args.arch=="100ab2r101" :
        net = AB2_Res100Net101()
    elif args.arch=="100ab2r152" :
        net = AB2_Res100Net152()
    
    elif args.arch=="ab3_100r18" :
        net = AB3_Res100Net18()
    elif args.arch=="ab3_100r34" :
        net = AB3_Res100Net34()
    elif args.arch=="ab3_100r50" :
        net = AB3_Res100Net50()
    elif args.arch=="ab3_100r101" :
        net = AB3_Res100Net101()
    elif args.arch=="ab3_100r152" :
        net = AB3_Res100Net152()
        
    elif args.arch=="100r18" :
        net = Res100Net18()
    elif args.arch=="100r34" :
        net = Res100Net34()
    elif args.arch=="100r50" :
        net = Res100Net50() 
    elif args.arch=="100r101" :
        net = Res100Net101()   
    elif args.arch=="100r150" :
        net = Res100Net152()    
    
    elif args.arch=="100ab1r20" :
        net = ab1res100net20(h=args.h, b1=args.k1)
    elif args.arch=="100ab1r32" :
        net = ab1res100net32(h=args.h, b1=args.k1)
    elif args.arch=="100ab1r44" :
        net = ab1res100net44(h=args.h, b1=args.k1)
    elif args.arch=="100ab1r56" :
        net = ab1res100net56(h=args.h, b1=args.k1)
    elif args.arch=="100ab1r98" :
        net = ab1res100net98(h=args.h, b1=args.k1)
    elif args.arch=="100ab1r104" :
        net = ab1res100net104(h=args.h, b1=args.k1)
    elif args.arch=="100ab1r110" :
        net = ab1res100net110(h=args.h, b1=args.k1)
    elif args.arch=="100ab1r116" :
        net = ab1res100net116(h=args.h, b1=args.k1)
    elif args.arch=="100ab1r122" :
        net = ab1res100net122(h=args.h, b1=args.k1)
    elif args.arch=="100ab1r128" :
        net = ab1res100net128(h=args.h, b1=args.k1)
    elif args.arch=="100ab1r134" :
        net = ab1res100net134(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r140" :
        net = ab1res100net140(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r146" :
        net = ab1res100net146(h=args.h,b1=args.k1)        
    elif args.arch=="100ab1r152" :
        net = ab1res100net152(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r158" :
        net = ab1res100net158(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r164" :
        net = ab1res100net164(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r170" :
        net = ab1res100net170(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r176" :
        net = ab1res100net176(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r182" :
        net = ab1res100net182(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r188" :
        net = ab1res100net188(h=args.h,b1=args.k1)   
    elif args.arch=="100ab1r218" :
        net = ab1res100net218(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r248" :
        net = ab1res100net248(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r278" :
        net = ab1res100net278(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r308" :
        net = ab1res100net308(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r338" :
        net = ab1res100net338(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r368" :
        net = ab1res100net368(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r398" :
        net = ab1res100net398(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r428" :
        net = ab1res100net428(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r458" :
        net = ab1res100net458(h=args.h,b1=args.k1)
    elif args.arch=="100ab1r488" :
        net = ab1res100net488(h=args.h,b1=args.k1)
        
        
    
    elif args.arch=="100ab2r20" :
        net = ab2res100net20(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r32" :
        net = ab2res100net32(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r44" :
        net = ab2res100net44(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r56" :
        net = ab2res100net56(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r68" :
        net = ab2res100net68(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r80" :
        net = ab2res100net80(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r92" :
        net = ab2res100net92(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r104" :
        net = ab2res100net104(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r110" :
        net = ab2res100net110(h=args.h, b1=args.b1, b2=args.b2)
        
    elif args.arch=="100ab2r20_a" :
        net = ab2res100net20_a(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r32_a" :
        net = ab2res100net32_a(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r44_a" :
        net = ab2res100net44_a(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r56_a" :
        net = ab2res100net56_a(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r68_a" :
        net = ab2res100net68_a(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r80_a" :
        net = ab2res100net80_a(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r92_a" :
        net = ab2res100net92_a(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r104_a" :
        net = ab2res100net104_a(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r110_a" :
        net = ab2res100net110_a(h=args.h, b1=args.b1, b2=args.b2)
        
    elif args.arch=="100ab2r20_b" :
        net = ab2res100net20_b(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r32_b" :
        net = ab2res100net32_b(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r44_b" :
        net = ab2res100net44_b(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r56_b" :
        net = ab2res100net56_b(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r68_b" :
        net = ab2res100net68_b(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r80_b" :
        net = ab2res100net80_b(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r92_b" :
        net = ab2res100net92_b(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r104_b" :
        net = ab2res100net104_b(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r110_b" :
        net = ab2res100net110_b(h=args.h, b1=args.b1, b2=args.b2)
        
    elif args.arch=="100ab2r20_c" :
        net = ab2res100net20_c(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r32_c" :
        net = ab2res100net32_c(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r44_c" :
        net = ab2res100net44_c(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r56_c" :
        net = ab2res100net56_c(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r68_c" :
        net = ab2res100net68_c(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r80_c" :
        net = ab2res100net80_c(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r92_c" :
        net = ab2res100net92_c(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r104_c" :
        net = ab2res100net104_c(h=args.h, b1=args.b1, b2=args.b2)
    elif args.arch=="100ab2r110_c" :
        net = ab2res100net110_c(h=args.h, b1=args.b1, b2=args.b2)

    elif args.arch=="100ab3r68" :
        net = ab3res100net68(h=args.h, b1=args.b1, b2=args.b2, b3=args.b3)
    elif args.arch=="100ab3r80" :
        net = ab3res100net80(h=args.h, b1=args.b1, b2=args.b2, b3=args.b3)
    elif args.arch=="100ab3r92" :
        net = ab3res100net92(h=args.h, b1=args.b1, b2=args.b2, b3=args.b3)
    elif args.arch=="100ab3r104" :
        net = ab3res100net104(h=args.h, b1=args.b1, b2=args.b2, b3=args.b3)
    
    elif args.arch=="100ab3r20" :
        net = ab3res100net20(h=args.h, b1=args.b1, b2=args.b2, b3=args.b3)
    elif args.arch=="100ab3r26" :
        net = ab3res100net26(h=args.h, b1=args.b1, b2=args.b2, b3=args.b3)
    elif args.arch=="100ab3r32" :
        net = ab3res100net32(h=args.h, b1=args.b1, b2=args.b2, b3=args.b3)
    elif args.arch=="100ab3r38" :
        net = ab3res100net38(h=args.h, b1=args.b1, b2=args.b2, b3=args.b3)
    elif args.arch=="100ab3r44" :
        net = ab3res100net44(h=args.h, b1=args.b1, b2=args.b2, b3=args.b3)
    elif args.arch=="100ab3r56" :
        net = ab3res100net56(h=args.h, b1=args.b1, b2=args.b2, b3=args.b3)
    elif args.arch=="100ab3r110" :
        net = ab3res100net110(h=args.h, b1=args.b1, b2=args.b2, b3=args.b3)
        
    elif args.arch=="100r20" :
        net = res100net20()
    elif args.arch=="100r32" :
        net = res100net32()
    elif args.arch=="100r44" :
        net = res100net44()
    elif args.arch=="100r56" :
        net = res100net56() 
    elif args.arch=="100r110" :
        net = res100net110()
        
    net = net.to(device)
    
    summary(net, (3, 32, 32))
    
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    #optimizer = SGD_atan(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, alpha=args.alpha, beta=args.beta)
    #optimizer = Adam(net.parameters(), betas=(0.9, 0.999), weight_decay=5e-4)
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9)
    # train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    # train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,225], gamma=0.1)
    # iter_per_epoch = len(trainloader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    train_time  = 0.0
    test_time = 0.0
    train_top1_acc = 0.0
    train_min_loss = 100
    test_top1_acc = 0.0
    test_min_loss = 100
    #lr_list = []
    

    writer = SummaryWriter(log_dir=args.save_path)


    for epoch in range(1, args.epoch):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        #lr_list.append(optimizer.param_groups[0]['lr'])
        train_acc_epoch, train_loss_epoch, train_epoch_time = train(epoch)
#        if epoch > args.warm:
#            train_scheduler.step(epoch)
        train_top1_acc = max(train_top1_acc, train_acc_epoch)
        train_min_loss = min(train_min_loss, train_loss_epoch)
        train_time += train_epoch_time
        acc, test_loss_epoch, test_epoch_time = test(epoch)
        test_top1_acc = max(test_top1_acc, acc)
        test_min_loss = min(test_min_loss, test_loss_epoch)
        test_time += test_epoch_time


    writer.close()
    end_train = train_time//60
    end_test = test_time//60
    print(model_name)
    print("train time: {}D {}H {}M".format(end_train//1440, (end_train%1440)//60, end_train%60))
    print("tset time: {}D {}H {}M".format(end_test//1440, (end_test%1440)//60, end_test%60))
    print("train_acc_top1:{}, train_min_loss:{}, train_time:{}, test_top1_acc:{}, test_min_loss:{}, test_time:{}".format(train_top1_acc, train_min_loss, train_time, test_top1_acc, test_min_loss, test_time))
    print("0.3-4.5")

    # rowqc=(args.tao*5)+args.nnlayer+1
    # colqc=args.repnum+3
    
    # from openpyxl import load_workbook
    # wb = load_workbook(r'log/9.25/9.25.xlsx')
    # sheet = wb.active
    # sheet.cell(row=rowqc, column=colqc).value = test_top1_acc.item()
    # coltem=colqc+7
    # sheet.cell(row=rowqc, column=coltem).value = train_top1_acc.item()
    # wb.save(r'log/9.25/9.25.xlsx')
    # print("运行结束！")
    # print("0.3-4.5")
