import os
from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision.models as models
import torch.optim as optim
import torchvision
from sklearn.metrics import accuracy_score
from MoCo_Linear_Model import LinearClassifierResNet
from torch.autograd import Variable


from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cifar_data
import MoCo_model as moco_model
import Linear_Model as lin_model
import numpy as np 
import time
torch.cuda.empty_cache()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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
class ProgressMeter(object):
    
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr']



#### Printing the GPU information
#gpu_information = os.system('nvidia-smi -i 0')
parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

parser.add_argument('-a', '--arch', default='resnet18')

# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

# knn monitor
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

# utils
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

'''
args = parser.parse_args()  # running in command line
'''
args = parser.parse_args('')  # running in ipynb

# set command line arguments here when running in ipynb
args.epochs = 800
args.cos = True
args.schedule = []  # cos in use
args.symmetric = False
if args.results_dir == '':
    args.results_dir = './MoCo_CIFAR_10/Caches/cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")


ssl_model = moco_model.ModelMoCo( dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, arch=args.arch, bn_splits=args.bn_splits, 
symmetric=args.symmetric).cuda()

ssl_optimizer = torch.optim.SGD(ssl_model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)


def train_ssl(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(ssl_optimizer, epoch, args)
    count = 0 
    loss_entire = []
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    
    num = (5000//args.batch_size)
    #A = A1
    for im_1, im_2 in train_bar:
        count = count + 1
        # print(count)
        # print(num)
       

        # if count == num :
        #     print("ENTRO")
        #     A1 = A[count*args.batch_size : len(A)]
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)
        loss   = net(im_1, im_2)
        loss_entire.append(loss)
        #loss_scaled = A1*loss
        loss =  torch.mean(loss)

        train_optimizer.zero_grad()
        loss.backward()
        #print(net.A1.grad)

        train_optimizer.step()
        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num))

    return loss_entire


# test using a knn monitor
def linear_training(model,classifier, train_loader, train_optimizer_2, criterion ,epoch, args):
    
    # We dont update the values of the ssl model 
    loss_entire = []
    test = []
    # We do not want to update the A values here

    model.eval()
    classifier.train()
    adjust_learning_rate(ssl_optimizer, epoch, args)
    count = 0 

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for idx, (x, y) in enumerate(train_loader):
        count = count + 1



        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        data_time.update(time.time() - end)

        with torch.no_grad():
            feature = model(x)
            feature = F.normalize(feature, dim=1)


        output = classifier(feature, y)

        # We create the A1 matrix that is going to multiply the loss

        loss = criterion(output, y)
        loss.backward()


        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0))

    

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % 10 == 0:
            print(
                f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Lr {train_optimizer_2.param_groups[0]["lr"]:.3f} \t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Acc@1 {top1.val:.3%} ({top1.avg:.3%})\t'
                f'Acc@5 {top5.val:.3%} ({top5.avg:.3%})')


    return loss_entire


def validate( model, classifier, val_loader, criterion, args):

    s_model = model.encoder_q
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    # switch to evaluate mode
    s_model.eval()
    classifier.eval()
    end = time.time()
    count = 0
    num = (5000//args.batch_size)

    loss_entire = []


    for idx, (x, y) in enumerate(val_loader):
            count = count + 1
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)


            # compute output
            feature = s_model(x)
            feature = F.normalize(feature, dim=1)

            output = classifier(feature, y)
    
            loss = criterion(output, y)

            acc1, acc5 = accuracy(output, y, topk=(1, 5))

            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10== 0:
                    print(
                        f'Test: [{idx}/{len(val_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Acc@1 {top1.val:.3%} ({top1.avg:.3%})\t'
                        f'Acc@5 {top5.val:.3%} ({top5.avg:.3%})')

    return losses




    
# Prepare the data transformation
def main():
    

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    # Loading the data
    # The training data will be used as the self surpervised data

    train_data = cifar_data.CIFAR10Pair(root='data', train=True, transform=train_transform, download=True) 
    train_data_linear = torchvision.datasets.CIFAR10(root='data', train=True, transform=train_transform, download=True) 
    memory_data = CIFAR10(root='data', train=True, transform=test_transform, download=True)
    val_data = CIFAR10(root='data', train=False, transform=test_transform, download=True)

    
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    train_linear_loader = DataLoader(train_data_linear, batch_size=512, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=512, shuffle=False, num_workers=16, pin_memory=True ,drop_last=True)

    training_size =  train_loader.dataset.data.shape


    ssl_optimizer = torch.optim.SGD(ssl_model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    ## creating the self supervised model



    # Parameters for the linear classifier 


    linear_classifier = lin_model.LinNet().cuda()
    linear_criterion = nn.CrossEntropyLoss()


    linear_optimizer = optim.SGD(linear_classifier.parameters(), lr=0.001, momentum=0.9)


    # define optimizer

    # load model if resume
    epoch_start = 1
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))

    # logging
    results = {'ssl_train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    # training loop

    for epoch in range(epoch_start, args.epochs + 1):
        ssl_train_loss = train_ssl(ssl_model, train_loader, ssl_optimizer, epoch, args)

        lin_training_loss = linear_training(ssl_model.encoder_q, linear_classifier, train_linear_loader, linear_optimizer, linear_criterion,epoch, args)

        val_loss = validate( ssl_model, linear_classifier, val_loader, linear_criterion,args)




        # results['test_acc@1'].append(test_acc_1)
        # # save statistics
        # data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        # data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
        # # save model
        # torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')


print("Training Finished")

if __name__=="__main__": 
    main() 