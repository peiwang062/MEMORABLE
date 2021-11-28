import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import datasets_augmentation
import models as models
import matplotlib.pyplot as plt
import torchvision.models as torch_models
import scipy.io as sio
from sklearn.metrics import confusion_matrix


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch butterflies')
parser.add_argument('-d', '--dataset', default='butterflies_s300', help='dataset name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='keras',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-c', '--channel', type=int, default=16,
                    help='first conv channel (default: 16)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpu', default='0,1,2,3', help='index of gpus to use')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step', default='80,120,160', help='decreasing strategy')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--first_epochs', default=80, type=int, metavar='N',
                    help='number of first stage epochs to run')


def main():
    global args, best_prec1
    args = parser.parse_args()

    # training multiple times

    # select gpus
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # data loader
    assert callable(datasets_augmentation.__dict__[args.dataset])
    get_dataset = getattr(datasets_augmentation, args.dataset)
    num_classes = datasets_augmentation._NUM_CLASSES[args.dataset]
    train_loader, val_loader = get_dataset(
        batch_size=args.batch_size, num_workers=args.workers)


    # create model
    model_main = models.__dict__['resnet18'](pretrained=False)
    model_main.fc = nn.Linear(512 * 1, num_classes)
    model_main = torch.nn.DataParallel(model_main, device_ids=range(len(args.gpu))).cuda()


    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    optimizer_m1 = torch.optim.SGD(model_main.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # train nn in order to get the feature vector for each sample
    # lr_step = np.arange(args.start_epoch + 1, args.epochs).tolist()

    lr_step = list(map(int, args.lr_step.split(',')))
    prec1, prec5, all_correct_te, all_targets, all_predicted = validate(val_loader, model_main, criterion)
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in lr_step:
            for param_group in optimizer_m1.param_groups:
                param_group['lr'] *= 0.1

        # train for one epoch
        prec1_tr = train(train_loader, model_main, optimizer_m1, epoch, criterion)
        prec1, prec5, all_correct_te, all_targets, all_predicted = validate(val_loader, model_main, criterion)
    prec1, prec5, all_correct_te, all_targets, all_predicted = validate(val_loader, model_main, criterion)


    # # compute confusion matrix
    # df_confusion = confusion_matrix(list(all_targets), list(all_predicted))
    # df_conf_norm = df_confusion / df_confusion.sum(axis=1)[:, np.newaxis]
    # print(df_conf_norm)
    #
    # plt.figure()
    # save_name = './butterflies/confusion_matrix_s300.jpg'
    # plot_confusion_matrix(df_conf_norm, ['CabbageWhite', 'Monarch', 'Queen', 'RedAdmiral', 'Viceroy'], save_name)
    #
    # # y_actu = pd.Series(list(all_targets), name='Actual')
    # # y_pred = pd.Series(list(all_predicted), name='Predicted')
    # # df_confusion = pd.crosstab(y_actu, y_pred)
    # # df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    # # print(df_conf_norm)
    #
    # save_checkpoint({
    #     'epoch': epoch + 1,
    #     'arch': args.arch,
    #     'state_dict_m': model_main.cpu().module.state_dict(),
    #     'optimizer_m1': optimizer_m1.state_dict(),
    # }, filename='./butterflies/checkpoint_pretrain_resnet18_s300.pth.tar')



def train(train_loader, model_main, optimizer_m, epoch, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_m = AverageMeter()
    losses_a = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model_main.train()

    end = time.time()
    for i, (input, target, index) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # input and target
        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        predicted_labels = model_main(input)

        loss_m = criterion(predicted_labels, target)
        prec1, prec5 = accuracy(predicted_labels, target, topk=(1, 5))
        losses_m.update(loss_m.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer_m.zero_grad()
        loss_m.backward()
        optimizer_m.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            curr_lr_m = optimizer_m.param_groups[0]['lr']
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: [{4}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_m {loss_m.val:.4f} ({loss_m.avg:.4f})\t'
                  'Loss_a {loss_a.val:.4f} ({loss_a.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, args.epochs, i, len(train_loader), curr_lr_m,
                batch_time=batch_time, data_time=data_time, loss_m=losses_m, loss_a=losses_a, top1=top1, top5=top5))
    return top1.avg



def validate(val_loader, model_main, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_main.eval()
    end = time.time()

    all_correct_te = []
    all_targets = []
    all_predicted = []
    for i, (input, target, index) in enumerate(val_loader):
        all_targets = np.concatenate((all_targets, target), axis=0)
        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        output = model_main(input)

        predicted = torch.argmax(output, dim=1)
        all_predicted = np.concatenate((all_predicted, predicted.cpu().numpy()), axis=0)

        loss = criterion(output, target)

        p_i_m = torch.max(output, dim=1)[1]
        p_i_m = p_i_m.long()
        p_i_m[p_i_m - target == 0] = -1
        p_i_m[p_i_m > -1] = 0
        p_i_m[p_i_m == -1] = 1
        correct = p_i_m.float()
        all_correct_te = np.concatenate((all_correct_te, correct.cpu()), axis=0)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    # print(' * Testing Prec@1 {top1.avg:.3f}'.format(top1=top1))
    print(top1.avg)
    return top1.avg, top5.avg, all_correct_te, all_targets, all_predicted


def save_checkpoint(state, filename='checkpoint_res.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def plot_confusion_matrix(cm, classes, name='conf_matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(name)

if __name__ == '__main__':
    main()
