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
import datasets
import models as models
import matplotlib.pyplot as plt
import torchvision.models as torch_models
import scipy.io as sio
from PIL import Image
import cv2

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch gull')
parser.add_argument('-d', '--dataset', default='gull', help='dataset name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='keras',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-c', '--channel', type=int, default=16,
                    help='first conv channel (default: 16)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpu', default='4', help='index of gpus to use')
parser.add_argument('--iters', default=20, type=int, metavar='N',
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step', default='2', help='decreasing strategy')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num_weaker', default=20, type=int, metavar='N',
                    help='number of first stage epochs to run')
parser.add_argument('--learning_rate', default=1, type=int, metavar='N',
                    help='number of first stage epochs to run')
parser.add_argument('--topK', default=10, type=int, metavar='N',
                    help='number of first stage epochs to run')
parser.add_argument('--S_max', default=1, type=int, metavar='N',
                    help='see professor writing')
parser.add_argument('--Lambda', default=10.0, type=int, metavar='N',
                    help='see professor writing')
parser.add_argument('--selection', default='MMT_largemargin', help='see professor writing')


def main():
    global args, best_prec1
    args = parser.parse_args()

    # training multiple times

    # select gpus
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # data loader
    assert callable(datasets.__dict__[args.dataset])
    get_dataset = getattr(datasets, args.dataset)
    num_classes = datasets._NUM_CLASSES[args.dataset]
    train_loader, val_loader = get_dataset(
        batch_size=args.batch_size, num_workers=args.workers)


    model_main = models.__dict__['resnet18_feature'](pretrained=True)
    if args.selection == 'MMT_largemargin' or args.selection == 'random':
        model_main.fc = nn.Linear(512 * 1, num_classes, bias=False)
    else:
        model_main.fc = nn.Linear(512 * 1, num_classes)
    model_main = torch.nn.DataParallel(model_main, device_ids=range(len(args.gpu))).cuda()

    criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    criterion2 = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    optimizer_m = torch.optim.SGD(model_main.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
    # optimizer_m = torch.optim.Adam(model_main.parameters(), lr=0.001, weight_decay=1e-4)

    teaching_set = './datasets/CUBgull/' + args.selection + '/CUBgull_Lt_gt_CE_tr.txt'
    remaining_set = './datasets/CUBgull/' + args.selection + '/CUBgull_Dt_gt_tr.txt'
    remaining_set_CE = './datasets/CUBgull/' + args.selection + '/CUBgull_Dt_gt_CE_tr.txt'
    teaching_example_index = 0
    all_test_acc_iter = np.zeros(args.iters)
    all_train_acc_iter = np.zeros(args.iters)

    all_possible_used_real_human_explanation = []
    all_possible_used_real_human_example = []
    for iter in range(args.iters):

        if iter == 0:
            imlist = []
            labellist = []
            with open('./datasets/CUBgull/CUBgull_gt_tr.txt', 'r') as rf:
                for line in rf.readlines():
                    impath, imlabel, imindex = line.strip().split()
                    imlist.append(impath)
                    labellist.append(imlabel)

            imlist_CE = []
            labellist_CE = []
            counterlist_CE = []
            indexlist = []
            with open('./datasets/CUBgull/CUBgull_gt_CE.txt', 'r') as rf:
                for line in rf.readlines():
                    impath, imlabel, imcounter, imindex = line.strip().split()
                    imlist_CE.append(impath)
                    labellist_CE.append(imlabel)
                    counterlist_CE.append(imcounter)
                    indexlist.append(imindex)

            assert callable(datasets.__dict__['gull_wo_shuffle'])
            get_dataset = getattr(datasets, 'gull_wo_shuffle')
            num_classes = datasets._NUM_CLASSES['gull_wo_shuffle']
            train_loader, val_loader = get_dataset(
                batch_size=1, num_workers=args.workers)

            assert callable(datasets.__dict__['gull_CE'])
            get_dataset = getattr(datasets, 'gull_CE')
            num_classes = datasets._NUM_CLASSES['gull_CE']
            train_loader_CE_Dt, val_loader_CE = get_dataset(
                batch_size=1, num_workers=args.workers)

            if args.selection == 'MMT' or args.selection == 'MMT_largemargin':
                added_example_indices, groundtruth, counter, other_counter_index, other_counter = selection(train_loader, train_loader_CE_Dt, model_main, optimizer_m, iter, criterion,
                                                  args.S_max, args.Lambda)
            elif args.selection == 'random':
                added_example_indices = selection_random(train_loader, model_main, optimizer_m, iter, criterion,
                                                         args.S_max, args.Lambda)

            # added_example_indices = added_example_indices.tolist()
            fl = open(teaching_set, 'w')
            for k in range(len(added_example_indices)):
                example_info = imlist[added_example_indices[k]] + " " + labellist[added_example_indices[k]] + " " + str(
                    teaching_example_index)
                fl.write(example_info)
                fl.write("\n")
                teaching_example_index = teaching_example_index + 1

            example_info = imlist[added_example_indices[0]] + " " + labellist[added_example_indices[0]] + " " + str(iter)
            all_possible_used_real_human_example.append(example_info)


            removed_index = []
            for i_c in range(len(imlist_CE)):
                if labellist_CE[i_c] == str(groundtruth) and counterlist_CE[i_c] == str(counter) and indexlist[i_c] == str(added_example_indices[0]):
                    example_info = imlist_CE[i_c] + " " + labellist_CE[i_c] + " " + str(teaching_example_index)
                    fl.write(example_info)
                    fl.write("\n")
                    teaching_example_index = teaching_example_index + 1
                if counterlist_CE[i_c] == str(groundtruth) and labellist_CE[i_c] == str(counter) and indexlist[i_c] == str(added_example_indices[1]):
                    example_info = imlist_CE[i_c] + " " + labellist_CE[i_c] + " " + str(teaching_example_index)
                    fl.write(example_info)
                    fl.write("\n")
                    teaching_example_index = teaching_example_index + 1
                if indexlist[i_c] == str(added_example_indices[0]) or indexlist[i_c] == str(added_example_indices[1]):
                    removed_index.append(i_c)

            fl.close()

            for i_r in range(4):
                cur_counter = other_counter[i_r]
                cur_counter_index = other_counter_index[i_r]
                for i_c in range(len(imlist_CE)):
                    if labellist_CE[i_c] == str(groundtruth) and counterlist_CE[i_c] == str(cur_counter) and indexlist[
                        i_c] == str(cur_counter_index):
                        example_info = imlist_CE[i_c] + " " + labellist_CE[i_c] + " " + str(cur_counter) + " " + str(iter)
                        all_possible_used_real_human_explanation.append(example_info)
                    if counterlist_CE[i_c] == str(groundtruth) and labellist_CE[i_c] == str(cur_counter) and indexlist[
                        i_c] == str(cur_counter_index):
                        example_info = imlist_CE[i_c] + " " + labellist_CE[i_c] + " " + str(groundtruth) + " " + str(iter)
                        all_possible_used_real_human_explanation.append(example_info)


            # update Dt
            imlist = [i for j, i in enumerate(imlist) if j not in added_example_indices]
            labellist = [i for j, i in enumerate(labellist) if j not in added_example_indices]
            fl = open(remaining_set, 'w')
            num = 0
            for k in range(len(imlist)):
                example_info = imlist[k] + " " + labellist[k] + " " + str(num)
                fl.write(example_info)
                fl.write("\n")
                num = num + 1
            fl.close()

            # update CE_Dt
            imlist_CE = [i for j, i in enumerate(imlist_CE) if j not in removed_index]
            labellist_CE = [i for j, i in enumerate(labellist_CE) if j not in removed_index]
            counterlist_CE = [i for j, i in enumerate(counterlist_CE) if j not in removed_index]
            indexlist = [i for j, i in enumerate(indexlist) if j not in removed_index]
            fl = open(remaining_set_CE, 'w')
            num = 0
            for k in range(len(imlist_CE)):
                example_info = imlist_CE[k] + " " + labellist_CE[k] + " " + counterlist_CE[k] + " " + str(num//4)
                fl.write(example_info)
                fl.write("\n")
                num = num + 1
            fl.close()

        else:

            assert callable(datasets.__dict__['gull_Dt'])
            get_dataset = getattr(datasets, 'gull_Dt')
            num_classes = datasets._NUM_CLASSES['gull_Dt']
            train_loader_Dt, val_loader = get_dataset(
                batch_size=1, num_workers=args.workers, selection=args.selection)

            assert callable(datasets.__dict__['gull_CE_Dt'])
            get_dataset = getattr(datasets, 'gull_CE_Dt')
            num_classes = datasets._NUM_CLASSES['gull_CE_Dt']
            train_loader_CE_Dt, val_loader_CE = get_dataset(
                batch_size=1, num_workers=args.workers, selection=args.selection)


            if args.selection == 'MMT' or args.selection == 'MMT_largemargin':
                added_example_indices, groundtruth, counter, other_counter_index, other_counter = selection(train_loader_Dt, train_loader_CE_Dt, model_main, optimizer_m, iter, criterion,
                                                  args.S_max, args.Lambda)
            elif args.selection == 'random':
                added_example_indices = selection_random(train_loader_Dt, model_main, optimizer_m, iter, criterion,
                                                         args.S_max, args.Lambda)

            imlist = []
            labellist = []
            with open(remaining_set, 'r') as rf:
                for line in rf.readlines():
                    impath, imlabel, imindex = line.strip().split()
                    imlist.append(impath)
                    labellist.append(imlabel)

            imlist_CE = []
            labellist_CE = []
            counterlist_CE = []
            indexlist = []
            with open(remaining_set_CE, 'r') as rf:
                for line in rf.readlines():
                    impath, imlabel, imcounter, imindex = line.strip().split()
                    imlist_CE.append(impath)
                    labellist_CE.append(imlabel)
                    counterlist_CE.append(imcounter)
                    indexlist.append(imindex)

            fl = open(teaching_set, 'a')
            for k in range(len(added_example_indices)):
                example_info = imlist[added_example_indices[k]] + " " + labellist[added_example_indices[k]] + " " + str(
                    teaching_example_index)
                fl.write(example_info)
                fl.write("\n")
                teaching_example_index = teaching_example_index + 1

            example_info = imlist[added_example_indices[0]] + " " + labellist[added_example_indices[0]] + " " + str(iter)
            all_possible_used_real_human_example.append(example_info)

            removed_index = []
            for i_c in range(len(imlist_CE)):
                if labellist_CE[i_c] == str(groundtruth) and counterlist_CE[i_c] == str(counter) and indexlist[i_c] == str(added_example_indices[0]):
                    example_info = imlist_CE[i_c] + " " + labellist_CE[i_c] + " " + str(teaching_example_index)
                    fl.write(example_info)
                    fl.write("\n")
                    teaching_example_index = teaching_example_index + 1
                if counterlist_CE[i_c] == str(groundtruth) and labellist_CE[i_c] == str(counter) and indexlist[i_c] == str(added_example_indices[1]):
                    example_info = imlist_CE[i_c] + " " + labellist_CE[i_c] + " " + str(teaching_example_index)
                    fl.write(example_info)
                    fl.write("\n")
                    teaching_example_index = teaching_example_index + 1
                if indexlist[i_c] == str(added_example_indices[0]) or indexlist[i_c] == str(added_example_indices[1]):
                    removed_index.append(i_c)
            fl.close()

            for i_r in range(4):
                cur_counter = other_counter[i_r]
                cur_counter_index = other_counter_index[i_r]
                for i_c in range(len(imlist_CE)):
                    if labellist_CE[i_c] == str(groundtruth) and counterlist_CE[i_c] == str(cur_counter) and indexlist[
                        i_c] == str(cur_counter_index):
                        example_info = imlist_CE[i_c] + " " + labellist_CE[i_c] + " " + str(cur_counter) + " " + str(iter)
                        all_possible_used_real_human_explanation.append(example_info)
                    if counterlist_CE[i_c] == str(groundtruth) and labellist_CE[i_c] == str(cur_counter) and indexlist[
                        i_c] == str(cur_counter_index):
                        example_info = imlist_CE[i_c] + " " + labellist_CE[i_c] + " " + str(groundtruth) + " " + str(iter)
                        all_possible_used_real_human_explanation.append(example_info)

            assert callable(datasets.__dict__['gull_CE_Lt'])
            get_dataset = getattr(datasets, 'gull_CE_Lt')
            num_classes = datasets._NUM_CLASSES['gull_CE_Lt']
            train_loader_Lt, val_loader = get_dataset(
                batch_size=teaching_example_index, num_workers=args.workers, selection=args.selection)

            for epoch in range(10):
                prec1_tr = train_largemargin(train_loader_Lt, model_main, optimizer_m, iter, criterion2)

            print('training acc', prec1_tr.item())
            all_train_acc_iter[iter] = prec1_tr
            prec1 = validate(val_loader, model_main)
            print('testing acc', prec1.item())
            all_test_acc_iter[iter] = prec1

            # update Dt
            imlist = [i for j, i in enumerate(imlist) if j not in added_example_indices]
            labellist = [i for j, i in enumerate(labellist) if j not in added_example_indices]
            fl = open(remaining_set, 'w')
            num = 0
            for k in range(len(imlist)):
                example_info = imlist[k] + " " + labellist[k] + " " + str(num)
                fl.write(example_info)
                fl.write("\n")
                num = num + 1
            fl.close()

            # update CE_Dt
            imlist_CE = [i for j, i in enumerate(imlist_CE) if j not in removed_index]
            labellist_CE = [i for j, i in enumerate(labellist_CE) if j not in removed_index]
            counterlist_CE = [i for j, i in enumerate(counterlist_CE) if j not in removed_index]
            indexlist = [i for j, i in enumerate(indexlist) if j not in removed_index]
            fl = open(remaining_set_CE, 'w')
            num = 0
            for k in range(len(imlist_CE)):
                example_info = imlist_CE[k] + " " + labellist_CE[k] + " " + counterlist_CE[k] + " " + str(num//4)
                fl.write(example_info)
                fl.write("\n")
                num = num + 1
            fl.close()

    np.save('./gull/all_train_acc_iter_' + args.selection + '_imagenet_CE.npy', all_train_acc_iter)
    np.save('./gull/all_test_acc_iter_' + args.selection + '_imagenet_CE.npy', all_test_acc_iter)


def selection(train_loader, train_loader_CE, model_main, optimizer_m, epoch, criterion, S_max, Lambda):
    # switch to train mode
    model_main.train()
    all_weights = []
    predicted_classes = []
    all_targets = []
    for i, (input, target, index) in enumerate(train_loader):
        input = input.cuda()
        all_targets.append(target.item())
        # target = target.cuda(async=True)

        # compute output
        predicted_labels, embeddings = model_main(input)  # embeddings is batch * 64
        predicted_labels = torch.argmax(predicted_labels)
        predicted_classes.append(predicted_labels.item())
        # gradient is -e^-v
        codewords = model_main.module.fc.weight  # codewords is batch * 5 * 64

        embeddings = embeddings.detach().cpu().numpy()
        codewords = codewords.detach().cpu().numpy()

        y_c = codewords[target, :]
        y_c = np.tile(y_c, (5, 1))
        embeddings = np.tile(embeddings, (5, 1))
        embeddings = np.delete(embeddings, target, 0)
        y_c = np.delete(y_c, target, 0)
        codewords = np.delete(codewords, target, 0)
        w_i_original = -np.exp(-0.5 * np.sum(embeddings * (y_c - codewords), axis=1))
        #        w_i_original = np.sum(w_i_original, axis = 1)
        w_i = np.sum(w_i_original)
        epsilon = w_i_original / w_i
        w_i_square = w_i * w_i
        epsilon = np.reshape(epsilon, (4, 1))
        psi = w_i_square * np.sum(
            (y_c[0, :] - np.sum(codewords * epsilon, axis=0)) * (y_c[0, :] - np.sum(codewords * epsilon, axis=0)))
        all_weights.append(psi)

    all_weights = np.array(all_weights)
    all_weights = np.repeat(all_weights, 4)

    predicted_classes = np.array(predicted_classes)
    predicted_classes = np.repeat(predicted_classes, 4)
    all_targets = np.array(all_targets)
    all_targets = np.repeat(all_targets, 4)

    all_weights_CE = []
    all_index_CE = []
    for i, (input, target, counter, index) in enumerate(train_loader_CE):
        input = input.cuda()
        all_index_CE.append(index.item())
        # target = target.cuda(async=True)

        # compute output
        predicted_labels, embeddings = model_main(input)  # embeddings is batch * 64
        # gradient is -e^-v
        codewords = model_main.module.fc.weight  # codewords is batch * 5 * 64

        embeddings = embeddings.detach().cpu().numpy()
        codewords = codewords.detach().cpu().numpy()

        y_c = codewords[target, :]
        y_c = np.tile(y_c, (5, 1))
        embeddings = np.tile(embeddings, (5, 1))
        embeddings = np.delete(embeddings, target, 0)
        y_c = np.delete(y_c, target, 0)
        codewords = np.delete(codewords, target, 0)
        w_i_original = -np.exp(-0.5 * np.sum(embeddings * (y_c - codewords), axis=1))
        #        w_i_original = np.sum(w_i_original, axis = 1)
        w_i = np.sum(w_i_original)
        epsilon = w_i_original / w_i
        w_i_square = w_i * w_i
        epsilon = np.reshape(epsilon, (4, 1))
        psi = w_i_square * np.sum(
            (y_c[0, :] - np.sum(codewords * epsilon, axis=0)) * (y_c[0, :] - np.sum(codewords * epsilon, axis=0)))
        all_weights_CE.append(psi)

    sum_weights = all_weights + all_weights_CE


    # select
    sum_weights = sum_weights.tolist()
    predicted_classes = predicted_classes.tolist()
    all_targets = all_targets.tolist()
    # all_index_CE = all_index_CE.tolist()
    _, sorted_predicted_classes = zip(*sorted(zip(sum_weights, predicted_classes), reverse=True))
    _, sorted_all_targets = zip(*sorted(zip(sum_weights, all_targets), reverse=True))
    _, sorted_all_index = zip(*sorted(zip(sum_weights, all_index_CE), reverse=True))

    added_example_indices = sorted_all_index[0]
    added_example_indices = int(added_example_indices)

    counterfactual_class = sorted_predicted_classes[0]
    added_counter_example_index = sorted_all_targets.index(counterfactual_class)
    added_counter_example_index = sorted_all_index[added_counter_example_index]

    candidate_counter_list = [0, 1, 2, 3, 4]
    candidate_counter_list.remove(sorted_all_targets[0])
    other_potential_counter = []
    other_potential_counter_index = []
    for i_c in candidate_counter_list:
        cur_added_counter_example_index = sorted_all_targets.index(i_c)
        cur_added_counter_example_index = sorted_all_index[cur_added_counter_example_index]
        other_potential_counter.append(i_c)
        other_potential_counter_index.append(cur_added_counter_example_index)


    return [added_example_indices, added_counter_example_index], sorted_all_targets[0], counterfactual_class, other_potential_counter_index, other_potential_counter


def selection_random(train_loader, model_main, optimizer_m, epoch, criterion, S_max, Lambda):
    # switch to train mode
    model_main.eval()
    all_weights = []
    for i, (input, target, index) in enumerate(train_loader):
        input = input.cuda()
        # target = target.cuda(async=True)

        # compute output
        _, embeddings = model_main(input)  # embeddings is batch * 64

        # gradient is -e^-v
        codewords = model_main.module.fc.weight  # codewords is batch * 5 * 64

        embeddings = embeddings.detach().cpu().numpy()
        codewords = codewords.detach().cpu().numpy()

        y_c = codewords[target, :]
        y_c = np.tile(y_c, (5, 1))
        embeddings = np.tile(embeddings, (5, 1))
        embeddings = np.delete(embeddings, target, 0)
        y_c = np.delete(y_c, target, 0)
        codewords = np.delete(codewords, target, 0)
        weights = np.exp(-0.5 * np.sum(embeddings * (y_c - codewords), axis=1))
        weight = np.sum(weights)
        all_weights.append(weight)

    # select
    all_weights = np.array(all_weights)
    all_weights = torch.from_numpy(all_weights)

    sorted_weight, indices_weight = torch.sort(all_weights, descending=True)

    cum_weights = torch.cumsum(sorted_weight, dim=0)
    cost = Lambda * torch.exp(torch.range(1, S_max))
    L = cum_weights[:S_max] - cost
    _, S_star = L.max(0)

    indices_grad = torch.randperm(sorted_weight.shape[0])
    added_example_indices = indices_grad[:S_star + 1]
    added_example_indices = added_example_indices.cpu().numpy()

    return added_example_indices


def train(train_loader, model_main, optimizer_m, epoch, criterion):
    losses_m = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model_main.train()

    for i, (input, target, index) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        predicted_labels, _ = model_main(input)

        loss_m = criterion(predicted_labels, target)
        prec1, prec5 = accuracy(predicted_labels, target, topk=(1, 5))
        losses_m.update(loss_m.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer_m.zero_grad()
        loss_m.backward()
        optimizer_m.step()

    return top1.avg


def train_largemargin(train_loader, model_main, optimizer_m, epoch, criterion):
    losses_m = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model_main.train()

    for i, (input, target, index) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        predicted_labels, embeddings = model_main(input)

        z_c = predicted_labels[torch.arange(input.size(0)), target]
        z_c = torch.reshape(z_c, (input.size(0), 1))
        z_c = z_c.repeat(1, 5)
        loss = torch.exp(predicted_labels - z_c)
        loss_m = torch.mean(loss) - 1.0

        # loss_m = criterion(predicted_labels, target)

        prec1, prec5 = accuracy(predicted_labels, target, topk=(1, 5))
        losses_m.update(loss_m.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer_m.zero_grad()
        loss_m.backward()
        optimizer_m.step()

    return top1.avg


def validate(val_loader, model_main):
    top1 = AverageMeter()
    # switch to evaluate mode
    model_main.eval()

    for i, (input, target, index) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        output, _ = model_main(input)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))

    return top1.avg



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


if __name__ == '__main__':
    main()
