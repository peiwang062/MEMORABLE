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
import datasets, datasets_augmentation
import models as models
import matplotlib.pyplot as plt
import torchvision.models as torch_models
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import utils
import scipy.io as sio
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import cv2
import seaborn as sns
import operator
import torch.nn.functional as F

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch end2end butterflies Training')
parser.add_argument('-d', '--dataset', default='butterflies_whole_s300', help='dataset name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet20)')
parser.add_argument('-c', '--channel', type=int, default=16,
                    help='first conv channel (default: 16)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpu', default='6', help='index of gpus to use')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step', default='5', help='decreasing strategy')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='butterflies_checkpoint_100.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--first_epochs', default=5, type=int, metavar='N',
                    help='number of first stage epochs to run')


best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # select gpus
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # data loader
    get_dataset = getattr(datasets_augmentation, "butterflies_simclr")
    train_loader, test_loader = get_dataset(
        batch_size=193, num_workers=8)

    # create model
    encoder = get_resnet('resnet18', pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, 64, n_features)
    model_fp = os.path.join('butterflies_checkpoint_100.tar')
    simclr_model.load_state_dict(torch.load(model_fp))
    simclr_model = simclr_model.to(torch.device("cuda"))
    simclr_model.eval()

    ## Logistic Regression
    n_classes = 5
    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(torch.device("cuda"))

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        simclr_model, train_loader, test_loader, torch.device("cuda")
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, 256
    )

    for epoch in range(500):
        loss_epoch, accuracy_epoch = train(
            args, arr_train_loader, simclr_model, model, criterion, optimizer
        )

        print('Epoch: [{0}/{1}]\t'
              'Loss: [{2}]\t'
              'Accuracy: [{3}]'.format(
            epoch, 500, loss_epoch/len(train_loader), accuracy_epoch/len(train_loader)))





    # load all image list
    imlist = []
    imclass = []
    with open('./datasets/butterflies_crop/butterflies_gt_s300_tr.txt', 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, imindex = line.strip().split()
            imlist.append(impath)
            imclass.append(int(imlabel))

    # make gradcam
    grad_cam_cls = GradCam_cls(simclr_model, model, target_layer_names=["42"], use_cuda=True)
    grad_cam_hp = GradCam_hp(simclr_model, model, target_layer_names=["42"], use_cuda=True)


    # data loader
    assert callable(datasets.__dict__['butterflies_whole_s300'])
    get_dataset = getattr(datasets, 'butterflies_whole_s300')
    num_classes = datasets._NUM_CLASSES['butterflies_whole_s300']
    train_whole_loader, _ = get_dataset(
        batch_size=5, num_workers=args.workers)

    remaining_mask_size_pool = np.arange(0.01, 0.31, 0.01)

    # mimic predicted classes is 0->1, 1->2, 2->3, 3->4, 4->0
    for i_loop in range(4):

        target_class = np.array(imclass)
        predicted_class = (target_class + i_loop + 1) % 5
        cross_match = np.zeros((len(imlist), 2))
        cross_match[:, 0] = predicted_class
        cross_match[:, 1] = target_class
        cf_proposal_extraction(train_whole_loader, grad_cam_hp, grad_cam_cls, predicted_class, imlist, remaining_mask_size_pool, cross_match)
        cf_proposal_extraction2(train_whole_loader, grad_cam_hp, grad_cam_cls, predicted_class, imlist, remaining_mask_size_pool, cross_match)



def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print("Computing features...")
            print('Epoch: [{0}/{1}]'.format(
                step, len(loader)))

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(torch.device("cuda"))
        y = y.to(torch.device("cuda"))

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch




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





class FeatureExtractor_hp():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)  # forward one layer each time
            if name in 'layer4':  # store the gradient of target layer
                x.register_hook(self.save_gradient)
                outputs += [x]  # after last feature map, nn.MaxPool2d(kernel_size=2, stride=2)] follows
        return outputs, x




class FeatureExtractor_cls():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)  # forward one layer each time
            if name in 'layer4':  # store the gradient of target layer
                x.register_hook(self.save_gradient)
                outputs += [x]  # after last feature map, nn.MaxPool2d(kernel_size=2, stride=2)] follows
        return outputs, x

class ModelOutputs_hp():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, simclr_model, model, target_layers):
        self.simclr_model = simclr_model.encoder
        self.model = model
        self.feature_extractor = FeatureExtractor_hp(self.simclr_model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model(output)
        confidence_score = F.softmax(output, dim=1)
        confidence_score = torch.max(confidence_score, dim=1)[0]
        return target_activations, confidence_score


class ModelOutputs_cls():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, simclr_model, model, target_layers):
        self.simclr_model = simclr_model.encoder
        self.model = model
        self.feature_extractor = FeatureExtractor_cls(self.simclr_model, target_layers)


    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model(output)
        # output = output.view(output.size(0), -1)
        # output = self.model._modules['module'].classifier(output)  # travel many fc layers
        return target_activations, output




def show_segment_on_image(img, mask):

    # draw contours for masked regions

    img = np.float32(img)
    img_dark = np.copy(img)
    mask255 = np.copy(mask)
    mask = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
    img = np.uint8(255 * img)


    mask255 = np.uint8(255 * mask255)
    contours, hierarchy = cv2.findContours(mask255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    # img_dark = img_dark * 0.4
    # img_dark = np.uint8(255 * img_dark)
    # img_dark[mask > 0] = img[mask > 0]
    # img = img_dark

    return img



def compute_average_outof_mask(cur_img, mask):
    mask = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
    img = cur_img * mask
    img[img == 0] = np.nan
    means = np.nanmean(img, axis=(0,1))
    return means


class GradCam_hp:
    def __init__(self, simclr_model, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
            self.simclr_model = simclr_model.cuda()

        self.extractor = ModelOutputs_hp(self.simclr_model, self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):

        features, output = self.extractor(input)

        grads_val = torch.autograd.grad(output, features[0], grad_outputs=torch.ones_like(output),
                                        create_graph=True)
        grads_val = grads_val[0].squeeze()
        grads_val = grads_val.cpu().data.numpy()

        mask_positive = np.copy(grads_val)
        # mask_positive[mask_positive < 0.0] = 0.0
        mask_positive = mask_positive.squeeze()

        target = features[-1]
        target = target.cpu().data.numpy()

        weights = np.mean(mask_positive, axis=(2, 3))
        weights = weights[:, :, np.newaxis, np.newaxis]
        weights = np.repeat(weights, 7, axis=2)
        weights = np.repeat(weights, 7, axis=3)

        cam = weights * target
        cam = np.sum(cam, axis=1)

        cam = np.maximum(cam, 0)
        channel_min = np.min(cam, (1, 2))
        channel_min = channel_min[:, np.newaxis, np.newaxis]
        cam = cam - channel_min
        channel_max = np.max(cam, (1, 2))
        channel_max = channel_max[:, np.newaxis, np.newaxis]
        cam = cam / channel_max

        return cam


class GradCam_cls:
    def __init__(self, simclr_model, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
            self.simclr_model = simclr_model.cuda()

        self.extractor = ModelOutputs_cls(self.simclr_model, self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, PredictedClass, CounterfactualClass):

        features, output = self.extractor(input)

        target = features[-1]
        target = target.cpu().data.numpy()

        classifier_heatmaps = np.zeros((input.shape[0], np.size(target, 2), np.size(target, 2), 2))
        one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(output.shape[0]), PredictedClass] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output, dim=1)
        grads_val = torch.autograd.grad(one_hot, features, grad_outputs=torch.ones_like(one_hot),
                                        create_graph=True)
        grads_val = grads_val[0].squeeze()
        grads_val = grads_val.cpu().data.numpy().squeeze()

        mask_positive = np.copy(grads_val)
        mask_positive = mask_positive.squeeze()

        weights = np.mean(mask_positive, axis=(2, 3))
        weights = weights[:, :, np.newaxis, np.newaxis]
        weights = np.repeat(weights, 7, axis=2)
        weights = np.repeat(weights, 7, axis=3)

        cam = weights * target
        cam = np.sum(cam, axis=1)

        cam = np.maximum(cam, 0)
        channel_min = np.min(cam, (1, 2))
        channel_min = channel_min[:, np.newaxis, np.newaxis]
        cam = cam - channel_min
        channel_max = np.max(cam, (1, 2))
        channel_max = channel_max[:, np.newaxis, np.newaxis]
        cam = cam / channel_max

        classifier_heatmaps[:, :, :, 0] = cam

        one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(output.shape[0]), CounterfactualClass] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output, dim=1)
        grads_val = torch.autograd.grad(one_hot, features, grad_outputs=torch.ones_like(one_hot),
                                        create_graph=True)
        grads_val = grads_val[0].squeeze()
        grads_val = grads_val.cpu().data.numpy().squeeze()

        mask_positive = np.copy(grads_val)
        mask_positive = mask_positive.squeeze()

        weights = np.mean(mask_positive, axis=(2, 3))
        weights = weights[:, :, np.newaxis, np.newaxis]
        weights = np.repeat(weights, 7, axis=2)
        weights = np.repeat(weights, 7, axis=3)

        cam = weights * target
        cam = np.sum(cam, axis=1)

        cam = np.maximum(cam, 0)
        channel_min = np.min(cam, (1, 2))
        channel_min = channel_min[:, np.newaxis, np.newaxis]
        cam = cam - channel_min
        channel_max = np.max(cam, (1, 2))
        channel_max = channel_max[:, np.newaxis, np.newaxis]
        cam = cam / channel_max

        classifier_heatmaps[:, :, :, 1] = cam

        return classifier_heatmaps



def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)



def cf_proposal_extraction(val_loader, grad_cam_hp, grad_cam_cls, predicted_class, imlist, remaining_mask_size_pool, cross_match):


    all_images = np.zeros((len(imlist), 224, 224, 3))
    all_heatmap_mask = np.zeros((len(imlist), np.size(remaining_mask_size_pool), 224, 224))

    all_X_Y_max = np.zeros((len(imlist), 2))

    i_sample = 0
    all_index = []
    for i, (input, target, index) in enumerate(val_loader):

        index = index.cpu().numpy()
        index = index.astype(int)
        all_index = np.concatenate((all_index, index))

        input = input.cuda()
        print('processing batch', i)

        easiness_heatmaps_set = grad_cam_hp(input)
        easiness_heatmaps_set = np.nan_to_num(easiness_heatmaps_set)
        easiness_heatmaps_set[easiness_heatmaps_set <= 0] = 1e-7
        classifier_heatmaps_set = grad_cam_cls(input, predicted_class[index], target)
        classifier_heatmaps_set = np.nan_to_num(classifier_heatmaps_set)
        classifier_heatmaps_set[classifier_heatmaps_set <= 0] = 1e-7
        predicted_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 0]
        counterfactual_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 1]

        for i_batch in range(index.shape[0]):
            easiness_heatmaps = easiness_heatmaps_set[i_batch, :, :].squeeze()
            predicted_class_heatmaps = predicted_class_heatmaps_set[i_batch, :, :].squeeze()
            counterfactual_class_heatmaps = counterfactual_class_heatmaps_set[i_batch, :, :].squeeze()

            img = cv2.imread(imlist[index[i_batch]])
            img_X_max = np.size(img, axis=0)
            img_Y_max = np.size(img, axis=1)
            img = np.float32(cv2.resize(img, (224, 224))) / 255
            all_X_Y_max[index[i_batch], :] = img_X_max, img_Y_max

            all_images[i_sample, :, :, :] = img

            for i_remain in range(np.size(remaining_mask_size_pool)):
                remaining_mask_size = remaining_mask_size_pool[i_remain]

                if np.mean(predicted_class_heatmaps) == 1e-7:
                    cf_heatmap = easiness_heatmaps * counterfactual_class_heatmaps
                else:
                    cf_heatmap = easiness_heatmaps * (np.amax(predicted_class_heatmaps) - predicted_class_heatmaps) * counterfactual_class_heatmaps

                # print(np.max(cf_heatmap))

                if np.mean(cf_heatmap) == cf_heatmap[0,0]:
                    print('check')

                cf_heatmap = cv2.resize(cf_heatmap, (224, 224))
                threshold = np.sort(cf_heatmap.flatten())[int(-remaining_mask_size * 224 * 224)]

                cf_heatmap_copy = np.zeros_like(cf_heatmap)

                cf_heatmap_copy[cf_heatmap >= threshold] = 1

                all_heatmap_mask[i_sample, i_remain, :, :] = cf_heatmap_copy
            i_sample = i_sample + 1



    all_index = all_index.astype(int)
    # plot and save
    predicted_class = cross_match[:, 0]
    target_class = cross_match[:, 1]

    for i_remain in range(np.size(remaining_mask_size_pool)):
        remaining_mask_size = remaining_mask_size_pool[i_remain]
        all_images_copy = np.copy(all_images)
        for i in range(len(imlist)):
            cur_mask = all_heatmap_mask[i, i_remain, :, :].squeeze()
            cur_img = all_images_copy[i, :, :, :]
            average = compute_average_outof_mask(cur_img, cur_mask)
            average = average[np.newaxis, np.newaxis, :]
            average = np.tile(average, (224, 224, 1))
            cur_img[cur_mask < 1] = average[cur_mask < 1]

            cur_img = np.float32(cur_img)
            cur_img = np.uint8(255 * cur_img)
            cur_mask = np.float32(cur_mask)
            cur_mask = np.uint8(255 * cur_mask)

            img_name = imlist[i]
            img_name = img_name.split('/')[-1]
            img_name = img_name.split('.')[-2]

            if not os.path.exists(
                    "./butterflies/counterfactual_gradcam_simclr_mask/" + str(remaining_mask_size)[:4]):
                os.makedirs(
                    "./butterflies/counterfactual_gradcam_simclr_mask/" + str(remaining_mask_size)[:4])
            cv2.imwrite("./butterflies/counterfactual_gradcam_simclr_mask/" + str(remaining_mask_size)[:4] + '/' + img_name + '_groundtruth_' + str(int(target_class[i])) + '_counter_' + str(int(predicted_class[i])) + '_index_' + str(all_index[i]) + '.png', cur_mask)

            if not os.path.exists(
                    "./butterflies/counterfactual_gradcam_simclr_img/" + str(remaining_mask_size)[:4]):
                os.makedirs(
                    "./butterflies/counterfactual_gradcam_simclr_img/" + str(remaining_mask_size)[:4])
            cv2.imwrite("./butterflies/counterfactual_gradcam_simclr_img/" + str(
                remaining_mask_size)[:4] + '/' + img_name + '_groundtruth_' + str(int(target_class[i])) + '_counter_' + str(int(predicted_class[i])) + '_index_' + str(all_index[i]) + '.png', cur_img)




def cf_proposal_extraction2(val_loader, grad_cam_hp, grad_cam_cls, predicted_class, imlist, remaining_mask_size_pool, cross_match):


    all_images = np.zeros((len(imlist), 224, 224, 3))
    all_heatmap_mask = np.zeros((len(imlist), np.size(remaining_mask_size_pool), 224, 224))

    all_X_Y_max = np.zeros((len(imlist), 2))

    i_sample = 0
    all_index = []
    for i, (input, target, index) in enumerate(val_loader):

        index = index.cpu().numpy()
        index = index.astype(int)
        all_index = np.concatenate((all_index, index))

        input = input.cuda()
        print('processing batch', i)

        easiness_heatmaps_set = grad_cam_hp(input)
        easiness_heatmaps_set = np.nan_to_num(easiness_heatmaps_set)
        easiness_heatmaps_set[easiness_heatmaps_set <= 0] = 1e-7
        classifier_heatmaps_set = grad_cam_cls(input, predicted_class[index], target)
        classifier_heatmaps_set = np.nan_to_num(classifier_heatmaps_set)
        classifier_heatmaps_set[classifier_heatmaps_set <= 0] = 1e-7
        predicted_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 0]
        counterfactual_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 1]

        for i_batch in range(index.shape[0]):
            easiness_heatmaps = easiness_heatmaps_set[i_batch, :, :].squeeze()
            predicted_class_heatmaps = predicted_class_heatmaps_set[i_batch, :, :].squeeze()
            counterfactual_class_heatmaps = counterfactual_class_heatmaps_set[i_batch, :, :].squeeze()

            img = cv2.imread(imlist[index[i_batch]])
            img_X_max = np.size(img, axis=0)
            img_Y_max = np.size(img, axis=1)
            img = np.float32(cv2.resize(img, (224, 224))) / 255
            all_X_Y_max[index[i_batch], :] = img_X_max, img_Y_max

            all_images[i_sample, :, :, :] = img

            for i_remain in range(np.size(remaining_mask_size_pool)):
                remaining_mask_size = remaining_mask_size_pool[i_remain]

                if np.mean(predicted_class_heatmaps) == 1e-7:
                    cf_heatmap = easiness_heatmaps * counterfactual_class_heatmaps
                else:
                    cf_heatmap = easiness_heatmaps * (np.amax(predicted_class_heatmaps) - predicted_class_heatmaps) * counterfactual_class_heatmaps



                cf_heatmap = cv2.resize(cf_heatmap, (224, 224))
                threshold = np.sort(cf_heatmap.flatten())[int(-remaining_mask_size * 224 * 224)]
                cf_heatmap_copy = np.zeros_like(cf_heatmap)
                cf_heatmap_copy[cf_heatmap >= threshold] = 1
                all_heatmap_mask[i_sample, i_remain, :, :] = cf_heatmap_copy

            i_sample = i_sample + 1

    all_index = all_index.astype(int)
    # plot and save
    predicted_class = cross_match[:, 0]
    target_class = cross_match[:, 1]

    for i_remain in range(np.size(remaining_mask_size_pool)):
        remaining_mask_size = remaining_mask_size_pool[i_remain]
        all_images_copy = np.copy(all_images)
        for i in range(len(imlist)):
            cur_mask = all_heatmap_mask[i, i_remain, :, :].squeeze()
            cur_img = all_images_copy[i, :, :, :]

            query_img = show_segment_on_image(cur_img, cur_mask)

            cur_img = np.float32(cur_img)
            cur_img = np.uint8(255 * cur_img)

            img_name = imlist[i]
            img_name = img_name.split('/')[-1]
            img_name = img_name.split('.')[-2]

            if not os.path.exists(
                    "./butterflies/demo_simclr/counterfactual_gradcam_mask/" + str(remaining_mask_size)[:4]):
                os.makedirs(
                    "./butterflies/demo_simclr/counterfactual_gradcam_mask/" + str(remaining_mask_size)[:4])
            cv2.imwrite("./butterflies/demo_simclr/counterfactual_gradcam_mask/" + str(remaining_mask_size)[
                                                                       :4] + '/' + img_name + '_groundtruth_' + str(
                int(target_class[i])) + '_counter_' + str(int(predicted_class[i])) + '_index_' + str(
                all_index[i]) + '.png', query_img)

            if not os.path.exists(
                    "./butterflies/demo_simclr/counterfactual_gradcam_img/" + str(remaining_mask_size)[:4]):
                os.makedirs(
                    "./butterflies/demo_simclr/counterfactual_gradcam_img/" + str(remaining_mask_size)[:4])
            cv2.imwrite("./butterflies/demo_simclr/counterfactual_gradcam_img/" + str(
                remaining_mask_size)[:4] + '/' + img_name + '_groundtruth_' + str(
                int(target_class[i])) + '_counter_' + str(int(predicted_class[i])) + '_index_' + str(
                all_index[i]) + '.png', cur_img)



if __name__ == '__main__':
    main()



