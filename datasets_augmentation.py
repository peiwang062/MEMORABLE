import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import os
from PIL import Image

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

_NUM_CLASSES = {
    'gull': 5,
    'butterflies': 5,
    'butterflies_simclr': 5,
    'butterflies_s300': 5,
    'butterflies_s300_after_labeling': 5,
    'butterflies_s300_wo_shuffle': 5,
    'butterflies_s100_rest': 5,
    'butterflies_s300_CE': 5,
    'butterflies_s300_Layer32CE': 5,
    'butterflies_HumanLabeled_s30': 5,
    'butterflies_HumanLabeled_s100': 5,
    'butterflies_HumanLabeled_s300': 5,
    'butterflies_SelfLabeled_s100': 5,
    'butterflies_whole': 5,
    'butterflies_whole_s300': 5,
    'butterflies_Dt': 5,
    'butterflies_s300_CE_Dt': 5,
    'butterflies_Lt': 5,
    'butterflies_s300_CE_Lt': 5,
    'butterflies_EXPLAIN': 5,
    'butterflies_STRICT': 5,
    'butterflies_curriculum': 5,
    'butterflies_relation_sup': 5,
    'butterflies_relation_pq': 5,
    'chineseChars': 3,
    'chineseChars_Dt': 3,
    'chineseChars_Lt': 3,
    'chineseChars_EXPLAIN': 3,
    'chineseChars_STRICT': 3,
    'chineseChars_curriculum': 3,
    'butterflies_for_inception': 5,
    'butterflies_BlackboxIMT': 5,
    'chineseChars_BlackboxIMT': 3,
    'gull_simclr': 5,
    'gull_after_labeling': 5,
    'gull_whole': 5,
}



def default_loader(path):
    return Image.open(path).convert('RGB')


def default_loader_mnist(path):
    return Image.open(path).convert('L')


# def default_loader(path):
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, imindex = line.strip().split()
            imlist.append((impath, int(imlabel), int(imindex)))

    return imlist


def default_flist_reader_CE(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, imcounter, imindex = line.strip().split()
            imlist.append((impath, int(imlabel), int(imcounter), int(imindex)))

    return imlist


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray



class ImageFilelist(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target, index = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imlist)


class ImageFilelist_CE(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader_CE, loader=default_loader):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target, counter, index = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, counter, index

    def __len__(self):
        return len(self.imlist)




class ImageFilelist_test(data.Dataset):
    def __init__(self, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target, index = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)





def gull(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/CUBgull/CUBgull_gt_tr.txt'
    val_list = './datasets/CUBgull/CUBgull_gt_te.txt'

    # train_list = './datasets/gull/gull_gt_tr.txt'
    # val_list = './datasets/gull/gull_gt_te.txt'

    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds




def butterflies(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_simclr(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_s300_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist_test(
                flist=train_list,
                # transform=transforms.Compose([
                #     transforms.RandomResizedCrop(224),
                #     transforms.RandomHorizontalFlip(),
                #     transforms.ToTensor(),
                #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # ])
                transform=TransformsSimCLR(size=224).test_transform
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist_test(
                flist=val_list,
                # transform=transforms.Compose([
                #     transforms.Resize(256),
                #     transforms.CenterCrop(224),
                #     transforms.ToTensor(),
                #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # ])
                transform=TransformsSimCLR(size=224).test_transform
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_s300(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_s300_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_s300_after_labeling(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_turker_label_gt_s300_xmaxgrad-simclr_labeled_tr2.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_s300_wo_shuffle(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_s300_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_s300_CE(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_s300_CE.txt'
    # train_list = './datasets/butterflies_crop/butterflies_gt_s300_Layer32CE.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist_CE(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_s300_Layer32CE(batch_size, train=True, val=True, **kwargs):
    # train_list = './datasets/butterflies_crop/butterflies_gt_s300_CE.txt'
    train_list = './datasets/butterflies_crop/butterflies_gt_s300_Layer32CE.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist_CE(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_s100_rest(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_s100_rest_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_HumanLabeled_s30(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_s30_rest_mimicHuman_labeled_tr.txt'
    # train_list = './datasets/butterflies_crop/butterflies_gt_s30_rest_SelfLinear_mimicHuman_labeled_tr.txt'
    # train_list = './datasets/butterflies_crop/butterflies_gt_s30_rest_randomLabel_mimicHuman_labeled_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_HumanLabeled_s100(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_s100_rest_mimicHuman_labeled_tr.txt'
    # train_list = './datasets/butterflies_crop/butterflies_gt_s100_rest_SelfLinear_mimicHuman_labeled_tr.txt'
    # train_list = './datasets/butterflies_crop/butterflies_gt_s100_rest_randomLabel_mimicHuman_labeled_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_HumanLabeled_s300(batch_size, train=True, val=True, **kwargs):
    # train_list = './datasets/butterflies_crop/butterflies_gt_s300_rest_RANDOM_mimicHuman_labeled_tr.txt'
    # train_list = './datasets/butterflies_crop/butterflies_gt_s300_rest_MaxGrad_mimicHuman_labeled_tr.txt'
    # train_list = './datasets/butterflies_crop/butterflies_gt_s300_rest_uniform_60_mimicHuman_labeled_tr.txt'
    # train_list = './datasets/butterflies_crop/butterflies_gt_s300_rest_uniform_80_mimicHuman_labeled_tr.txt'
    # train_list = './datasets/butterflies_crop/butterflies_gt_s300_rest_omniIMT_mimicHuman_labeled_tr.txt'
    # train_list = './datasets/butterflies_crop/butterflies_gt_s300_rest_imiIMT_mimicHuman_labeled_tr.txt'
    train_list = './datasets/butterflies_crop/butterflies_gt_s300_rest_bbIMT_mimicHuman_labeled_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_SelfLabeled_s100(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_s100_rest_SelfLinearClassifier_labeled_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_for_inception(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_Dt(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/butterflies_crop/' + selection + '/butterflies_Dt_gt_s300_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_s300_CE_Dt(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/butterflies_crop/' + selection + '/butterflies_Dt_gt_s300_CE_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist_CE(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_Lt(batch_size, train=True, val=True, selection='MMT_largemargin', **kwargs):
    train_list = './datasets/butterflies_crop/' + selection + '/butterflies_Lt_gt_s300_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_s300_CE_Lt(batch_size, train=True, val=True, selection='MMT_largemargin', **kwargs):
    train_list = './datasets/butterflies_crop/' + selection + '/butterflies_Lt_gt_s300_CE_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_relation_sup(batch_size, train=True, val=True, selection='relation_fewshot', **kwargs):
    train_list = './datasets/butterflies_crop/' + selection + '/butterflies_relation_sup_gt.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_relation_pq(batch_size, train=True, val=True, selection='relation_fewshot', **kwargs):
    train_list = './datasets/butterflies_crop/' + selection + '/butterflies_relation_pq_gt.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_BlackboxIMT(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/butterflies_crop/' + selection + '/butterflies_BlackboxIMT_gt_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_EXPLAIN(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/butterflies_crop/' + selection + '/butterflies_EXPLAIN_gt_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_STRICT(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/butterflies_crop/' + selection + '/butterflies_STRICT_gt_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_curriculum(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/butterflies_crop/' + selection + '/butterflies_curriculum_gt_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_whole(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_tr_te.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_tr_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def butterflies_whole_s300(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/butterflies_crop/butterflies_gt_s300_tr.txt'
    val_list = './datasets/butterflies_crop/butterflies_gt_s300_tr.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def chineseChars(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/chinese_chars/ChineseChars_gt_tr.txt'
    val_list = './datasets/chinese_chars/ChineseChars_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def chineseChars_Dt(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/chinese_chars/' + selection + '/ChineseChars_Dt_gt_tr.txt'
    val_list = './datasets/chinese_chars/ChineseChars_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def chineseChars_Lt(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/chinese_chars/' + selection + '/ChineseChars_Lt_gt_tr.txt'
    val_list = './datasets/chinese_chars/ChineseChars_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def chineseChars_BlackboxIMT(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/chinese_chars/' + selection + '/ChineseChars_BlackboxIMT_gt_tr.txt'
    val_list = './datasets/chinese_chars/ChineseChars_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def chineseChars_EXPLAIN(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/chinese_chars/' + selection + '/ChineseChars_EXPLAIN_gt_tr.txt'
    val_list = './datasets/chinese_chars/ChineseChars_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def chineseChars_STRICT(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/chinese_chars/' + selection + '/ChineseChars_STRICT_gt_tr.txt'
    val_list = './datasets/chinese_chars/ChineseChars_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def chineseChars_curriculum(batch_size, train=True, val=True, selection='random', **kwargs):
    train_list = './datasets/chinese_chars/' + selection + '/ChineseChars_curriculum_gt_tr.txt'
    val_list = './datasets/chinese_chars/ChineseChars_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def gull_simclr(batch_size, train=True, val=True, **kwargs):
    train_list = '/data7/peiwang/projects/MT_scalable_learning/datasets/CUBgull/CUBgull_gt_tr.txt'
    val_list = '/data7/peiwang/projects/MT_scalable_learning/datasets/CUBgull/CUBgull_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist_test(
                flist=train_list,
                # transform=transforms.Compose([
                #     transforms.RandomResizedCrop(224),
                #     transforms.RandomHorizontalFlip(),
                #     transforms.ToTensor(),
                #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # ])
                transform=TransformsSimCLR(size=224).test_transform
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist_test(
                flist=val_list,
                # transform=transforms.Compose([
                #     transforms.Resize(256),
                #     transforms.CenterCrop(224),
                #     transforms.ToTensor(),
                #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # ])
                transform=TransformsSimCLR(size=224).test_transform
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def gull_after_labeling(batch_size, train=True, val=True, **kwargs):
    train_list = './gull/gull_turker_label_gt_xmaxgrad-gull_labeled_tr2.txt'
    val_list = './datasets/CUBgull/CUBgull_gt_te.txt'
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def gull_whole(batch_size, train=True, val=True, **kwargs):
    train_list = './datasets/CUBgull/ALLgull_gt_tr.txt'
    val_list = './datasets/CUBgull/CUBgull_gt_te.txt'

    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building data loader with {} workers".format(num_workers))
    ds = []

    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=train_list,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        print("Training data size: {}".format(len(train_loader.dataset)))
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFilelist(
                flist=val_list,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        print("Testing data size: {}".format(len(test_loader.dataset)))
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds