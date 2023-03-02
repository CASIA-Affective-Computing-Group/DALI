# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 13:20:46 2022

@author: xumin
"""

#import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

#from utils.randaugment import RandomAugment
#from utils.utils_algo import generate_uniform_cv_candidate_labels, generate_noise_labels,generate_uniform_cv_candidate_labels_PiCO


def load_cifar100N(args):
    
    #######################################################
    print ('obtain train_loader')
    ## train_loader: (data, labels), only read data (target data: (60000, 32, 32, 3))
    temp_train = torchvision.datasets.CIFAR100(root='../dataset/CIFAR100', train=True, download=True)
    data_train, dlabels_train = temp_train.data, temp_train.targets # (50000, 32, 32, 3)
    assert np.min(dlabels_train) == 0, f'min(dlabels) != 0'

    ## train_loader: train_givenY
    dlabels_train = np.array(dlabels_train).astype('int')
    num_sample = len(dlabels_train)
    noise_label = torch.load('../dataset/CIFAR-N/CIFAR-100_human.pt') 
    train_givenY = np.eye(100)[noise_label['noisy_label'] .reshape(-1)]  
    print(train_givenY.shape)
    bingo_rate = np.sum(train_givenY[np.arange(num_sample), dlabels_train] == 1.0) / num_sample
    print('Average noise rate: ', 1 - bingo_rate)
    dlabels_train = np.array(dlabels_train).astype('float')
    train_givenY = np.array(train_givenY).astype('float')
    plabels_train = (train_givenY!=0).astype('float')

    partial_matrix_dataset = Augmentention(data_train, plabels_train, dlabels_train, train_flag=True,args=args)
    partial_matrix_train_loader = torch.utils.data.DataLoader(
        dataset=partial_matrix_dataset, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True, 
        drop_last=True)


    #######################################################
    print ('obtain test_loader')
    temp_test = torchvision.datasets.CIFAR100(root='../dataset/CIFAR100', train=False, download=True)
    data_test, dlabels_test = temp_test.data, temp_test.targets # (50000, 32, 32, 3)
    assert np.min(dlabels_test) == 0, f'min(dlabels) != 0'

    ## (data, dlabels) -> test_loader
    test_dataset = Augmentention(data_test, dlabels_test, dlabels_test, train_flag=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False)

    return partial_matrix_train_loader, train_givenY, test_loader


class Augmentention(Dataset):
    def __init__(self, images, plabels, dlabels, train_flag=True,args=None):
        self.images = images
        self.plabels = plabels
        self.dlabels = dlabels
        self.train_flag = train_flag
        if args!=None:
            self.augment_type = args.augment_type
        normalize_mean = (0.5071, 0.4865, 0.4409)
        normalize_std  = (0.2673, 0.2564, 0.2762)
        if self.train_flag == True:
            self.weak_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(), 
                transforms.Normalize(normalize_mean, normalize_std) # the mean and std on cifar training set
                ])
            self.strong_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                RandomAugment(3, 5),
                transforms.ToTensor(), 
                transforms.Normalize(normalize_mean, normalize_std) # the mean and std on cifar training set
                ])
        else:
            self.test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
                ])


    def __len__(self):
        return len(self.dlabels)
        
    def __getitem__(self, index):
        if self.train_flag == True:
            each_image_w1 = self.weak_transform(self.images[index])
            each_image_s1 = self.strong_transform(self.images[index])
            each_plabel = self.plabels[index]
            each_dlabel = self.dlabels[index]
            return each_image_w1, each_image_s1, each_plabel, each_dlabel, index
        else:
            each_image = self.test_transform(self.images[index])
            each_dlabel = self.dlabels[index]
            return each_image, each_dlabel


import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw



def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    #assert -30 <= v <= 30
    #if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)



def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert v >= 0.0
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert 0 <= v
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):  #[0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

    
def augment_list():  
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l

    
class RandomAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

        
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            img = op(img, val) 
        cutout_val = random.random() * 0.5 
        img = Cutout(img, cutout_val) #for fixmatch
        return img
if __name__ =='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch implementation of noise partial label learning')
    parser.add_argument('--augment_type', default='pico', type=str, help='augment_type')
    parser.add_argument('--noisy_type', default='flip', type=str, help='flip or pico')
    parser.add_argument('--workers', default=6, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')

    args = parser.parse_args()
    load_cifar100N(args)