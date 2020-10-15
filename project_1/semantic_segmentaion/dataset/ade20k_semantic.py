import collections
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from torch.utils import data
from torchvision import transforms, models
import re
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import albumentations as albu
import numpy as np

class ADE20K(data.Dataset):
    def __init__(
        self,
        root,
        split="TRAIN",
        transforms=None,
        img_size=512,
        augmentations=None,
        preprocessing=None, 
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.img_norm = img_norm
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        if split == "TRAIN":
            with open("dataset/train.pkl", "rb") as fp:
                self.list = pickle.load(fp)
        elif split == "TEST":
            with open("dataset/val.pkl", "rb") as fp:
                self.list = pickle.load(fp)
        elif split == "TEST_FULL":
            with open("dataset/val_full.pkl", "rb") as fp:
                self.list = pickle.load(fp)
        else:
            print("check list file and split type")
        self.non_bbox = {3757, 14300, }

    def __len__(self):
        return int(len(self.list) / 2)

    def __getitem__(self, index):
        path = self.list[2*index]
        idx = int(path[-10:-4])
        while idx in self.non_bbox:
            index += 1
            path = self.list[2*index]
            idx = int(path[-10:-4])
        img_path = self.list[2*index]
        lbl_path = self.list[2*index+1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(lbl_path, 0)
        mask = [mask != 0]
        mask = np.stack(mask, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentations:
            sample = self.augmentations(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=576, min_width=615, always_apply=True, border_mode=0),
        albu.RandomCrop(height=512, width=512, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(576, 615),
        albu.RandomCrop(height=512, width=512, always_apply=True)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


            
#if __name__ == "__main__":
#    def collate_fn(batch):
#        return tuple(zip(*batch))
#    
#    local_path = "../../data/ADE20K_chair/"
#    train_transforms = torchvision.transforms.Compose([
#        transforms.ToTensor(),
#        ])
#    dataset = ADE20K(local_path, split="validation_full", transforms=train_transforms)
#    print("Number of example: ", dataset.__len__())
#    # define training and validation data loaders
#    data_loader = torch.utils.data.DataLoader(
#        dataset, batch_size=1, shuffle=False,
#        collate_fn=collate_fn)
#
#    for i in range(dataset.__len__()):
#        h = dataset.__getitem__(i)
#            
