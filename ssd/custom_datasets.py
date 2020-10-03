import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
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


class ADE20K(data.Dataset):
    def __init__(
        self,
        root,
        split="TRAIN",
        transforms=None,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.augmentations = augmentations
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
        else:
            print("check list file and split type")
        self.non_bbox = {3757, 14300, }
    def __len__(self):
        return int(len(self.list) / 2)
        # return 10

    def __getitem__(self, index):
        path = self.list[2*index]
        idx = int(path[-10:-4])
        while idx in self.non_bbox:
            index += 1
            path = self.list[2*index]
            idx = int(path[-10:-4])
        img_path = self.list[2*index]
        lbl_path = self.list[2*index+1]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(lbl_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        difficulties = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        # print(image_id, boxes.shape)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd

        if self.transforms is not None:
            img, boxes, labels, difficulties = self.transforms(img, boxes, labels, difficulties, split=self.split)
        return img, boxes, labels, difficulties

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each


class ADE20K_full(data.Dataset):
    def __init__(
        self,
        root,
        split="test_full",
        transforms=None,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        if split == "TEST_FULL":
            with open("dataset/val_full.pkl", "rb") as fp:
                self.list = pickle.load(fp)
        else:
            print("check list file and split type")
        self.non_bbox = {3757, 14300, }
    def __len__(self):
        return int(len(self.list) / 2)
        # return 10

    def __getitem__(self, index):
        path = self.list[2*index]
        idx = int(path[-10:-4])
        while idx in self.non_bbox:
            index += 1
            path = self.list[2*index]
            idx = int(path[-10:-4])
        img_path = self.list[2*index]
        lbl_path = self.list[2*index+1]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(lbl_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        # convert everything into a torch.Tensor
        if num_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            difficulties = torch.zeros((num_objs,), dtype=torch.int64)
        else:
            boxes.append([0, 0, 1.0, 1.0])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            difficulties = torch.zeros((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        # print(image_id, boxes.shape)
        if num_objs > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = 0

        # suppose all instances are not crowd

        if self.transforms is not None:
            img, boxes, labels, difficulties = self.transforms(img, boxes, labels, difficulties, split=self.split)
        return img, boxes, labels, difficulties

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
