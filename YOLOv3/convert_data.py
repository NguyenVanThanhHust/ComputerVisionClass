import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import collections
import pickle

class ADE20K(Dataset):
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
        mask = Image.open(lbl_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        new_boxes = []
        img_cv = cv2.imread(img_path)
        height, width, channels = img_cv.shape
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            x_center = (xmax + xmin) /(2 * width)
            y_center = (ymax + ymin) /(2 * height)
            new_w = (xmax - xmin) / (width)
            new_h = (ymax - ymin) / (height)
            boxes.append([xmin, ymin, xmax, ymax])
            new_boxes.append([x_center, y_center, new_w, new_h])
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        difficulties = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        # print(image_id, boxes.shape)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd

        return img_path, new_boxes, labels, difficulties

class ADE20K_full(Dataset):
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
        new_boxes = []
        img_cv = cv2.imread(img_path)
        height, width, channels = img_cv.shape
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            x_center = (x_max + x_min) /(2 * width)
            y_center = (y_max + y_min) /(2 * height)
            new_w = (x_max - x_min) / (width)
            new_h = (y_max - y_min) / (height)
            boxes.append([xmin, ymin, xmax, ymax])
            new_boxes.append([x_center, y_center, new_w, new_h])

        return img_path, new_boxes, labels, difficulties

if __name__ == "__main__":
    ade20k = ADE20K(root="../../data/ADE20K_chair", split="TEST")
    f = open("data/custom/valid.txt", "w")
    for k in range(ade20k.__len__()):
        h = ade20k.__getitem__(k)
        img_path, boxes, labels, difficulties = h
        f.write(img_path)
        f.write(" ")
        for box in boxes:
            f.write(str(0))
            f.write(" ")
            f.write(str(box[0]))
            f.write(" ")
            f.write(str(box[1]))
            f.write(" ")
            f.write(str(box[2]))
            f.write(" ")
            f.write(str(box[3]))
            f.write(" ")
        f.write("\n")
