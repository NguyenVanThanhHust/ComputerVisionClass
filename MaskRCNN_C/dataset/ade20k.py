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
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        if split == "TRAIN":
            with open("dataset/train.pkl", "rb") as fp:
                self.list = pickle.load(fp)
            with open("dataset/train_foldable.pkl", "rb") as fp:
                self.f_list = pickle.load(fp)
                self.foldable = set(self.f_list)
            with open("dataset/train_non_foldable.pkl", "rb") as fp:
                self.nf_list = pickle.load(fp)
                self.non_foldable = set(self.nf_list)
            
        elif split == "TEST":
            with open("dataset/val.pkl", "rb") as fp:
                self.list = pickle.load(fp)
            with open("dataset/val_foldable.pkl", "rb") as fp:
                self.f_list = pickle.load(fp)
                self.foldable = set(self.f_list)
            with open("dataset/val_non_foldable.pkl", "rb") as fp:
                self.nf_list = pickle.load(fp)
                self.non_foldable = set(self.nf_list)
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
        tmp = img_path.split("/")
        img_name = tmp[-1]
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
        labels = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            if img_name in self.foldable:
                labels.append(1)
            else:
                labels.append(2)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

def test_batchify_fn(data):
    error_msg = "batch must contain tensors, tuples or lists; found {}"
    if isinstance(data[0], (str, torch.Tensor)):
        return list(data)
    elif isinstance(data[0], (tuple, list)):
        data = zip(*data)
        return [test_batchify_fn(i) for i in data]
    raise TypeError((error_msg.format(type(batch[0]))))
    
#            
#if __name__ == "__main__":
#    def collate_fn(batch):
#        return tuple(zip(*batch))
#    
#    local_path = "../../data/ADE20K_chair/"
#    train_transforms = torchvision.transforms.Compose([
#        transforms.ToTensor(),
#        ])
#    dataset = ADE20K(local_path, split="TRAIN", transforms=train_transforms)
#    print("Number of example: ", dataset.__len__())
#    # define training and validation data loaders
#    data_loader = torch.utils.data.DataLoader(
#        dataset, batch_size=1, shuffle=False,
#        collate_fn=collate_fn)
#
#    for i in range(dataset.__len__()):
#        h = dataset.__getitem__(i)
#            