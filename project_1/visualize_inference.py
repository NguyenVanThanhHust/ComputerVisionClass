import collections
import torch
import torchvision
import numpy as np
import cv2
import pickle
from torch.utils import data
from torchvision import transforms, models
from utils import recursive_glob
import re
import math
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

from dataset.ade20k_vis import ADE20K

from model import get_model_instance_segmentation
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import defaultdict, deque
import pickle
import time
import datetime
import torch.distributed as dist

import errno
import os
import os.path as osp
import utils

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    header = 'Test:'
    if not osp.isdir("visulized_img"):
        os.mkdir("visulized_img")
    for idx, (img_paths, images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        img_path = img_paths[0]
        folder = img_path.split("/")
        draw_img = cv2.imread(img_path)
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        bboxes = outputs[0]["boxes"]
        masks = outputs[0]["masks"]
        scores = outputs[0]["scores"]
        draw_path = osp.join("visulized_img", folder[-1])
        for mask in masks:
            mask = mask.mul(255).byte().cpu().numpy()
            mask = np.squeeze(mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(draw_img, contours, -1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(draw_path, draw_img)

def main():
    def collate_fn(batch):
        return tuple(zip(*batch))
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    
    local_path = "../../data/ADE20K_chair/"
    train_transforms = torchvision.transforms.Compose([
        transforms.ToTensor(),
        ])
    # dataset_test = ADE20K(local_path, split="validation", transforms=train_transforms)
    dataset_test = ADE20K(local_path, split="TEST", transforms=train_transforms)
    print("Number of val example: ", dataset_test.__len__())

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        collate_fn=collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)
    state = torch.load("./saved_state_dict/state_dict_21_0.2586383521556854.pt")
    model.load_state_dict(state["model_state_dict"])
    evaluate(model, data_loader_test, device=device)
    print("That's it!")

main()
