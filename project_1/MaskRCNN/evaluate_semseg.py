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

from dataset.ade20k_full_semantic import ADE20K_full_semantic

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
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    list_iou = []
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        tmp_res = outputs[0]
        out_masks = tmp_res["masks"].mul(255).byte().cpu().numpy()
        tar_mask = targets[0]["masks"].mul(255).byte().cpu().numpy().squeeze()
        out_mask = np.zeros(shape = tar_mask.shape, dtype= tar_mask.dtype)
        for each_out_mask in out_masks:
            out_mask = np.logical_or(out_mask, each_out_mask)
        out_mask = out_mask.squeeze()
        intersection = np.logical_and(out_mask, tar_mask)
        union = np.logical_or(out_mask, tar_mask)
        iou_score = np.sum(intersection) / np.sum(union)
        print(iou_score)
        if math.isnan(iou_score):
            k = 1
        else:
            list_iou.append(iou_score)
        del out_mask, tar_mask
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    print("Mean iou: {}".format(sum(list_iou)/len(list_iou)))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


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
    dataset_test = ADE20K_full_semantic(local_path, split="validation_full", transforms=train_transforms)
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
