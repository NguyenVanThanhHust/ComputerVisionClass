import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset.ade20k_semantic import ADE20K, get_validation_augmentation, to_tensor, get_preprocessing


data_path = ""

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['chair']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
trained_model = torch.load(ENCODER + "_model.pth")


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

valid_dataset = ADE20K(root=data_path, split="TEST_FULL", augmentations=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
print("number of evalution sample: {}".format(valid_dataset.__len__()))
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

valid_epoch = smp.utils.train.ValidEpoch(
    model = trained_model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


max_score = 0
ts = time.time()
valid_logs = valid_epoch.run(valid_loader)
te = time.time()

infer_st = (te-ts)/valid_dataset.__len__()
print("Evalution time: {}. for single example: {}".format(te-ts, infer_st))
if max_score < valid_logs['iou_score']:
    max_score = valid_logs['iou_score']

print(valid_logs)
