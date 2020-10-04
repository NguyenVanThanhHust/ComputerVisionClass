# ComputerVisionClass

1. Dataset:
- Download dataset from https://groups.csail.mit.edu/vision/datasets/ADE20K/ and unzip
- Put it in 'data' folder
Your folder structure should look like this

.
├── ...
├── projects                        # Documentation files (alternatively `doc`)
│   ├── data/ADE20K_2016_07_26      # Dataset folder
│   ├── ComputerVisionClass         # Source 
│       ├──MaskRCNN                 # Instance Segmentation with Mask RCNN 
│       ├──YOLOv3                   # Instance Detection with YOLOv3
│       ├──semantic segmentation    # Semantic Segmentation with Mask RCNN
└── ...

2. To reproduce results
- Convert ADE2OK dataset to suitable format with ComputerVisionClass/MaskRCNN/Filter_Chair.ipynb
- For instance segmentation of MaskRCNN
```cd ~/projects/ComputerVisionClass/MaskRCNN```
To train
```nohup python3.7 -u finetune.py &> train.out &```
To test
```nohup python3.7 -u evaluate.py &> eval.out &```

- For instance detection of YOLOv3
```cd ~/projects/ComputerVisionClass/YOLOv3```
To train
```nohup python3.7 -u train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74 &> train.out &```
To train
```nohup python3.7 -u --model_def config/yolov3-custom.cfg --data_config config/custom.data --weights_path checkpoints/yolov3_ckpt_34.pth --class_path data/custom/classes.names &> eval.out &```
To visualize result
```python3.7 visualize_img.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --weights_path checkpoints/yolov3_ckpt_34.pth --class_path data/custom/classes.names```

- For semantic segmentation 
```cd ~/projects/ComputerVisionClass/semantic\ segmentation```
```nohup python3.7 -u train.py &> train.out &```
```nohup python3.7 -u test.py &> eval.out &```

Training log is at train.out
Test log is at test.out