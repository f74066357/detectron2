# VRDL_InstanceSegmentation
## Use MASK RCNN in detectron2
---

## Dataset:
24 training images with 14,598 nuclear  
6 test images with 2,360 nuclear

## Dataset Preparation
I split 24 images to 20 for training set and 4 for validation set, totally 11691 nuclear: 2907 nuclear.  
Then, convert the data to COCO dataset format for following training.
I provide a python code ```train_val.py```, just moddify the dataset path and pathlist in line 74 to image path list.
and install imantics by ```pip install imantics```
```
python train_val.py
```

## Install  Dependencies
```
1. Linux or macOS with Python ≥ 3.6
2. PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation
3. git clone https://github.com/facebookresearch/detectron2.git
   python -m pip install -e detectron2
4. pip install opencv-python
5. pip install pyyaml
6. pip install pycocotools
```

## Train 
```
python train.py
```

* pre-train model: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" in model zoo

## Inference & Generate answer
```
python inference.py
```

## My training weight
https://drive.google.com/drive/folders/1shJ9dnprTWWBb4SLl3w-z_qZllXxRuNi?usp=sharing
PUT this folder output2/ under detectron2/

## Colab link


## Reference

