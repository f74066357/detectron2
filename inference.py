import os
import json
import cv2
import random
import logging
import numpy as np
from collections import OrderedDict
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import detectron2.utils.comm as common
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.visualizer import ColorMode
import pycocotools._mask as _mask


# regist dataset
def register_dataset():
    # purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(
            name=key,
            metadate=get_dataset_instances_meta(),
            json_file=json_file,
            image_root=image_root,
        )


def get_dataset_instances_meta():
    # purpose: get metadata of dataset from DATASET_CATEGORIES
    # return: dict[metadata]
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [
        k["color"] for k in DATASET_CATEGORIES
        if k["isthing"] == 1
    ]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [
        k["name"] for k in DATASET_CATEGORIES
        if k["isthing"] == 1
    ]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name)
    )
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="coco", **metadate
    )


# CHECk annotations
def checkout_dataset_annotation(name="val_2019"):
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5
        )
        vis = visualizer.draw_dataset_dict(d)
        cv2_imshow(vis.get_image()[:, :, ::-1])


# Setup detectron2 logger
setup_logger()

# PATH
DATASET_ROOT = "haha/"
TRAIN_PATH = os.path.join(DATASET_ROOT, "train2017/")
VAL_PATH = os.path.join(DATASET_ROOT, "val2017/")
TEST_PATH = os.path.join(DATASET_ROOT, "test2017/")
TRAIN_JSON = os.path.join(DATASET_ROOT, "train_123.json")
VAL_JSON = os.path.join(DATASET_ROOT, "val_123.json")
TEST_JSON = os.path.join(DATASET_ROOT, "test_123.json")

# CLASS
DATASET_CATEGORIES = [
    {"name": "Nuclie", "id": 1, "isthing": 1, "color": [220, 20, 60]},
]

# DATASET
PREDEFINED_SPLITS_DATASET = {
    "train_2019": (TRAIN_PATH, TRAIN_JSON),
    "val_2019": (VAL_PATH, VAL_JSON),
    "test_2019": (TEST_PATH, TEST_JSON),
}


register_dataset()

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)
cfg.DATASETS.TRAIN = ("train_2019",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 3000  # iterations
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class

cfg.OUTPUT_DIR = "output2/"
cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, "model_final.pth"
)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001  # set testing threshold
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
predictor = DefaultPredictor(cfg)


def encode(bimask):
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order="F"))[0]


def area(rleObjs):
    if type(rleObjs) == list:
        return _mask.area(rleObjs)
    else:
        return _mask.area([rleObjs])[0]


def toBbox(rleObjs):
    if type(rleObjs) == list:
        return _mask.toBbox(rleObjs)
    else:
        return _mask.toBbox([rleObjs])[0]


data = []
images = [
    "TCGA-A7-A13E-01Z-00-DX1.png",
    "TCGA-50-5931-01Z-00-DX1.png",
    "TCGA-G2-A2EK-01A-02-TSB.png",
    "TCGA-AY-A8YK-01A-01-TS1.png",
    "TCGA-G9-6336-01Z-00-DX1.png",
    "TCGA-G9-6348-01Z-00-DX1.png",
]
for i in range(len(images)):

    image = cv2.imread(DATASET_ROOT + "test2017/" + images[i])
    image_id = i + 1
    category_id = 1
    seglist = []
    boxlist = []
    scorelist = []
    outputs = predictor(image)
    ''' # Visualize
    v = Visualizer(
        image[:, :, ::-1],
        metadata=MetadataCatalog.get("val_2019"),
        scale=0.6,
        instance_mode=ColorMode.IMAGE_BW
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("show",out.get_image()[:, :, :])
    cv2.waitKey(0)
    '''
    for score in outputs["instances"].scores:
        scorelist.append(float(score))
    masks = outputs["instances"].pred_masks.cpu()
    for mask in masks:
        mask = mask.long()
        mask = mask.numpy()
        mmask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        m = np.asfortranarray(np.uint8(mmask))
        seg = encode(m)
        # print(seg)
        segcounts = seg["counts"].decode("UTF-8")
        tmpseg = {
            "size": seg["size"],
            "counts": segcounts,
        }
        seglist.append(tmpseg)
        box = toBbox(seg)
        tmpb = []
        for b in box:
            tmpb.append(int(b))
        boxlist.append(tmpb)
    print(len(boxlist))
    for i in range(len(boxlist)):
        k = {
            "image_id": int(image_id),
            "bbox": boxlist[i],
            "score": scorelist[i],
            "category_id": int(category_id),
            "segmentation": seglist[i],
        }
        data.append(k)

with open("answer.json", "w") as outfile:
    json.dump(data, outfile, indent=4)
