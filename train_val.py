# coding=utf-8
import os
import json
import numpy as np
from tqdm import tqdm
from imantics import Mask
from imantics import Image
from imantics import Category
from imantics import Dataset
import cv2
import sys

print(sys.getrecursionlimit())
sys.setrecursionlimit(2500000)
dataset = Dataset("My")

alllist = [
    "TCGA-18-5592-01Z-00-DX1",
    "TCGA-A7-A13F-01Z-00-DX1",
    "TCGA-B0-5711-01Z-00-DX1",
    "TCGA-G9-6356-01Z-00-DX1",
    "TCGA-HE-7130-01Z-00-DX1",
    "TCGA-21-5784-01Z-00-DX1",
    "TCGA-AR-A1AK-01Z-00-DX1",
    "TCGA-CH-5767-01Z-00-DX1",
    "TCGA-G9-6362-01Z-00-DX1",
    "TCGA-KB-A93J-01A-01-TS1",
    "TCGA-21-5786-01Z-00-DX1",
    "TCGA-AR-A1AS-01Z-00-DX1",
    "TCGA-G9-6363-01Z-00-DX1",
    "TCGA-NH-A8F7-01A-01-TS1",
    "TCGA-B0-5698-01Z-00-DX1",
    "TCGA-E2-A14V-01Z-00-DX1",
    "TCGA-HE-7128-01Z-00-DX1",
    "TCGA-RD-A8N9-01A-01-TS1",
    "TCGA-B0-5710-01Z-00-DX1",
    "TCGA-E2-A1B5-01Z-00-DX1",
    "TCGA-DK-A2I6-01A-01-TS1",
    "TCGA-49-4488-01Z-00-DX1",
    "TCGA-38-6178-01Z-00-DX1",
    "TCGA-HE-7129-01Z-00-DX1",
]

trainlist = [
    "TCGA-18-5592-01Z-00-DX1",
    "TCGA-A7-A13F-01Z-00-DX1",
    "TCGA-B0-5711-01Z-00-DX1",
    "TCGA-G9-6356-01Z-00-DX1",
    "TCGA-HE-7130-01Z-00-DX1",
    "TCGA-21-5784-01Z-00-DX1",
    "TCGA-AR-A1AK-01Z-00-DX1",
    "TCGA-CH-5767-01Z-00-DX1",
    "TCGA-G9-6362-01Z-00-DX1",
    "TCGA-KB-A93J-01A-01-TS1",
    "TCGA-21-5786-01Z-00-DX1",
    "TCGA-AR-A1AS-01Z-00-DX1",
    "TCGA-G9-6363-01Z-00-DX1",
    "TCGA-NH-A8F7-01A-01-TS1",
    "TCGA-B0-5698-01Z-00-DX1",
    "TCGA-E2-A14V-01Z-00-DX1",
    "TCGA-HE-7128-01Z-00-DX1",
    "TCGA-RD-A8N9-01A-01-TS1",
    "TCGA-B0-5710-01Z-00-DX1",
    "TCGA-E2-A1B5-01Z-00-DX1",
]

vallist = [
    "TCGA-DK-A2I6-01A-01-TS1",
    "TCGA-49-4488-01Z-00-DX1",
    "TCGA-38-6178-01Z-00-DX1",
    "TCGA-HE-7129-01Z-00-DX1",
]

pathlist = vallist

for p in range(len(pathlist)):
    path = "../../dataset/dataset/train/" + pathlist[p] + "/masks/"
    file = (
        "../../dataset/dataset/train/" + pathlist[p] + "/images/" +
        pathlist[p] + ".png"
    )
    image = cv2.imread(file)[:, :, ::-1]
    image = Image(image, id=p)
    image.file_name = "{}.png".format(pathlist[p])
    image.path = file
    print(file)
    for index, i in enumerate(tqdm(os.listdir(path))):
        # print(len(os.listdir(path)))
        mask_file = os.path.join(path, i)
        mask = cv2.imread(mask_file, 0)
        t = cv2.imread(file)
        if t.shape[:-1] != mask.shape:
            h, w, _ = t.shape
            mask = cv2.resize(mask, (w, h), cv2.INTER_CUBIC)

        # Category
        t = Category("Nuclie")
        t.id = 1  # class id number
        mask = Mask(mask)
        image.add(mask, t)
    dataset.add(image)

t = dataset.coco()  # coco format
with open("t_123.json", "w") as output_json_file:  # output json
    json.dump(t, output_json_file, indent=4)
