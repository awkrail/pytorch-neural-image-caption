from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pickle
import argparse

"""
Parse MSCOCO Dataset

Output:
  pickle file
  e.x)
    [(
      image(C * H * W)(for pytorch)
      ,[
      captions..
    ])]
"""
parser = argparse.ArgumentParser(description="indicate onput path dir")
parser.add_argument("--input_path", "-i", type=str, required=True)
parser.add_argument("--output_path", "-o", type=str, required=True)
args = parser.parse_args()

dataTypes = ["train2017", "val2017"]
dataDir = "../data"

for dataType in dataTypes:
  annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)
  capFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
  coco = COCO(annFile)
  coco_caps = COCO(capFile)

  # get all images
  catIds = coco.getCatIds()
  imgIds = coco.getImgIds()
  imgIds = coco.getImgIds(imgIds = imgIds)
  for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]
    filename = img["file_name"]
    captions = coco_caps.loadAnns(img["id"])
    image = io.imread(args.input_path + dataType + "/" + filename)
    output_dict = {
      "image" : image,
      "captions" : captions
    }
    with open(args.output_path + str(imgId) + ".pkl", "wb") as f:
      pickle.dump(output_dict, f)