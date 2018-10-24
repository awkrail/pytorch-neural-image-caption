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
parser.add_argument("--output_train_path", "-o_train", type=str, required=True)
parser.add_argument("--output_val_path", "-o_val", type=str, required=True)
args = parser.parse_args()

dataTypes = ["train2017", "val2017"]
dataDir = "../data"

for dataType in dataTypes:
  if dataType == "train2017":
    output_path = args.output_train_path
  else:
    output_path = args.output_val_path

  annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)
  capFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
  coco = COCO(annFile)
  coco_caps = COCO(capFile)

  # get all images
  catIds = coco.getCatIds()
  imgIds = coco.getImgIds()
  imgIds = coco.getImgIds(imgIds = imgIds)
  all_captions = []
  for imgId in sorted(imgIds):
    img = coco.loadImgs(imgId)[0]
    filename = img["file_name"]
    annIds = coco_caps.getAnnIds(imgIds=img["id"])
    anns = coco_caps.loadAnns(annIds)
    captions = [annDict["caption"].lower() for annDict in anns]
    for caption in captions:
        all_captions.append(caption)
    image = io.imread(args.input_path + dataType + "/" + filename)
    if image.shape[0] <= 32 or image.shape[1] <= 32:
        continue
    output_dict = {
            "image" : image,
            "captions" : captions
            }
    with open(output_path + str(imgId) + ".pkl", "wb") as f:
        pickle.dump(output_dict, f)
    print(imgId, ", done!")
  with open(input_path + "/" + dataType + "_captions.pkl", "wb") as f:
      pickle.dump(all_captions, f)
