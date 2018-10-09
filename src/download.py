from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pickle

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
  outputs = []
  for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]
    filename = img["file_name"]
    captions = coco_caps.loadAnns(img["id"])
    image = io.imread(dataDir + "/" + dataType + "/" + filename)
    outputs.append(image, captions)
    
with open(dataDir + "pickles/output.pkl", "wb") as f:
  pickle.dump(outputs, f)

"""
# data load
dataDir = "../data"
dataType = "val2017"
# initialize COCO api for caption annotations
annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)

import ipdb; ipdb.set_trace()
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
imgIds = coco.getImgIds(catIds=catIds )
imgIds = coco.getImgIds(imgIds = [324158])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# 画像
I = io.imread(img["coco_url"])
plt.axis('off')
plt.imshow(I)
plt.savefig("image.jpg")

# load and display caption annotations
# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)
annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
"""