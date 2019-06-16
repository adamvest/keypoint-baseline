import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import shutil
import os

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "./configs/my_configs/res_50_keypoints_modified.yaml"

#cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 10

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
#cfg.merge_from_list(["MODEL.MASK_ON", True])
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")


def imwrite(img, path_save):
    cv2.imwrite(path_save, img)


dir_coco_test = "./maskrcnn_benchmark/data/datasets/reno_rectified/test/"
dir_save = "./demo/all_rec_results/"
dir_pred_save  = "./demo/all_rec_vector_segments"

if os.path.isdir(dir_save):
    shutil.rmtree(dir_save)

if os.path.isdir(dir_pred_save):
    shutil.rmtree(dir_pred_save)

os.makedirs(dir_save)
os.makedirs(dir_pred_save)

for root, dirs, filenames in os.walk(dir_coco_test):
    filenames.sort()

    for ind, name in enumerate(filenames):
        if not name.endswith('.jpg'):
            continue

        img_path = os.path.join(root, name)
        img = cv2.imread(img_path)
        fname_base = name.split(".")[0]
        masked_image = coco_demo.run_on_opencv_image(img, os.path.join(dir_pred_save, "%s_vector_results.json" % fname_base), dir_coco_test)

        if masked_image is None:
            continue

        path_save = os.path.join(dir_save, name)
        cv2.imwrite(path_save, masked_image)
