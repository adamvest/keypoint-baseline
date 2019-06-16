import os
import json
import copy
import shutil
import numpy as np
from collections import defaultdict


dir_save = "./demo/combined_rec_vector_segments"
preds_root = "./demo/all_rec_vector_segments"

if os.path.isdir(dir_save):
    shutil.rmtree(dir_save)

os.makedirs(dir_save)

fname_base_to_fnames_map = defaultdict(list)

for root, _, fnames in os.walk(preds_root):
    for fname in fnames:
        fname_base = fname.split("_")[0]
        fname_base_to_fnames_map[fname_base].append(os.path.join(root, fname))

for fname_base, fnames in fname_base_to_fnames_map.items():
    res = {"boxes": [], "vector_masks": [], "scores": [], "labels": [], "keypoints": [], "kp_boxes": []}

    for fname in fnames:
        f = open(fname)
        data = json.load(f)
        f.close()

        res["boxes"].extend(data["boxes"])
        res["vector_masks"].extend(data["vector_masks"])
        res["scores"].extend(data["scores"])
        res["labels"].extend(data["labels"])
        res["keypoints"].extend(data["keypoints"])
        res["kp_boxes"].extend(data["kp_boxes"])

    f = open(os.path.join(dir_save, "%s_vector_results.json" % fname_base), "w")
    json.dump(res, f)
    f.close()
