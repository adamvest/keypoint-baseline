# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import BuildingKeypoints


min_keypoints_per_image = 10

def get_vertice(segmen):
    segmen = [segmen[i] for i in range(len(segmen) - 2)]
    if len(segmen) == 6:
        x = np.array([segmen[i] for i in range(0, len(segmen), 2)])
        y = np.array([segmen[i] for i in range(1, len(segmen), 2)])
        y_sort = np.argsort(y)
        vertices = [x[y_sort[0]], y[y_sort[0]], 2]
        if x[y_sort[1]] < x[y_sort[2]]:
            t_left = [x[y_sort[1]], y[y_sort[1]], 2]
            t_right = [x[y_sort[2]], y[y_sort[2]], 2]
        else:
            t_left = [x[y_sort[2]], y[y_sort[2]], 2]
            t_right = [x[y_sort[1]], y[y_sort[1]], 2]
        vertices.extend(t_left)
        vertices.extend(t_right)
        vertices.extend([0, 0, 0, 0, 0, 0])
    elif len(segmen) == 8:
        vertices = [0, 0, 0]
        vertices.extend(sort_quad(segmen))
    elif len(segmen) == 10:
        y = np.array([segmen[i + 1] for i in range(0, len(segmen), 2)])
        y_min = np.argmin(y)
        vertices = [segmen[2 * y_min], segmen[2 * y_min + 1], 2]
        segmen.pop(2 * y_min)
        segmen.pop(2 * y_min)
        vertices.extend(sort_quad(segmen))
    return vertices

def sort_quad(segmen):
    x = np.array([segmen[i] for i in range(0, len(segmen), 2)])
    y = np.array([segmen[i] for i in range(1, len(segmen), 2)])
    x_sort = np.argsort(x)
    if x[x_sort[1]] == x[x_sort[2]]:
        if y[x_sort[1]] > y[x_sort[2]]:
            x[x_sort[1]], x[x_sort[2]] = x[x_sort[2]], x[x_sort[1]]
    if y[x_sort[0]] < y[x_sort[1]]:
        t_left = [x[x_sort[0]], y[x_sort[0]]]
        b_left = [x[x_sort[1]], y[x_sort[1]]]
    else:
        t_left = [x[x_sort[1]], y[x_sort[1]]]
        b_left = [x[x_sort[0]], y[x_sort[0]]]
    if y[x_sort[2]] < y[x_sort[3]]:
        t_right = [x[x_sort[2]], y[x_sort[2]]]
        b_right = [x[x_sort[3]], y[x_sort[3]]]
    else:
        t_right = [x[x_sort[3]], y[x_sort[3]]]
        b_right = [x[x_sort[2]], y[x_sort[2]]]
    return [t_left[0], t_left[1], 2, t_right[0], t_right[1], 2,  b_left[0], b_left[1], 2,  b_right[0], b_right[1], 2]
def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        # self.ids = self.ids[:4]

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

        self.shape_len_to_shape_type_map = {6: 0, 8: 1, 10: 2}

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]

        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        polygons = masks
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        keypoints = []
        for seg in polygons:
            seg = seg[0]
            res = get_vertice(seg)
            keypoints.append(res)

        keypoints = BuildingKeypoints(keypoints, img.size)

        target.add_field("keypoints", keypoints)

        shape_lens = [len(polygon[0][:-2]) for polygon in polygons]
        shape_labels = [self.shape_len_to_shape_type_map[l] for l in shape_lens]
        shape_labels = torch.tensor(shape_labels)
        target.add_field("shapes", shape_labels)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
