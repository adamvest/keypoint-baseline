# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import cv2
import json
import torch
from torchvision import transforms as T
import numpy as np
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0], [255, 0, 128],
                        [128, 255, 0], [128, 0, 255], [128, 128, 0], [128, 0, 128], [0, 128, 128], [255, 100, 100],
                        [100, 255, 100], [100, 100, 255]]
color_table = np.array([[255.0, 255.0, 255.0], [255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0], [255.0, 128.0, 0.0], [255.0, 0.0, 128.0],
                                [128.0, 255.0, 0.0], [128.0, 0.0, 255.0], [128.0, 128.0, 0.0], [128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [255.0, 100.0, 100.0],
                                [100.0, 255.0, 100.0], [100.0, 100.0, 255.0], [0.0, 255.0, 255.0], [255.0, 0.0, 255.0], [255.0, 255.0, 0.0],[255.0, 155.0, 0.0],
                                [155.0, 255.0, 0.0],[255.0, 255.0, 155.0],[155.0, 155.0, 0.0],[155.0, 55.0, 0.0], [0.0, 0.0, 0.0], [200.0, 200.0, 200.0], [200.0, 0.0, 200.0]])

KEEP_MAP = {
    0: torch.tensor([0, 3, 4], dtype=torch.int32),
    1: torch.tensor([1, 2, 3, 4], dtype=torch.int32),
    2: torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
}


class COCODemo(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "Wall",
        "Garage",
        "Window",
        "Door"
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR

        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image, save_path=None, root=None):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """

        predictions = self.compute_prediction(image)

        if predictions is None:
            return None

        top_predictions, idxes = self.select_top_predictions(predictions)

        if save_path is not None:
            self.save_predictions(top_predictions, save_path, root)

        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        # result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            result = self.overlay_keypoints(result, top_predictions)

        result = self.overlay_class_names(result, top_predictions)
        return result

    def save_predictions(self, preds, save_path, root):
        fname_base = save_path.split("/")[-1].split("_")[0] + "_" + save_path.split("/")[-1].split("_")[1]
        boxes = preds.bbox.numpy().tolist()
        shapes = preds.get_field("shape").numpy().tolist()
        labels = preds.get_field("labels").numpy().tolist()
        scores = preds.get_field("scores").numpy().tolist()
        keypoints = preds.get_field("keypoints").keypoints.numpy()

        res = {"boxes": [], "vector_masks": [], "scores": [], "labels": [], "keypoints": []}

        for region_box, region_label, region_score, region_shape, region_kps in zip(boxes, labels, scores, shapes, keypoints):
            res["scores"].append(region_score)
            res["labels"].append(region_label)

            keep = KEEP_MAP[region_shape].numpy()
            region_kps = region_kps[keep, :2].tolist()

            if region_shape == 1:
                x_min, y_min, x_max, y_max = region_box
                bbox_points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                res["vector_masks"].append(bbox_points)
            else:
                res["vector_masks"].append(region_kps)

            res["keypoints"].append(region_kps)

        res["vector_masks"] = self.warp_points_to_original_space(res["vector_masks"], fname_base, root)
        res["keypoints"] = self.warp_points_to_original_space(res["keypoints"], fname_base, root)
        res["boxes"] = self.compute_boxes_from_points(res["vector_masks"])
        res["kp_boxes"] = self.compute_boxes_from_points(res["keypoints"])

        f = open(save_path, "w")
        json.dump(res, f)
        f.close()

    def compute_boxes_from_points(self, vector_masks):
        boxes = []

        for points in vector_masks:
            x = [pt[0] for pt in points]
            y = [pt[1] for pt in points]
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            w, h = x_max - x_min, y_max - y_min
            boxes.append([x_min, y_min, w, h])

        return boxes

    def warp_points_to_original_space(self, vector_masks, fname_base, root):
        hom_mat = np.load(os.path.join(root, "hom_mats", "%s.npy" % fname_base))
        hom_mat_inv = np.linalg.inv(hom_mat)
        warped_vector_masks = []

        for points in vector_masks:
            warped_points = []

            for pt in points:
                hom_warped_pt = np.dot(hom_mat_inv, [pt[0], pt[1], 1]).tolist()

                if hom_warped_pt[2] == 0.0:
                    hom_warped_pt[2] += 1e-6

                warped_pt = [hom_warped_pt[0] / hom_warped_pt[2], hom_warped_pt[1] / hom_warped_pt[2]]
                warped_points.append(warped_pt)

            warped_vector_masks.append(warped_points)

        return warped_vector_masks

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image

        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        # compute predictions
        with torch.no_grad():
            try:
                predictions = self.model(image_list)
            except RuntimeError as e:
                print("No regions kept after NMS! Skipping image")
                return None

        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]
        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)

        return predictions[idx], idx

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """


        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for ind, box in enumerate(boxes):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color_table[ind]), 5
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        shape_labels = predictions.get_field("shape")

        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()

        for i, region in enumerate(kps):
            image = vis_keypoints(i, image, region.transpose((1, 0)), shape_labels[i])

        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image



import numpy as np
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.keypoint import BuildingKeypoints


def vis_keypoints(ind, img, kps, shape_label, kp_thresh=0.0, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """

    dataset_keypoints = BuildingKeypoints.NAMES
    kp_lines = BuildingKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    keep = KEEP_MAP[shape_label.item()]
    kps = kps[:, keep]

    # Draw the keypoints.
    for l in range(kps.shape[1]):
        p1 = kps[0, l], kps[1, l]
        if kps[2, l] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=10, color=color_table[ind], thickness=-1, lineType=cv2.LINE_AA)


    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
