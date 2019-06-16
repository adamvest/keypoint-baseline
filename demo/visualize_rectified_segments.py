import os
import json
import shutil
import numpy as np
from matplotlib import pyplot as plt


def reorder_clockwise(points):
    points = np.array(points)
    top_idx = np.argmin(points[:, 1])

    if len(points) != 4:
        reordered_points = [points[top_idx].tolist()]
        points = np.delete(points, top_idx, axis=0)
    else:
        reordered_points = []

    if len(points) == 2: # handles triangle case
        if points[0, 0] != points[1, 0]:
            right_idx = np.argmax(points[:, 0])
            left_idx = np.argmin(points[:, 0])
        else:
            if reordered_points[0][0] < points[0, 0]:
                right_idx = np.argmin(points[:, 1])
                left_idx = np.argmax(points[:, 1])
            else:
                right_idx = np.argmax(points[:, 1])
                left_idx = np.argmin(points[:, 1])

        reordered_points.append(points[right_idx].tolist())
        reordered_points.append(points[left_idx].tolist())
    elif len(points) == 4: #handles quad and pentagon cases
        top_idxs = np.argsort(points[:, 1])[:2]
        bottom_idxs = np.argsort(points[:, 1])[::-1][:2]
        top_left_idx = top_idxs[0] if points[top_idxs[0], 0] < points[top_idxs[1], 0] else top_idxs[1]
        top_right_idx = top_idxs[0] if points[top_idxs[0], 0] >= points[top_idxs[1], 0] else top_idxs[1]
        bottom_left_idx = bottom_idxs[0] if points[bottom_idxs[0], 0] < points[bottom_idxs[1], 0] else bottom_idxs[1]
        bottom_right_idx = bottom_idxs[0] if points[bottom_idxs[0], 0] >= points[bottom_idxs[1], 0] else bottom_idxs[1]
        reordered_points.append(points[top_right_idx])
        reordered_points.append(points[bottom_right_idx])
        reordered_points.append(points[bottom_left_idx])
        reordered_points.append(points[top_left_idx])
    else:
        raise ValueError("Invalid number of points in polygon!")

    return np.array(reordered_points)


#path to original (unrectified images)
data_root = "../predict-vp-masks/maskrcnn_benchmark/data/datasets/merged/test"
dir_save = "./demo/rec_segments_visualizations"
preds_root = "./demo/combined_rec_vector_segments"

if os.path.isdir(dir_save):
    shutil.rmtree(dir_save)

os.makedirs(dir_save)

for root, _, fnames in os.walk(preds_root):
    for fname in fnames:
        fname_base = fname.split("_")[0]
        f = open(os.path.join(root, fname))
        data = json.load(f)
        f.close()
        vector_masks = data["vector_masks"]
        keypoints = data["keypoints"]
        boxes = data["boxes"]
        kp_boxes = data["kp_boxes"]
        img = plt.imread(os.path.join(data_root, "%s.jpg" % fname_base))
        fig, ax = plt.subplots(1)
        cmap = plt.get_cmap('jet')
        ax.imshow(img)
        # axes[1].imshow(img)

        for i, vector_mask in enumerate(vector_masks):
            points = reorder_clockwise(vector_mask)
            color = cmap(float(i) / len(vector_masks))

            for pt in points:
                ax.plot(pt[0], pt[1], 'o', c=color)

            x = [pt[0] for pt in points] + [points[0][0]]
            y = [pt[1] for pt in points] + [points[0][1]]
            ax.plot(x, y, linewidth=1, c=color)

        for i, bbox in enumerate(boxes):
            color = cmap(float(i) / len(boxes))
            x_min, y_min, w, h = bbox
            x_max, y_max = x_min + w, y_min + h
            bbox_points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            x = [pt[0] for pt in bbox_points] + [bbox_points[0][0]]
            y = [pt[1] for pt in bbox_points] + [bbox_points[0][1]]
            ax.plot(x, y, dashes=[5, 5], linewidth=.5, c=color)

        # for i, vector_mask in enumerate(keypoints):
        #     points = reorder_clockwise(vector_mask)
        #     color = cmap(float(i) / len(vector_masks))
        #
        #     for pt in points:
        #         axes[1].plot(pt[0], pt[1], 'o', c=color)
        #
        #     x = [pt[0] for pt in points] + [points[0][0]]
        #     y = [pt[1] for pt in points] + [points[0][1]]
        #     axes[1].plot(x, y, linewidth=1, c=color)
        #
        # for i, bbox in enumerate(kp_boxes):
        #     color = cmap(float(i) / len(boxes))
        #     x_min, y_min, w, h = bbox
        #     x_max, y_max = x_min + w, y_min + h
        #     bbox_points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        #     x = [pt[0] for pt in bbox_points] + [bbox_points[0][0]]
        #     y = [pt[1] for pt in bbox_points] + [bbox_points[0][1]]
        #     axes[1].plot(x, y, dashes=[5, 5], linewidth=.5, c=color)

        plt.savefig(os.path.join(dir_save, "%s_viz.jpg" % fname_base))
        plt.close()
