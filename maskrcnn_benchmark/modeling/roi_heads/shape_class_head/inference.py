import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder


class ShapeClassPostProcessor(nn.Module):
    """
    Post processor to convert raw shape logits into scores and labels
    and add them to the result boxlist
    """

    def __init__(self):
        """
        """
        super(ShapeClassPostProcessor, self).__init__()

    def forward(self, shape_class_logits, boxes):
        """
        Arguments:
            shape_class_logits (tuple[tensor, tensor])
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra 'shape' and 'shape_scores' fields
        """
        shape_class_probs = F.softmax(shape_class_logits, -1)
        shape_labels = torch.argmax(shape_class_probs, dim=1)
        boxes_per_image = [len(box) for box in boxes]
        shape_class_probs = shape_class_probs.split(boxes_per_image, dim=0)
        shape_labels = shape_labels.split(boxes_per_image, dim=0)
        results = []

        for shape_class_prob, shape_label, box in zip(shape_class_probs, shape_labels, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")

            for field in box.fields():
                bbox.add_field(field, box.get_field(field))

            bbox.add_field("shape_scores", shape_class_prob)
            bbox.add_field("shape", shape_label)
            results.append(bbox)

        return results

def make_roi_shape_class_post_processor(cfg):
    shape_class_post_processor = ShapeClassPostProcessor()
    return shape_class_post_processor
