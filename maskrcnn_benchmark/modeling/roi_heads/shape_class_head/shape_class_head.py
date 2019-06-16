import torch
from torch import nn
from maskrcnn_benchmark.structures.bounding_box import BoxList
from .roi_shape_class_feature_extractors import make_roi_shape_class_feature_extractor
from .roi_shape_class_predictors import make_roi_shape_class_predictor
from .inference import make_roi_shape_class_post_processor
from .loss import make_roi_shape_class_loss_evaluator


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIShapeClassHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIShapeClassHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_shape_class_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_shape_class_predictor(cfg)
        self.post_processor = make_roi_shape_class_post_processor(cfg)
        self.loss_evaluator = make_roi_shape_class_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `shape` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        if self.training and self.cfg.MODEL.ROI_SHAPE_CLASS_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)

        shape_class_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(shape_class_logits, proposals)
            return x, result, {}

        loss_shape_class = self.loss_evaluator(proposals,shape_class_logits,
            targets)
        return x, all_proposals, dict(loss_shape_class=loss_shape_class)


def build_roi_shape_class_head(cfg, in_channels):
    return ROIShapeClassHead(cfg, in_channels)
