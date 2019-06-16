import torch
from torch.nn import functional as F
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


class ShapeClassMaskRCNNLossComputation(object):
    """
    Computes the shape classification loss
    """

    def __init__(self, proposal_matcher):
        """
        Arguments:
            proposal_matcher (Matcher): used to match proposed regions to GT regions
        """
        self.proposal_matcher = proposal_matcher

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(["labels", "shapes"])

        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels, shapes = [], []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(proposals_per_image,
                targets_per_image)
            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # mask scores are only computed on positive proposals
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)
            shapes_per_image = matched_targets.get_field("shapes")
            shapes_per_image = shapes_per_image[positive_inds]
            positive_proposals = proposals_per_image[positive_inds]
            labels.append(labels_per_image)
            shapes.append(shapes_per_image)

        return labels, shapes

    def __call__(self, proposals, shape_class_logits, targets):
        """
        Computes the shape classification loss

        Arguments:
            proposals (list[BoxList])
            shape_class_logits (list[Tensor])
            targets (list[BoxList])

        Returns:
            loss_shape_class (Tensor): scalar tensor containing the loss
        """
        _, shape_class_targets = self.prepare_targets(proposals, targets)
        shape_class_targets = cat(shape_class_targets, dim=0)
        loss_shape_class = F.cross_entropy(shape_class_logits, shape_class_targets)
        return loss_shape_class


def make_roi_shape_class_loss_evaluator(cfg):
    matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)

    loss_evaluator = ShapeClassMaskRCNNLossComputation(matcher)
    return loss_evaluator
