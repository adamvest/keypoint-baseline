from torch import nn
from torch.nn import functional as F
from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3


class ShapeClassMaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Feature extractor for shape classifier
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg: YACS configuration file for the Mask RCNN instance
        """
        super(ShapeClassMaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_SHAPE_CLASS_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_SHAPE_CLASS_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_SHAPE_CLASS_HEAD.POOLER_SAMPLING_RATIO
        input_size = in_channels
        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        self.pooler = Pooler(output_size=(resolution, resolution), scales=scales,
            sampling_ratio=sampling_ratio)

        next_feature = input_size
        self.blocks = []

        for layer_idx, layer_features in enumerate(layers):
            layer_name = "shape_class_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


_ROI_SHAPE_CLASS_FEATURE_EXTRACTORS = {
    "ResNet50Conv5ROIFeatureExtractor": ResNet50Conv5ROIFeatureExtractor,
    "ShapeClassMaskRCNNFPNFeatureExtractor": ShapeClassMaskRCNNFPNFeatureExtractor,
}


def make_roi_shape_class_feature_extractor(cfg, in_channels):
    func = _ROI_SHAPE_CLASS_FEATURE_EXTRACTORS[cfg.MODEL.ROI_SHAPE_CLASS_HEAD.FEATURE_EXTRACTOR]
    return func(cfg, in_channels)
