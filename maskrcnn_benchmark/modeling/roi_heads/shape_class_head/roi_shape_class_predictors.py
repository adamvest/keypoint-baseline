from torch import nn


class ShapeClassMaskRCNNC4Predictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(ShapeClassMaskRCNNC4Predictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = (res2_out_channels * stage2_relative_factor) + res2_out_channels

        num_shape_types = config.MODEL.ROI_SHAPE_CLASS_HEAD.NUM_POLYGON_TYPES

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2)
        self.shape_score = nn.Linear(num_inputs, num_shape_types)

        nn.init.normal_(self.shape_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.shape_score.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        shape_class_logits = self.shape_score(x)
        return shape_class_logits


_ROI_SHAPE_CLASS_PREDICTOR = {
    "ShapeClassMaskRCNNC4Predictor": ShapeClassMaskRCNNC4Predictor
}


def make_roi_shape_class_predictor(cfg):
    func = _ROI_SHAPE_CLASS_PREDICTOR[cfg.MODEL.ROI_SHAPE_CLASS_HEAD.PREDICTOR]
    return func(cfg)
