"""
detection/model/backbone.py

Wraps the classification MobileViTv2 backbone (conv_1 -> layer_1..layer_5) as a
multi-scale feature extractor for SSDLite-style object detection, mirroring the
MobileViT(v2) + SSDLite pairing used in the original papers.

Feature maps returned (5 total, keyed '0'..'4'):
    '0': output of layer_3   (stride ~8)
    '1': output of layer_4   (stride ~16)
    '2': output of layer_5   (stride ~32)
    '3': extra block 1       (stride ~64)
    '4': extra block 2       (stride ~128)
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn

from mobilevitv2.mobilevit_v2 import MobileViTv2
from mobilevitv2.mobilevit_v2_cfg import *  # noqa: F401,F403  (brings get_mobilevit_v2_* into scope)

_WIDTH_CFG_FN = {
    "w0_5": "get_mobilevit_v2_w0_5",
    "w0_75": "get_mobilevit_v2_w0_75",
    "w1_0": "get_mobilevit_v2_w1_0",
    "w1_25": "get_mobilevit_v2_w1_25",
    "w1_5": "get_mobilevit_v2_w1_5",
    "w1_75": "get_mobilevit_v2_w1_75",
    "w2_0": "get_mobilevit_v2_w2_0",
}


def _extra_block(in_channels, out_channels):
    """Depthwise-separable 'SSDLite-style' downsampling block (stride 2)."""
    mid_channels = max(out_channels // 2, 8)
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1,
                  groups=mid_channels, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class MobileViTv2SSDBackbone(nn.Module):
    """
    Multi-scale feature extractor built from the MobileViTv2 classification
    backbone, for use as a `backbone` inside torchvision's SSD detector.
    """

    def __init__(self, width="w0_5", pretrained_backbone_path=None, image_size=320,
                 extra_channels=(512, 256)):
        super().__init__()
        if width not in _WIDTH_CFG_FN:
            raise ValueError(f"Unknown MobileViTv2 width '{width}'. Choose from {list(_WIDTH_CFG_FN)}")

        cfg = eval(_WIDTH_CFG_FN[width])()  # noqa: S307 - mirrors the pattern used by the original repo
        self.body = MobileViTv2(cfg=cfg, classifier_num=1000)

        if pretrained_backbone_path and os.path.isfile(pretrained_backbone_path):
            state_dict = torch.load(pretrained_backbone_path, map_location="cpu")
            missing, unexpected = self.body.load_state_dict(state_dict, strict=False)
            print(f"[MobileViTv2SSDBackbone] Loaded pretrained backbone from '{pretrained_backbone_path}'")
            if missing:
                print(f"[MobileViTv2SSDBackbone] Missing keys (expected: classifier_layer.*): {missing}")
            if unexpected:
                print(f"[MobileViTv2SSDBackbone] Unexpected keys: {unexpected}")
        else:
            print(f"[MobileViTv2SSDBackbone] WARNING: '{pretrained_backbone_path}' not found — "
                  f"backbone will train from random initialization.")

        # The ImageNet classification head is not used for detection.
        del self.body.classifier_layer

        # Probe feature-map channel counts with a dummy forward pass (robust to
        # cfg internals we don't have visibility into).
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            f3, f4, f5 = self._forward_body(dummy)
        c3, c4, c5 = f3.shape[1], f4.shape[1], f5.shape[1]

        e1_channels, e2_channels = extra_channels
        self.extra1 = _extra_block(c5, e1_channels)
        self.extra2 = _extra_block(e1_channels, e2_channels)

        # Exposed for reference / building detection heads externally if needed.
        self.out_channels = [c3, c4, c5, e1_channels, e2_channels]

    def _forward_body(self, x):
        x = self.body.conv_1(x)
        x = self.body.layer_1(x)
        x = self.body.layer_2(x)
        f3 = self.body.layer_3(x)
        f4 = self.body.layer_4(f3)
        f5 = self.body.layer_5(f4)
        return f3, f4, f5

    def forward(self, x):
        f3, f4, f5 = self._forward_body(x)
        f6 = self.extra1(f5)
        f7 = self.extra2(f6)
        return OrderedDict([("0", f3), ("1", f4), ("2", f5), ("3", f6), ("4", f7)])
