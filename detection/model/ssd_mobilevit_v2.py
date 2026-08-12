"""
detection/model/ssd_mobilevit_v2.py

Builds a full SSDLite object detector using the MobileViTv2 backbone, reusing
torchvision's tested SSD implementation (box matching, MultiBox loss, NMS,
postprocessing, and the fixed-size resize + normalize transform) so only the
backbone integration is custom.

Requires torchvision >= 0.13 (SSDLiteHead / DefaultBoxGenerator availability).
"""

import torch
from torchvision.models.detection import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssdlite import SSDLiteHead

from .backbone import MobileViTv2SSDBackbone

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# One aspect-ratio list per feature-map level (5 levels: layer_3, layer_4, layer_5, extra1, extra2).
ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2]]


def create_ssd_model(num_fg_classes, device, width="w0_5", pretrained_backbone_path=None,
                      image_size=320, extra_channels=(512, 256),
                      score_thresh=0.01, nms_thresh=0.45, detections_per_img=200):
    """
    num_fg_classes: number of *foreground* object categories in your dataset
                     (background class is added automatically, as class 0).
    """
    backbone = MobileViTv2SSDBackbone(
        width=width,
        pretrained_backbone_path=pretrained_backbone_path,
        image_size=image_size,
        extra_channels=extra_channels,
    )

    anchor_generator = DefaultBoxGenerator(aspect_ratios=ASPECT_RATIOS)
    num_anchors = anchor_generator.num_anchors_per_location()

    num_classes = num_fg_classes + 1  # +1 for background

    head = SSDLiteHead(
        in_channels=backbone.out_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=torch.nn.BatchNorm2d,
    )

    model = SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(image_size, image_size),
        num_classes=num_classes,
        head=head,
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
    )

    return model.to(device)
