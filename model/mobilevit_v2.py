"""
model/mobilevit_v2.py

Drop-in MobileViTv2 replacement for model/mobilevit.py, exposing the same
three-function interface used by the training pipeline (main.py):

    create_model(n_classes, device)
    freeze_backbone(model)
    unfreeze_finetune_layers(model)

Notes on the v1 -> v2 mapping
------------------------------
- The original MobileViTv2 class (from the mobilevitv2 package) names its
  output layer `classifier_layer`. We replace it with a freshly-initialized
  `nn.Linear(in_features, n_classes)` and expose it as `model.head`, so the
  rest of the training script (which does `model.head.parameters()` and
  filters params with `"head" in name`) works without any other changes.
- v1's fine-tuning step unfroze the last transformer block, matched via
  `"blocks.4" in name`. In this MobileViTv2 implementation the analogous
  last stage is `layer_5` (the final MobileViTBlockV2 stage before global
  pooling), so FINETUNE_STAGE = "layer_5" plays the same role.
"""

import os
import torch
import torch.nn as nn

from mobilevitv2.mobilevit_v2 import MobileViTv2
from mobilevitv2.mobilevit_v2_cfg import *  # noqa: F401,F403  (brings get_mobilevit_v2_* into scope)

# ----------------------------- CONFIG -----------------------------
# Width variant to use. Must match one of the get_mobilevit_v2_<width>()
# config functions in mobilevitv2.mobilevit_v2_cfg, and must match the
# checkpoint you point PRETRAINED_WEIGHTS at.
MODEL_WIDTH = "w0_5"

# Path to the ImageNet-pretrained MobileViTv2 backbone checkpoint
# (produced by mobilevitv2/mobilevit_v2.py's run_model() converter).
PRETRAINED_WEIGHTS = "weights/mobilevit-w0.5.pth"

# Last backbone stage name, used for the "unfreeze last stage + head" fine-tune step.
FINETUNE_STAGE = "layer_5"

_WIDTH_CFG_FN = {
    "w0_5": "get_mobilevit_v2_w0_5",
    "w0_75": "get_mobilevit_v2_w0_75",
    "w1_0": "get_mobilevit_v2_w1_0",
    "w1_25": "get_mobilevit_v2_w1_25",
    "w1_5": "get_mobilevit_v2_w1_5",
    "w1_75": "get_mobilevit_v2_w1_75",
    "w2_0": "get_mobilevit_v2_w2_0",
}


def _build_forward():
    """Forward pass identical to MobileViTv2.forward, but reading `self.head`
    instead of the original `self.classifier_layer`."""

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = torch.mean(x, dim=-1).squeeze(dim=-1)
        x = self.head(x)
        return x

    return forward


def create_model(n_classes, device, width=MODEL_WIDTH, pretrained_path=PRETRAINED_WEIGHTS):
    """
    Build a MobileViTv2 classifier for `n_classes`, initialize it from an
    ImageNet-pretrained backbone checkpoint, and attach a fresh classification
    head (`model.head`) sized for the target task.
    """
    if width not in _WIDTH_CFG_FN:
        raise ValueError(f"Unknown MobileViTv2 width '{width}'. Choose from {list(_WIDTH_CFG_FN)}")

    cfg = eval(_WIDTH_CFG_FN[width])()  # noqa: S307 - mirrors the pattern used by the original repo

    # classifier_num=1000 keeps the checkpoint's classifier_layer shape aligned
    # during loading; we discard/replace it with our own head immediately after.
    model = MobileViTv2(cfg=cfg, classifier_num=1000)

    if pretrained_path and os.path.isfile(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[create_model] Loaded pretrained backbone from '{pretrained_path}'")
        if missing:
            print(f"[create_model] Missing keys (expected: classifier_layer.*): {missing}")
        if unexpected:
            print(f"[create_model] Unexpected keys: {unexpected}")
    else:
        print(f"[create_model] WARNING: '{pretrained_path}' not found — training from random init.")

    # Swap classifier_layer -> head, resized for this task's n_classes.
    in_features = model.classifier_layer.in_features
    del model.classifier_layer
    model.head = nn.Linear(in_features, n_classes)

    # Bind the patched forward (uses self.head instead of self.classifier_layer).
    model.forward = _build_forward().__get__(model, MobileViTv2)

    model = model.to(device)
    return model


def freeze_backbone(model):
    """Freeze every parameter except the classification head."""
    for name, param in model.named_parameters():
        param.requires_grad = "head" in name
    return model


def unfreeze_finetune_layers(model, finetune_stage=FINETUNE_STAGE):
    """
    Unfreeze the classification head plus the final backbone stage
    (`layer_5` by default) for fine-tuning; everything else stays frozen.
    """
    for name, param in model.named_parameters():
        param.requires_grad = (finetune_stage in name) or ("head" in name)
    return model
