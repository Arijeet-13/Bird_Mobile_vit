"""
detection/eval_coco.py

Evaluate a trained MobileViTv2-SSDLite checkpoint against a COCO-format val set.

Example:
    python eval_coco.py \
        --val_images ./data/val2017 --val_ann ./data/annotations/instances_val.json \
        --checkpoint weights/mobilevitv2_ssdlite_w0_5/best_model.pt --width w0_5
"""

import argparse

import torch
from torch.utils.data import DataLoader

from model.ssd_mobilevit_v2 import create_ssd_model
from data.coco_dataset import CocoDetectionDataset, collate_fn
from engine import evaluate_coco

parser = argparse.ArgumentParser(description="Evaluate a MobileViTv2-SSDLite checkpoint's COCO mAP")
parser.add_argument("--val_images", type=str, required=True)
parser.add_argument("--val_ann", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a full detector .pt checkpoint")
parser.add_argument("--width", type=str, default="w0_5",
                     choices=["w0_5", "w0_75", "w1_0", "w1_25", "w1_5", "w1_75", "w2_0"])
parser.add_argument("--image_size", type=int, default=320)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_dataset = CocoDetectionDataset(args.val_images, args.val_ann, train=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate_fn)

# pretrained_backbone_path=None: we're loading full detector weights below,
# so the ImageNet-only backbone init is irrelevant here.
model = create_ssd_model(
    num_fg_classes=val_dataset.num_fg_classes,
    device=DEVICE,
    width=args.width,
    pretrained_backbone_path=None,
    image_size=args.image_size,
)
model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
print(f"Loaded checkpoint from '{args.checkpoint}'")

evaluate_coco(model, val_loader, DEVICE, val_dataset.coco, val_dataset.label2catid)
