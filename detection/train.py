"""
detection/train.py

Train MobileViTv2 + SSDLite on a COCO-format dataset.

Example:
    python train.py \
        --train_images ./data/train2017 --train_ann ./data/annotations/instances_train.json \
        --val_images   ./data/val2017   --val_ann   ./data/annotations/instances_val.json \
        --epochs 50 --batch_size 16 --width w0_5 \
        --pretrained_backbone weights/mobilevit-w0.5.pth
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader

from model.ssd_mobilevit_v2 import create_ssd_model
from data.coco_dataset import CocoDetectionDataset, collate_fn
from engine import train_one_epoch, evaluate_coco

parser = argparse.ArgumentParser(description="Train MobileViTv2-SSDLite on a COCO-format dataset")
parser.add_argument("--train_images", type=str, required=True, help="Path to training images folder")
parser.add_argument("--train_ann", type=str, required=True, help="Path to COCO train annotations JSON")
parser.add_argument("--val_images", type=str, required=True, help="Path to validation images folder")
parser.add_argument("--val_ann", type=str, required=True, help="Path to COCO val annotations JSON")
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--image_size", type=int, default=320)
parser.add_argument("--width", type=str, default="w0_5",
                     choices=["w0_5", "w0_75", "w1_0", "w1_25", "w1_5", "w1_75", "w2_0"])
parser.add_argument("--pretrained_backbone", type=str, default="weights/mobilevit-w0.5.pth",
                     help="ImageNet-pretrained MobileViTv2 backbone checkpoint (converted format)")
parser.add_argument("--eval_every", type=int, default=1, help="Run COCO mAP eval every N epochs")
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

EXP_NAME = f"mobilevitv2_ssdlite_{args.width}"
WEIGHT_DIR = os.path.join("weights", EXP_NAME)
os.makedirs(WEIGHT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

train_dataset = CocoDetectionDataset(args.train_images, args.train_ann, train=True)
val_dataset = CocoDetectionDataset(args.val_images, args.val_ann, train=False)

# The val set must use the exact same category-id <-> label mapping as train,
# in case some categories happen to be absent from one annotation file.
val_dataset.catid2label = train_dataset.catid2label
val_dataset.label2catid = train_dataset.label2catid

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate_fn)

print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)} | "
      f"Foreground classes: {train_dataset.num_fg_classes}")

model = create_ssd_model(
    num_fg_classes=train_dataset.num_fg_classes,
    device=DEVICE,
    width=args.width,
    pretrained_backbone_path=args.pretrained_backbone,
    image_size=args.image_size,
)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[max(1, int(args.epochs * 0.7)), max(2, int(args.epochs * 0.9))],
    gamma=0.1,
)

best_map = 0.0
for epoch in range(1, args.epochs + 1):
    train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
    scheduler.step()
    print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}")

    if epoch % args.eval_every == 0 or epoch == args.epochs:
        coco_eval = evaluate_coco(model, val_loader, DEVICE, val_dataset.coco, val_dataset.label2catid)
        if coco_eval is not None:
            current_map = coco_eval.stats[0]  # mAP @ IoU=[0.50:0.95]
            if current_map > best_map:
                best_map = current_map
                torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "best_model.pt"))
                print(f"Saved best_model.pt (mAP@[.5:.95]: {best_map:.4f})")

    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "last_model.pt"))

print(f"Training complete. Best mAP@[.5:.95]: {best_map:.4f}")
