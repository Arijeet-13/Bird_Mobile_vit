"""
detection/engine.py

train_one_epoch(): standard torchvision-detection-style training step
                    (model(images, targets) returns a dict of losses in train mode).
evaluate_coco():    runs inference on a val loader and computes COCO mAP via pycocotools
                    (model(images) returns a list of {"boxes","labels","scores"} dicts in eval mode).
"""

import math
import sys

import torch
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    total_loss = 0.0

    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} [train]")):
        images = [img.to(device) for img in images]
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
                   for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value} (non-finite) — stopping. loss_dict={loss_dict}")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += loss_value

        if (i + 1) % print_freq == 0:
            loss_str = ", ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
            print(f"Epoch {epoch} [{i + 1}/{len(data_loader)}] total={loss_value:.4f} ({loss_str})")

    return total_loss / len(data_loader)


@torch.no_grad()
def evaluate_coco(model, data_loader, device, coco_gt, label2catid):
    """
    Runs inference on `data_loader`, builds COCO-format detection results, and
    evaluates them against `coco_gt` (a pycocotools.coco.COCO object) using
    pycocotools' COCOeval. Returns the COCOeval object (stats[0] = mAP@[.5:.95],
    stats[1] = mAP@0.5), or None if no detections were produced.
    """
    from pycocotools.cocoeval import COCOeval

    model.eval()
    results = []

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = target["image_id"]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                results.append({
                    "image_id": int(image_id),
                    "category_id": int(label2catid[int(label)]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })

    if len(results) == 0:
        print("No detections produced across the val set; skipping COCO eval.")
        return None

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
