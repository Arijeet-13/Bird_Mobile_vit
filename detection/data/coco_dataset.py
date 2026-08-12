"""
detection/data/coco_dataset.py

Standard-COCO-layout dataset (images/ folder + instances_*.json annotations),
producing (image, target) pairs compatible with torchvision's detection models
(SSD, Faster R-CNN, etc):

    image  : FloatTensor[C, H, W], values in [0, 1]
    target : {"boxes": FloatTensor[N, 4] (xyxy, pixel coords),
              "labels": Int64Tensor[N]  (1..num_fg_classes; 0 = background),
              "image_id": int}

Requires torchvision >= 0.15 (tv_tensors / transforms v2) and pycocotools.
"""

import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from torchvision import tv_tensors


class CocoDetectionDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, train=True):
        super().__init__(img_folder, ann_file)

        # COCO category ids are not contiguous (gaps in the 1..90 range) -
        # remap to contiguous labels 1..num_fg_classes (0 reserved for background).
        cat_ids = sorted(self.coco.getCatIds())
        self.catid2label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        self.label2catid = {v: k for k, v in self.catid2label.items()}
        self.num_fg_classes = len(cat_ids)

        if train:
            self.tfms = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.tfms = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])

    def __getitem__(self, index):
        img, anns = super().__getitem__(index)
        image_id = self.ids[index]

        boxes, labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.catid2label[ann["category_id"]])

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

        canvas_size = (img.height, img.width)
        boxes_tv = tv_tensors.BoundingBoxes(boxes_t, format="XYXY", canvas_size=canvas_size)

        target = {"boxes": boxes_tv, "labels": labels_t, "image_id": image_id}
        img, target = self.tfms(img, target)
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        return img, target


def collate_fn(batch):
    """Detection batches have a variable number of boxes per image, so we
    return a tuple of individual samples instead of stacking into one tensor."""
    return tuple(zip(*batch))
