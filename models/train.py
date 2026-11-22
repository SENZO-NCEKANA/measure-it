"""Item 3: fine-tune Mask R-CNN (pretrained on COCO) on my card photos.

Run from the repo root:  python models/train.py
Reads dataset/labels/coco.json, dataset/images/ and dataset/splits/.
Keeps the checkpoint with the best val mask mAP in models/best.pth.
The test split is never opened here.
"""
import contextlib
import io
import json
import os
import random
import time

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_Weights,
                                          maskrcnn_resnet50_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# %% Settings, chosen before training and not tuned.
SEED = 0
EPOCHS = 5
LEARNING_RATE = 0.005
SHORT_SIDE = 800       # photos are resized so the short side is 800 px
TRAINABLE_LAYERS = 1   # backbone stages that learn; the rest stay frozen
LABELS = "dataset/labels/coco.json"
IMAGES = "dataset/images"


def load_split(split):
    """Load one split as a list of (name, RGB image, [polygons])."""
    with open(LABELS, encoding="utf-8") as fh:
        coco = json.load(fh)
    polygons = {}
    for ann in coco["annotations"]:
        points = np.array(ann["segmentation"][0], dtype=np.float32)
        polygons.setdefault(ann["image_id"], []).append(points.reshape(-1, 2))
    by_name = {}
    for img in coco["images"]:
        # Label Studio saves names like ".../eb82bc39-card_001.jpg".
        name = os.path.basename(img["file_name"]).split("-", 1)[1]
        by_name[name] = polygons.get(img["id"], [])
    with open(f"dataset/splits/{split}.txt", encoding="utf-8") as fh:
        names = [line.strip() for line in fh if line.strip()]
    samples = []
    for name in names:
        bgr = cv2.imread(os.path.join(IMAGES, name))
        scale = SHORT_SIDE / min(bgr.shape[:2])
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        samples.append((name, rgb, [p * scale for p in by_name[name]]))
    return samples


def polygon_masks(image, polygons):
    """Fill each polygon into its own 0/1 mask the size of the image."""
    masks = []
    for polygon in polygons:
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
        masks.append(mask)
    return masks


def augment(image, masks, rng):
    """Random flips and a 90-degree turn: my card always lay one way."""
    if rng.random() < 0.5:
        image, masks = image[:, ::-1], [m[:, ::-1] for m in masks]
    if rng.random() < 0.5:
        image, masks = image[::-1], [m[::-1] for m in masks]
    if rng.random() < 0.5:
        image, masks = np.rot90(image), [np.rot90(m) for m in masks]
    return (np.ascontiguousarray(image),
            [np.ascontiguousarray(m) for m in masks])


def to_tensors(image, masks):
    """Image as a 0-1 float tensor; target with boxes taken from the masks."""
    boxes = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.ones(len(masks), dtype=torch.int64),  # 1 = card
        "masks": torch.from_numpy(np.stack(masks)),
    }
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255, target


def build_model(pretrained=True):
    """Mask R-CNN with its two output heads replaced: background + card."""
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights, weights_backbone=None,
        trainable_backbone_layers=TRAINABLE_LAYERS,
        min_size=SHORT_SIDE, max_size=1333)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, 2)
    return model


@torch.no_grad()
def predict(model, samples):
    """Run the model on each sample; returns one output dict per sample."""
    model.eval()
    outputs = []
    for _, image, _ in samples:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        outputs.append(model([tensor])[0])
    return outputs


def coco_mask_map(samples, outputs, quiet=True):
    """Mask mAP@0.5:0.95 and mAP@0.5 with pycocotools."""
    truth = {"images": [], "annotations": [],
             "categories": [{"id": 1, "name": "card"}]}
    found = []
    for i, ((name, image, polygons), out) in enumerate(zip(samples, outputs)):
        height, width = image.shape[:2]
        truth["images"].append(
            {"id": i, "width": width, "height": height, "file_name": name})
        for polygon in polygons:
            x, y = polygon[:, 0], polygon[:, 1]
            truth["annotations"].append({
                "id": len(truth["annotations"]) + 1, "image_id": i,
                "category_id": 1, "iscrowd": 0,
                "segmentation": [polygon.ravel().tolist()],
                "area": float(cv2.contourArea(polygon)),
                "bbox": [float(x.min()), float(y.min()),
                         float(x.max() - x.min()), float(y.max() - y.min())],
            })
        for mask, score in zip(out["masks"], out["scores"]):
            binary = (mask[0] > 0.5).numpy().astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(binary))
            rle["counts"] = rle["counts"].decode("ascii")
            found.append({"image_id": i, "category_id": 1,
                          "segmentation": rle, "score": float(score)})
    if not found:
        return 0.0, 0.0
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed if quiet else None):
        coco_truth = COCO()
        coco_truth.dataset = truth
        coco_truth.createIndex()
        result = COCOeval(coco_truth, coco_truth.loadRes(found), "segm")
        result.evaluate()
        result.accumulate()
        result.summarize()
    return float(result.stats[0]), float(result.stats[1])


def main():
    # %% Same seed every run, so the result can be repeated.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    train, val = load_split("train"), load_split("val")
    print("train photos:", len(train), " val photos:", len(val))

    model = build_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=LEARNING_RATE, momentum=0.9,
                                weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4,
                                                gamma=0.1)

    # %% Train, scoring val after every epoch; keep the best by val mAP.
    best = -1.0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        for index in order:
            _, image, polygons = train[index]
            image, masks = augment(image, polygon_masks(image, polygons), rng)
            tensor, target = to_tensors(image, masks)
            loss = sum(model([tensor], [target]).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        map_all, map_50 = coco_mask_map(val, predict(model, val))
        print(f"epoch {epoch}: train loss {np.mean(losses):.3f}, "
              f"val mask mAP@0.5:0.95 {map_all:.3f}, mAP@0.5 {map_50:.3f}, "
              f"{(time.time() - started) / 60:.1f} min so far", flush=True)
        if map_all > best:
            best = map_all
            torch.save(model.state_dict(), "models/best.pth")
            print("  saved models/best.pth (best val so far)", flush=True)
    print(f"Done in {(time.time() - started) / 60:.1f} min. "
          f"Best val mask mAP@0.5:0.95: {best:.3f}")


if __name__ == "__main__":
    main()
