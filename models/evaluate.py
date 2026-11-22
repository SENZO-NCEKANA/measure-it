"""Item 3: score the best checkpoint on the test split, once.

Run from the repo root:  python models/evaluate.py
Prints the raw pycocotools output, then mask IoU, precision and recall.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.train import (build_model, coco_mask_map,  # noqa: E402
                          load_split, polygon_masks, predict)

WEIGHTS = "models/best.pth"
MIN_SCORE = 0.5   # a detection counts from this score
MIN_IOU = 0.5     # a detection is correct from this overlap with my mask


def iou(a, b):
    """Overlap of two 0/1 masks: shared area / combined area."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def main():
    test = load_split("test")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    outputs = predict(model, test)

    print("Test photos:", len(test))
    print("\nRaw pycocotools output (segm):")
    map_all, map_50 = coco_mask_map(test, outputs, quiet=False)

    # Each photo has one card. The best-scoring mask is the answer.
    ious, true_pos, false_pos, false_neg = [], 0, 0, 0
    for (name, image, polygons), out in zip(test, outputs):
        truth = polygon_masks(image, polygons)[0]
        kept = [i for i, s in enumerate(out["scores"]) if s >= MIN_SCORE]
        best_iou = 0.0
        if kept:
            # torchvision sorts detections by score, so kept[0] is the best.
            found = (out["masks"][kept[0], 0] > 0.5).numpy()
            best_iou = iou(found, truth)
        ious.append(best_iou)
        hit = best_iou >= MIN_IOU
        true_pos += int(hit)
        false_neg += int(not hit)
        false_pos += len(kept) - int(hit)
        print(f"  {name}: mask IoU {best_iou:.3f}, "
              f"detections above {MIN_SCORE}: {len(kept)}")

    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)
    print(f"\nmask mAP@0.5:0.95 {map_all:.3f}, mask mAP@0.5 {map_50:.3f}")
    print(f"mean mask IoU {np.mean(ious):.3f} (min {np.min(ious):.3f})")
    print(f"precision {precision:.3f}, recall {recall:.3f} "
          f"(score >= {MIN_SCORE}, IoU >= {MIN_IOU})")


if __name__ == "__main__":
    main()
