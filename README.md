# measure-it

One command turns a phone photo into the width and height of my card in millimetres, with a confidence and an overlay image. When it can't trust the answer, it refuses and says why.

```
python inference/measure.py dataset/measure/IMG_20261122_200741.jpg
python inference/measure.py dataset/refusal/no_marker.jpg     # REFUSED, marker not found
```

## Results
| What | Result | Where |
|---|---|---|
| Calibration (item 1) | RMS 2.29 px (0.19 mm at 245 mm) over 23 board photos; fx 3002; board lines 8.24 px off straight raw, 5.27 px undistorted. Misses my 1 px target | `calibration/calibration_output.txt`, `calibration/undistort_output.txt` |
| Segmentation, test split, run once (item 3) | Mask mAP@0.5:0.95 0.956, mAP@0.5 1.000, mean mask IoU 0.948 (min 0.938), precision 1.000, recall 1.000 on 8 photos. Optimistic: see Dataset | `results/test_evaluation.txt` |
| Accuracy against my ruler (item 5) | MAE 2.29 mm over 16 photos, median 2.04 mm, worst 4.05 mm. **MO-1 (2 mm) missed** | `results/accuracy.md` |
| Raw vs undistorted (item 6) | Raw 0.71 mm, undistorted 2.29 mm: my calibration makes it worse. Explained in `docs/OBJECTIVES.md` | `results/accuracy.md` |
| Speed (item 7) | Median 2.68 s per photo on the CPU, plus 0.7 s to load the model. **MO-2 (5 s) met** | `results/accuracy_output.txt` |
| Refusals (item 9) | Refuses below 0.90 confidence, above 25° tilt, with the marker within 50 px of the edge, or with the card touching the edge. None of my 16 photos crossed a threshold, so the error each one removes is not shown yet | `results/accuracy.md`, `results/refusal_example.txt` |
| What breaks it (item 8) | Not reached; see "Changes to this plan" in `docs/PLAN.md` | |

The plan, objectives and definition of done are in `docs/`. Overlays for every measurement photo are in `results/overlays/`.

## Hardware
- **Phone:** Samsung Galaxy A04e, 13 MP main camera, Open Camera app at 4160 × 3120 with focus and exposure locked. Held by hand, about 20–25 cm above the table.
- **Computer:** MacBook Pro 2019, Intel i7-9750H (6 cores), 32 GB RAM, macOS 26.6. Everything runs on the CPU.
- **Object:** my Dis-Chem Benefit card, front side up. Ruler size 85.5 × 54.5 mm.
- **Reference:** a printed ArUco marker (DICT_4X4_50, id 0), 90.0 mm a side, measured with the ruler. It lies beside the card on the same sheet.
- **Calibration board:** a printed checkerboard, 10 × 8 squares (9 × 7 inner corners), 21.75 mm squares, measured with the ruler.
- **Ground truth:** a plastic ruler with 1 mm marks, read to about ±0.5 mm.

## Setup
Python 3.12 (torch 2.2.2 is the last build for Intel Macs and needs NumPy 1.x).
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
gh release download v1.0 -R SENZO-NCEKANA/measure-it -p best.pth -D models
shasum -a 256 models/best.pth
```
Weights: `models/best.pth` in release v1.0, the epoch-3 checkpoint.
SHA-256: `04ab06ceac08f804e83add366b6a14dc7bf1a494c8b89362472d237573e994ef`

## Run it again from scratch
```
python calibration/calibrate.py      # board photos -> calibration/camera.npz
python calibration/undistort.py      # dataset/raw -> dataset/images (straightened)
python models/train.py               # -> models/best.pth, best epoch by val mask mAP
python models/evaluate.py            # test split, once
python evaluation/accuracy.py        # items 5, 6, 7 and 9 -> results/accuracy.md
```
Masks were drawn by hand in Label Studio, running locally, on `dataset/images/`, and exported to `dataset/labels/coco.json`. Training was stopped after epoch 3 of 5 (see `docs/PLAN.md`); `models/train_log.txt` has the log.

## Dataset
- **80 photos I took** of the card with the marker beside it, renamed `card_001` to `card_080` in the order taken (`dataset/rename_log.csv`). The raw photos are in release v1.0 as `dataset_raw.zip`; unzip it at the repo root, and `calibration/undistort.py` rebuilds the straightened copies.
- **Masks:** one 4-point polygon per photo, drawn by hand on the straightened photos: `dataset/labels/coco.json`.
- **Split:** 56 train, 16 val, 8 test, fixed with `random.Random(0)` and committed before any photo was taken (`dataset/splits/`).
- **Known weakness:** I didn't turn the card between shots, so many photos are near-copies that fall across train, val and test. The test scores are optimistic. Training uses flip and 90° rotation augmentation to make up for it in part.
- **Measurement photos:** 16 separate photos (`dataset/measure/`), never used in training: one placement in the centre of the frame.
- **Refusal example:** `dataset/refusal/no_marker.jpg`, a photo without the marker.
