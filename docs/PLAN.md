# Plan

**Task:** measure my Dis-Chem Benefit card in millimetres from one phone photo.
**Clock:** started 16:00 on Sunday 22 November 2026 (SAST). Reply by 00:00.

## Setup
- **Object:** my Dis-Chem Benefit card, front side up (about 85.5 × 54 mm).
- **Reference:** a printed ArUco marker (dictionary 4×4_50, id 0), measured with my ruler at 90.0 mm a side.
- **Camera:** Samsung Galaxy A04e on a stand, using Open Camera at 4160 × 3120, landscape. Focus, exposure and zoom are locked once and kept for every photo.
- **Ground truth:** a plastic ruler with 1 mm marks, read to about ±0.5 mm.
- **Calibration board:** a printed checkerboard, 8 × 10 squares (7 × 9 inner corners), with 21.75 mm squares.

## Order and time
| Clock | Block |
|---|---|
| 16:00–16:45 | Repo, split lists, this plan, definition of done, first commit |
| 16:45–17:55 | One photo session on the stand, focus locked once and Open Camera never closed: 30 board photos, 80 dataset photos, 12 measurement photos, 5 repeatability photos, tilt photos at 0°, 10°, 20° and 30° |
| 17:55–18:40 | Transfer the photos. Item 1: calibrate, undistort, report the reprojection error |
| 18:40–19:40 | Item 2: undistort the 80 dataset photos, label the masks in Label Studio, export COCO |
| 19:40–20:10 | Item 3: training code and config, then start training on the CPU (it runs while I build item 4) |
| 20:10–21:20 | Item 4: the one command, with refusals. When training ends, evaluate once on test |
| 21:20–22:20 | Items 5, 6, 7 and 9: ruler table, raw vs undistorted, timing, refusal thresholds. Then item 8 (tilt) |
| 22:20–23:05 | Objectives, definition-of-done ticks, README, weights and SHA-256, pull request |
| 23:05–23:40 | Loom: practise twice, record once (5 minutes or less) |
| 23:40–00:00 | Reply to Rehman, Mahnoor; spare time |

## Split (fixed before any photo was taken)
- 80 dataset photos, renamed `card_001.jpg` to `card_080.jpg` in the order I take them.
- Shuffled with Python's `random.Random(0).shuffle`, then cut: the first 56 go to train, the next 16 to val, the last 8 to test (70/20/10).
- The lists are in `dataset/splits/`. Test photos are never used for training, tuning or choosing a checkpoint.
- A blurry or cut-off photo is retaken under the same number, so the split never changes.
- Between shots I move and turn the card, so no two photos are near-copies.
- The 12 measurement photos are separate from the 80, and never used for training.

## Targets (details in OBJECTIVES.md)
- **MO-1, accuracy:** mean absolute error of 2 mm or less on the card's width and height, over the 12 measurement photos. My ruler reads to about ±0.5 mm, so I can't prove a smaller error.
- **MO-2, speed:** the one command takes 5 s or less per photo on my laptop's CPU (the median of 5 runs).

## Order of the items
Items 1–4 come first: they are the floor. Then items 5 and 6, then 7 and 9. Item 8 starts only after item 5 is finished. I take the tilt photos in the same session so the camera stays the same, but I analyse them only after item 5. If time runs short, I stop and write down what I didn't reach, and why.

## What I'm skipping
- A calliper: I use a ruler, so my target is 2 mm, not 1 mm.
- A ChArUco board: I use the plain checkerboard I already printed.
- Correcting tilt: the command refuses above a tilt limit instead.
- Tuning: I use one training config, chosen before training.
- Automatic labelling: every mask is drawn by hand, locally.
- A GPU or the cloud: everything runs on my laptop's CPU.

## Changes to this plan
None yet. Each change gets the time and the reason.

## Decision record (item 10)
- **Took:** torchvision's Mask R-CNN, pretrained on COCO and fine-tuned on my photos.
- **Rejected:** U-Net.
- **Why:** Mask R-CNN gives a confidence score for each object, which my refusals need (item 9), and pycocotools gives the standard mAP for its output. U-Net gives no score for each object, and it would need my own mAP code.
- **What rejecting U-Net cost:** speed. Mask R-CNN is a large model. My speed test before the clock gave 5.71 s per training step and 2.35 s per photo on my CPU, at size 800. I'll add the real training time at the end.
