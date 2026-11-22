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
- **17:30, board photos, first set.** My first 30 board photos held only 3 board positions: I took them in bursts and didn't move the board between shots. A trial calibration gave RMS 2.3 px with the centre 245 px off. I moved them out of the repo and retook them.
- **17:50, board photos, second set.** 25 of 30 were usable, but all view the board from one side, so the free calibration put the centre at cy = 1731. I fixed the centre at the middle of the image, with fx = fy and k3 = 0, and dropped the 2 photos above twice the median error. Result: 23 photos, RMS 2.29 px (0.19 mm at 245 mm), fx 3002, against 3005 in my practice run on 8 Nov. This misses my 1 px target. I kept it to protect the floor.
- **18:47 to 19:36, dataset photos.** My first three attempts were unusable: the card and the marker were in separate photos, or a ruler stood in for the marker, and many were burst copies. I moved each set out of the repo and retook them, checking one test photo first. This cost about 1.5 hours.
- **19:36, tilt photos (item 8) skipped** to protect the floor. Item 8 is not reached.
- **19:50, dataset kept with a known weakness.** 83 photos with the card and the marker together (marker found in 82); I kept the first 80. I didn't turn the card between shots, so the variety comes from moving the phone, and many photos are near-copies that fall across train, val and test. My test scores will be optimistic. I added flip and 90° rotation augmentation in training.
- **20:12, measurement photos.** 16 photos of one placement in the centre of the frame, not 12 placements, and none near the frame edges. They serve both item 5 (one row per photo) and repeatability, so there are no separate repeatability photos. Item 6 is therefore tested near the centre only, where distortion is weakest.
- **Camera held by hand, not on a stand.** Distance and tilt vary a little between photos. The marker sets the scale in each photo, so this doesn't change the mm conversion.
- **21:46, training settings before the first run:** 5 epochs instead of an open number, and only the last backbone stage trainable, to fit the CPU time left. Nothing was tuned after seeing a result.
- **Item 4, tilt.** The mm conversion maps the card's outline onto the marker's plane with a homography from the marker's 4 corners, which corrects moderate camera tilt as well. The tilt limit stays as a refusal.
- **22:21, training stopped after epoch 3 of 5.** I ran the test evaluation while training was still in epoch 4, on the epoch-3 checkpoint (the best val so far, mask mAP 0.991). To keep the test split scored once, on the final model, I stopped training there. Epochs 4 and 5 were not completed.
- **22:30, accuracy rerun.** The first accuracy run overlapped the end of training, which slowed its timing (median 6.36 s). I reran it with nothing else running: same measurements, median 2.68 s.
- **Refusals (item 9) only partly shown.** None of my 16 measurement photos crossed a threshold, so I can show where each threshold sits but not yet the error it removes. A photo without the marker is refused (`results/refusal_example.txt`).

## Decision record (item 10)
- **Took:** torchvision's Mask R-CNN, pretrained on COCO and fine-tuned on my photos.
- **Rejected:** U-Net.
- **Why:** Mask R-CNN gives a confidence score for each object, which my refusals need (item 9), and pycocotools gives the standard mAP for its output. U-Net gives no score for each object, and it would need my own mAP code.
- **What rejecting U-Net cost:** speed. Mask R-CNN is a large model. My speed test before the clock gave 5.71 s per training step and 2.35 s per photo on my CPU, at size 800. **On the day:** 3 epochs took 24 minutes on the CPU (about 6.4 minutes per epoch with nothing else running), which is why I froze all but the last backbone stage and stopped at 5 epochs or fewer. The one command takes a median of 2.68 s per photo.
