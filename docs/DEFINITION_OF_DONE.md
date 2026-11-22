# Definition of Done

Written at the start, before any photo or code. At the end, I tick each line and put a number or a link beside it. A tick without evidence counts as unticked.

## The floor (items 1–4)
- [ ] The camera is calibrated from 20 or more board photos, with an RMS reprojection error under 1 px, reported in px and mm.
- [ ] fx is between 2,700 and 3,600 with fx ≈ fy, and the centre is near (2080, 1560).
- [ ] Straight edges stay straight after undistortion (one before-and-after image).
- [ ] 80 photos I took myself, each undistorted, with a mask I drew by hand, exported as COCO.
- [ ] The split lists were pushed before I took any photo.
- [ ] Mask R-CNN is fine-tuned, and the best checkpoint is chosen on val only.
- [ ] The test split is evaluated once: mask IoU, mAP@0.5, mAP@0.5:0.95, precision and recall, as raw output.
- [ ] One command takes a photo and gives width and height in mm, a confidence and an overlay image.
- [ ] It refuses, with a reason, when the card or the marker isn't found.

## Proof (items 5–9)
- [ ] 12 measurement photos against my ruler: a row for each photo, then the MAE, mean % error, median and worst case. MO-1 (MAE ≤ 2 mm) is met, or reported as missed.
- [ ] The same 12 photos measured raw and undistorted, side by side.
- [ ] Repeatability: one placement photographed 5 times, with the spread of the 5 measurements.
- [ ] Speed: the command timed 5 times, with the median and spread. MO-2 (5 s or less) is met, or reported as missed.
- [ ] Refusals for low confidence, a marker at the frame edge and too much tilt: each threshold, the error it removes and the % of photos it refuses.
- [ ] Tilt at 0°, 10°, 20° and 30°: how the error grows, started only after item 5 was finished.

## Repo and hand-in
- [ ] Commits come in stages, and each message says what changed and why.
- [ ] No dead code, commented-out blocks or stray files.
- [ ] The seed, versions and settings are pinned, in `requirements.txt` and the config.
- [ ] PLAN, OBJECTIVES and DEFINITION_OF_DONE are in `docs/`, and the README (hardware, setup, how to run, dataset) is at the root.
- [ ] The weights are a release asset, with their SHA-256 in the README.
- [ ] One pull request from `dev-test01` into `main`.
- [ ] A Loom of 5 minutes or less: a new photo measured live, the code walk-through, and K1–K4.
