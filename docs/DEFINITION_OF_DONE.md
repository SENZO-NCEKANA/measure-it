# Definition of Done

Written at the start, before any photo or code. At the end, I tick each line and put a number or a link beside it. A tick without evidence counts as unticked.

## The floor (items 1–4)
- [ ] The camera is calibrated from 20 or more board photos, with an RMS reprojection error under 1 px, reported in px and mm. **Missed:** 23 photos, RMS 2.29 px (0.19 mm at 245 mm). `calibration/calibration_output.txt`
- [x] fx is between 2,700 and 3,600 with fx ≈ fy, and the centre is near (2080, 1560). fx = fy = 3002 (69° view). The centre is fixed at (2080, 1560) by constraint, because the free fit put it at cy = 1731. `calibration/calibration_output.txt`
- [x] Straight edges stay straight after undistortion. Shown as a number rather than an image: the worst board corner sits 8.24 px off its straight line raw, 5.27 px undistorted (median over 25 photos). `calibration/undistort_output.txt`
- [x] 80 photos I took myself, each undistorted, with a mask I drew by hand, exported as COCO. `dataset/labels/coco.json`, `dataset/rename_log.csv`
- [x] The split lists were pushed before I took any photo. Commit `128a506`, 16:40.
- [x] Mask R-CNN is fine-tuned, and the best checkpoint is chosen on val only. Epoch 3, val mask mAP@0.5:0.95 0.991. `models/train_log.txt`
- [x] The test split is evaluated once: mask mAP@0.5:0.95 0.956, mAP@0.5 1.000, mean mask IoU 0.948, precision 1.000, recall 1.000, as raw output. `results/test_evaluation.txt`
- [x] One command takes a photo and gives width and height in mm, a confidence and an overlay image. `inference/measure.py`, `results/overlays/`
- [x] It refuses, with a reason, when the card or the marker isn't found. `results/refusal_example.txt`

## Proof (items 5–9)
- [x] 16 measurement photos against my ruler: a row for each photo, then the MAE, mean % error, median and worst case. **MO-1 reported as missed:** MAE 2.29 mm. `results/accuracy.md`
- [x] The same photos measured raw and undistorted, side by side: raw 0.71 mm, undistorted 2.29 mm. `results/accuracy.md`, `docs/OBJECTIVES.md`
- [x] Repeatability: one placement photographed 16 times; width SD 0.53 mm, height SD 0.34 mm. `results/accuracy.md`
- [x] Speed: the command timed on 16 photos, median 2.68 s, range 2.47–3.19 s. **MO-2 met.** `results/accuracy_output.txt`
- [ ] Refusals for low confidence, a marker at the frame edge and too much tilt: each threshold, the error it removes and the % of photos it refuses. **Partly:** thresholds and the % refused (0 of 16) are shown; no photo crossed one, so the error removed is not shown. `results/accuracy.md`
- [ ] Tilt at 0°, 10°, 20° and 30°. **Not reached:** skipped at 19:36 to protect the floor. `docs/PLAN.md`

## Repo and hand-in
- [x] Commits come in stages, and each message says what changed and why. (One slip: `8a5c2cf` and `f7ccd90` share a message; the second added `undistort.py` and shortened lines.)
- [x] No dead code, commented-out blocks or stray files.
- [x] The seed, versions and settings are pinned: `requirements.txt`; seed 0 and every training setting at the top of `models/train.py`.
- [x] PLAN, OBJECTIVES and DEFINITION_OF_DONE are in `docs/`, and the README (hardware, setup, how to run, dataset) is at the root.
- [x] The weights are a release asset, with their SHA-256 in the README. Release v1.0, SHA-256 `04ab06ce…`
- [ ] One pull request from `dev-test01` into `main`. Ticked in the pull request itself.
- [ ] A Loom of 5 minutes or less: a new photo measured live, the code walk-through, and K1–K4. Linked in my reply.
