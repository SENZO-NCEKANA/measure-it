# Objectives

Set at the start, in PLAN.md, before any photo. Measured by `evaluation/accuracy.py`; the raw output is in `results/accuracy_output.txt` and `results/accuracy.md`.

## MO-1, accuracy: missed
**Target:** mean absolute error (MAE) of 2 mm or less on the card's width and height, against my ruler.
**Why 2 mm:** my ruler has 1 mm marks and I read it to about ±0.5 mm, so I can't prove a smaller error.
**Truth:** 85.5 × 54.5 mm (3 readings per side, all equal).
**Result, the system as designed (undistorted):** MAE **2.29 mm** over 16 photos (width and height together), mean error 3.11 %, median 2.04 mm, worst 4.05 mm (a width). **Missed by 0.29 mm.** The width is the problem: it comes out about 3.3 mm short in every photo, while the height is about 1.2 mm short.
**Same photos without undistortion:** MAE **0.71 mm**, median 0.63 mm, worst 1.72 mm. My calibration shrinks the card by about 4 %. Its board photos all viewed the board from one side and its centre was fixed, so its lens estimate (k1 = 0.073) is probably wrong, and the phone already corrects its own lens in software. I did not switch the default to raw after seeing this, because that would be tuning on my measurement photos. The next step is to recalibrate with board views from every side and measure again on new photos.
**Limit of this test:** the 16 photos are one placement in the centre of the frame, taken by hand. They test repeatability (width SD 0.53 mm, height SD 0.34 mm) and the centre of the lens, not the frame edges.

## MO-2, speed: met
**Target:** the one command takes 5 s or less per photo on my laptop's CPU (median).
**Result:** median **2.68 s** per photo over 16 photos (range 2.47–3.19 s), plus 0.7 s to load the model once. **Met.**
**Hardware:** MacBook Pro 2019, Intel i7-9750H, 6 cores, no GPU. Measured with nothing else running (a first run overlapped training and gave a median of 6.36 s; that run is not used).
