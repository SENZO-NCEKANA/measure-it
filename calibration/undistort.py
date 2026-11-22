"""Items 1, 2 and 6: straighten photos with the saved calibration.

Other scripts import read_photo() and undistort() from here.
Run from the repo root:  python calibration/undistort.py
It straightens the 80 dataset photos into dataset/images/ for labelling,
and checks that the board's straight lines are straighter afterwards.
"""
import glob
import os

import cv2
import numpy as np

_saved = np.load(os.path.join(os.path.dirname(__file__), "camera.npz"))
CAMERA_MATRIX = _saved["camera_matrix"]
DIST_COEFFS = _saved["dist_coeffs"]


def read_photo(photo_path):
    """Open a photo in the sensor's own 4160 x 3120 layout.

    That is the same layout the calibration used.
    """
    return cv2.imread(
        photo_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def undistort(photo):
    """Straighten one photo. Same size out as in, same camera matrix."""
    return cv2.undistort(photo, CAMERA_MATRIX, DIST_COEFFS)


def worst_bend_px(photo):
    """How far the worst board corner sits off its straight line, in px.

    Each row and column of board corners should lie on a straight line.
    Returns None if the board is not found.
    """
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4))
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(small, (9, 7), flags=flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    grid = cv2.cornerSubPix(
        gray, corners * 4, (11, 11), (-1, -1), criteria).reshape(7, 9, 2)
    worst = 0.0
    for line in list(grid) + list(grid.transpose(1, 0, 2)):
        vx, vy, x0, y0 = cv2.fitLine(
            line, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        off = np.abs((line[:, 0] - x0) * vy - (line[:, 1] - y0) * vx)
        worst = max(worst, float(off.max()))
    return worst


if __name__ == "__main__":
    # %% Straighten the 80 dataset photos for labelling.
    os.makedirs("dataset/images", exist_ok=True)
    paths = sorted(glob.glob("dataset/raw/card_*.jpg"))
    for path in paths:
        out = os.path.join("dataset/images", os.path.basename(path))
        cv2.imwrite(out, undistort(read_photo(path)),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("Straightened", len(paths), "dataset photos into dataset/images/")

    # %% Check: are the board's lines straighter after undistortion?
    raw_bends, fixed_bends = [], []
    for path in sorted(glob.glob("calibration/images/*.jpg")):
        image = read_photo(path)
        raw, fixed = worst_bend_px(image), worst_bend_px(undistort(image))
        if raw is not None and fixed is not None:
            raw_bends.append(raw)
            fixed_bends.append(fixed)
    print("Board photos checked:", len(raw_bends))
    print("Worst corner off its straight line, median over photos (px):")
    print("  raw:", round(float(np.median(raw_bends)), 2),
          "  undistorted:", round(float(np.median(fixed_bends)), 2))
