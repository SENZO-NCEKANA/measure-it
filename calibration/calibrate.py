"""Item 1: calibrate the phone camera from the checkerboard photos.

Run from the repo root:  python calibration/calibrate.py
Reads calibration/images/*.jpg and saves calibration/camera.npz.
"""
import glob

import cv2
import numpy as np

np.set_printoptions(suppress=True)  # print plain numbers, not 3.1e+03

# %% The board: 9 x 7 inner corners, squares measured at 21.75 mm.
PATTERN = (9, 7)
SQUARE_MM = 21.75

# Where each inner corner really is on the flat board, in mm.
# z is 0 because the board is flat.
board_points = np.array(
    [[col * SQUARE_MM, row * SQUARE_MM, 0]
     for row in range(PATTERN[1]) for col in range(PATTERN[0])],
    dtype=np.float32,
)

# Refine each corner for up to 30 rounds, or until it moves < 0.001 px.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

# %% Find the board in every photo.
# IGNORE_ORIENTATION keeps every photo in the sensor's own 4160 x 3120
# layout: the phone tagged some photos portrait and some landscape.
paths = sorted(glob.glob("calibration/images/*.jpg"))
object_points = []  # the board's real corner positions, once per photo
image_points = []   # where those corners were found in each photo
used_paths = []     # the photos where the board was found
image_size = None

for path in paths:
    gray = cv2.imread(
        path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    image_size = gray.shape[::-1]  # (width, height)
    # Search a quarter-size copy (much faster),
    # then refine the corners on the full photo.
    small = cv2.resize(gray, (image_size[0] // 4, image_size[1] // 4))
    found, corners = cv2.findChessboardCorners(
        small, PATTERN, flags=find_flags)
    print(path, "board found" if found else "board NOT found, skipped")
    if found:
        corners = cv2.cornerSubPix(
            gray, corners * 4, (11, 11), (-1, -1), criteria)
        object_points.append(board_points)
        image_points.append(corners)
        used_paths.append(path)

if not image_points:
    raise SystemExit("No board found in any photo in calibration/images/.")
print("Photos used:", len(image_points), "of", len(paths))

# %% First, a free calibration: every number is worked out from the photos.
rms_free, matrix_free, _, _, _ = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None)
print("\nFree calibration (kept only for the record):")
print("  RMS (px):", round(rms_free, 3))
print("  fx, fy:", matrix_free[0, 0].round(0), matrix_free[1, 1].round(0),
      " centre cx, cy:",
      matrix_free[0, 2].round(0), matrix_free[1, 2].round(0))

# %% The calibration I use: centre fixed at the middle of the photo,
# fx = fy, k3 = 0. My board photos all view the board from one side,
# so the free fit cannot pin the centre down. Fixing it is a standard
# constraint for a phone's main lens.
start_matrix = np.array([[3000.0, 0, image_size[0] / 2],
                         [0, 3000.0, image_size[1] / 2],
                         [0, 0, 1]])
flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
         | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3)


def calibrate(obj, img):
    """Fit the camera with the fixed centre, fx = fy and k3 = 0."""
    return cv2.calibrateCamera(
        obj, img, image_size, start_matrix.copy(), None, flags=flags)


def per_photo_errors(img, matrix, dist, rotations, translations):
    """Error per photo in px.

    Puts the board's real corners back into each photo using the
    calibration, and measures how far they land from where they were found.
    """
    result = []
    for found_corners, rvec, tvec in zip(img, rotations, translations):
        predicted, _ = cv2.projectPoints(
            board_points, rvec, tvec, matrix, dist)
        squared = np.sum((found_corners - predicted) ** 2, axis=2)
        result.append(float(np.sqrt(np.mean(squared))))
    return np.array(result)


rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate(
    object_points, image_points)

# %% Drop the photos that stand out (more than twice the median error),
# then fit once more. When only a few stand out, the problem is those
# photos (blur, a moved board), not the board.
errors = per_photo_errors(
    image_points, camera_matrix, dist_coeffs, rvecs, tvecs)
limit = 2 * np.median(errors)
for path, error in zip(used_paths, errors):
    if error > limit:
        print("Dropped", path, "error", round(error, 2), "px")
keep = errors <= limit
object_points = [p for p, k in zip(object_points, keep) if k]
image_points = [p for p, k in zip(image_points, keep) if k]
used_paths = [p for p, k in zip(used_paths, keep) if k]
rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate(
    object_points, image_points)
print("Photos in the final fit:", len(image_points))

fx = camera_matrix[0, 0]
# How far away the board typically was, in mm.
distance_mm = float(np.median([t[2, 0] for t in tvecs]))
view_deg = np.degrees(2 * np.arctan(image_size[0] / 2 / fx))
print("\nCalibration used (centre fixed, fx = fy, k3 = 0):")
print("  RMS (px):", round(rms, 3))
print("  Camera matrix:\n", camera_matrix.round(1))
print("  Distortion (k1, k2, p1, p2, k3):", dist_coeffs.ravel().round(4))
print("  Field of view across:", round(float(view_deg), 1), "degrees")
print("  Board distance (median):", round(distance_mm), "mm")
print("  RMS in mm at that distance:", round(rms * distance_mm / fx, 3))

# %% Per-photo error in the final fit.
print("\nError per photo (px):")
final_errors = per_photo_errors(
    image_points, camera_matrix, dist_coeffs, rvecs, tvecs)
for path, error in zip(used_paths, final_errors):
    print(" ", path, round(error, 2))

# %% Save what the other scripts need.
np.savez("calibration/camera.npz", camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs, image_size=np.array(image_size), rms=rms)
print("\nSaved calibration/camera.npz")
