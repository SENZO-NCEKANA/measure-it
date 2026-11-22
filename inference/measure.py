"""Item 4: one command from a phone photo to the card's size in mm.

Run from the repo root:  python inference/measure.py PHOTO [PHOTO ...]
For each photo it prints width x height in mm, a confidence and the
camera tilt, and saves an overlay image to results/overlays/.
It refuses, with a reason, when it cannot trust the answer (item 9).
"""
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration.undistort import (CAMERA_MATRIX, DIST_COEFFS,  # noqa: E402
                                   read_photo, undistort)
from models.train import SHORT_SIDE, build_model  # noqa: E402

# %% Settings, written down before measuring the ruler photos.
MARKER_MM = 90.0        # my printed ArUco marker, measured with the ruler
WEIGHTS = "models/best.pth"
OVERLAYS = "results/overlays"
MIN_SCORE = 0.90        # refuse a card mask the model is less sure of
MIN_EDGE_PX = 50        # refuse when the marker is this close to the frame
MAX_TILT_DEG = 25.0     # refuse when the camera leans more than this

DETECTOR = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters())
MARKER_SQUARE_MM = np.array(
    [[0, 0], [MARKER_MM, 0], [MARKER_MM, MARKER_MM], [0, MARKER_MM]],
    dtype=np.float32)


def load_model():
    """The fine-tuned Mask R-CNN, on the CPU, ready to predict."""
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return model.eval()


def find_marker(gray):
    """The marker's 4 corners in full-size pixels, or None.

    The marker is found on a half-size copy (at full size it is missed),
    then each corner is sharpened on the full photo.
    """
    height, width = gray.shape
    small = cv2.resize(gray, (width // 2, height // 2))
    corners, ids, _ = DETECTOR.detectMarkers(small)
    if ids is None or 0 not in ids.ravel():
        return None
    found = corners[list(ids.ravel()).index(0)].reshape(4, 1, 2) * 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    return cv2.cornerSubPix(gray, found, (11, 11), (-1, -1),
                            criteria).reshape(4, 2)


def tilt_degrees(marker_px, dist_coeffs):
    """How far the camera leans from straight down, from the marker."""
    marker_3d = np.hstack([MARKER_SQUARE_MM, np.zeros((4, 1), np.float32)])
    ok, rvec, _ = cv2.solvePnP(marker_3d, marker_px, CAMERA_MATRIX,
                               dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return 90.0
    rotation, _ = cv2.Rodrigues(rvec)
    return float(np.degrees(np.arccos(abs(rotation[2, 2]))))


@torch.no_grad()
def card_mask(model, photo):
    """The best card mask at full size, and the model's score for it."""
    scale = SHORT_SIDE / min(photo.shape[:2])
    small = cv2.resize(photo, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    output = model([tensor])[0]
    if len(output["scores"]) == 0:
        return None, 0.0
    best = int(output["scores"].argmax())
    soft = output["masks"][best, 0].numpy()
    full = cv2.resize(soft, (photo.shape[1], photo.shape[0]),
                      interpolation=cv2.INTER_LINEAR)
    return (full > 0.5).astype(np.uint8), float(output["scores"][best])


def measure(model, path, straighten=True):
    """Measure the card in one photo. Returns a dict; 'refused' says why not.

    straighten=False skips undistortion (used only to show it matters).
    """
    started = time.perf_counter()
    photo = read_photo(path)
    dist_coeffs = DIST_COEFFS
    if straighten:
        photo = undistort(photo)
        dist_coeffs = np.zeros(5)
    result = {"photo": os.path.basename(path), "refused": None}
    height, width = photo.shape[:2]

    marker = find_marker(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY))
    if marker is None:
        result["refused"] = "marker not found"
        return result
    edge = min(marker[:, 0].min(), marker[:, 1].min(),
               width - marker[:, 0].max(), height - marker[:, 1].max())
    result["marker_edge_px"] = float(edge)
    if edge < MIN_EDGE_PX:
        result["refused"] = f"marker {edge:.0f} px from the frame edge"
        return result
    result["tilt_deg"] = tilt_degrees(marker, dist_coeffs)
    if result["tilt_deg"] > MAX_TILT_DEG:
        result["refused"] = f"camera tilted {result['tilt_deg']:.0f} deg"
        return result

    mask, score = card_mask(model, photo)
    result["confidence"] = score
    if mask is None or score < MIN_SCORE:
        result["refused"] = f"card not found confidently (score {score:.2f})"
        return result
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if border.any():
        result["refused"] = "card touches the frame edge"
        return result

    # Map the card's outline onto the marker's flat plane, in mm, and fit the
    # smallest rectangle around it. Card and marker lie on the same table.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea).astype(np.float32)
    to_mm = cv2.getPerspectiveTransform(marker.astype(np.float32),
                                        MARKER_SQUARE_MM)
    outline_mm = cv2.perspectiveTransform(outline, to_mm)
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(outline_mm)
    result["width_mm"] = max(side_a, side_b)
    result["height_mm"] = min(side_a, side_b)
    result["seconds"] = time.perf_counter() - started

    # Overlay: card outline in green, marker in blue, the answer on top.
    cv2.drawContours(photo, [outline.astype(np.int32)], -1, (0, 255, 0), 8)
    cv2.polylines(photo, [marker.astype(np.int32)], True, (255, 0, 0), 8)
    label = (f"{result['width_mm']:.1f} x {result['height_mm']:.1f} mm  "
             f"conf {score:.2f}  tilt {result['tilt_deg']:.0f} deg")
    cv2.putText(photo, label, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 3,
                (0, 0, 255), 8)
    os.makedirs(OVERLAYS, exist_ok=True)
    result["overlay"] = os.path.join(OVERLAYS, os.path.basename(path))
    cv2.imwrite(result["overlay"], cv2.resize(photo, None, fx=0.5, fy=0.5))
    return result


def main():
    """Measure every photo named on the command line."""
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python inference/measure.py PHOTO ...")
    model = load_model()
    for path in sys.argv[1:]:
        result = measure(model, path)
        if result["refused"]:
            print(f"{result['photo']}: REFUSED, {result['refused']}")
            continue
        print(f"{result['photo']}: {result['width_mm']:.1f} x "
              f"{result['height_mm']:.1f} mm, "
              f"confidence {result['confidence']:.3f}, "
              f"tilt {result['tilt_deg']:.1f} deg, "
              f"{result['seconds']:.1f} s, overlay {result['overlay']}")


if __name__ == "__main__":
    main()
