"""Items 5, 6, 7 and 9: accuracy against my ruler, raw vs undistorted,
speed, and what the refusal thresholds cost.

Run from the repo root:  python evaluation/accuracy.py
Measures every photo in dataset/measure/ twice (undistorted and raw),
prints the tables and writes them to results/accuracy.md.
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inference.measure as measuring  # noqa: E402

# %% Ground truth: my ruler readings of the card, as raw marks. The ruler's
# 10 mm mark sat on one edge, so 10 is subtracted. Read to about +-0.5 mm.
LONG_RAW = [95.5, 95.5, 95.5]
SHORT_RAW = [64.5, 64.5, 64.5]
TRUE_WIDTH = float(np.mean(LONG_RAW)) - 10
TRUE_HEIGHT = float(np.mean(SHORT_RAW)) - 10


def errors(rows, key):
    """Absolute errors in mm and % errors for width and height together."""
    abs_mm, pct = [], []
    for row in rows:
        result = row[key]
        for found, true in ((result["width_mm"], TRUE_WIDTH),
                            (result["height_mm"], TRUE_HEIGHT)):
            abs_mm.append(abs(found - true))
            pct.append(100 * abs(found - true) / true)
    return np.array(abs_mm), np.array(pct)


def summary(rows, key):
    """MAE, mean % error, median and worst case, as one line of text."""
    abs_mm, pct = errors(rows, key)
    worst = int(abs_mm.argmax())
    side = "width" if worst % 2 == 0 else "height"
    return (f"MAE {abs_mm.mean():.2f} mm, mean error {pct.mean():.2f} %, "
            f"median {np.median(abs_mm):.2f} mm, worst {abs_mm.max():.2f} mm "
            f"({rows[worst // 2]['photo']}, {side})")


def main():
    # %% Measure every photo with the refusals switched off, so item 9 can
    # show afterwards what each threshold would have removed.
    limits = (measuring.MIN_SCORE, measuring.MAX_TILT_DEG,
              measuring.MIN_EDGE_PX)
    measuring.MIN_SCORE, measuring.MAX_TILT_DEG = 0.0, 90.0
    measuring.MIN_EDGE_PX = 0

    started = time.perf_counter()
    model = measuring.load_model()
    load_seconds = time.perf_counter() - started
    rows = []
    for path in sorted(glob.glob("dataset/measure/*.jpg")):
        straight = measuring.measure(model, path)
        raw = measuring.measure(model, path, straighten=False)
        if straight["refused"] or raw["refused"]:
            print(path, "could not be measured:",
                  straight["refused"] or raw["refused"])
            continue
        rows.append({"photo": straight["photo"], "straight": straight,
                     "raw": raw})

    lines = [f"Truth (ruler): {TRUE_WIDTH:.1f} x {TRUE_HEIGHT:.1f} mm. "
             f"Photos: {len(rows)}, one card placement in the centre, "
             "each a separate handheld photo.", "",
             "| Photo | Width | Height | Width error | Height error "
             "| Raw width | Raw height | Confidence | Tilt |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        s, r = row["straight"], row["raw"]
        lines.append(
            f"| {row['photo']} | {s['width_mm']:.1f} | {s['height_mm']:.1f} "
            f"| {s['width_mm'] - TRUE_WIDTH:+.1f} "
            f"| {s['height_mm'] - TRUE_HEIGHT:+.1f} "
            f"| {r['width_mm']:.1f} | {r['height_mm']:.1f} "
            f"| {s['confidence']:.3f} | {s['tilt_deg']:.1f} |")

    # %% Item 5: accuracy and repeatability. Item 6: raw vs undistorted.
    widths = np.array([row["straight"]["width_mm"] for row in rows])
    heights = np.array([row["straight"]["height_mm"] for row in rows])
    lines += ["", "Item 5, undistorted: " + summary(rows, "straight"),
              "Item 6, raw (no undistortion): " + summary(rows, "raw"),
              f"Repeatability: width spread (SD) {widths.std():.2f} mm, "
              f"range {np.ptp(widths):.1f} mm; height SD "
              f"{heights.std():.2f} mm, range {np.ptp(heights):.1f} mm"]

    # %% Item 7: speed of the one command per photo (model already loaded).
    seconds = np.array([row["straight"]["seconds"] for row in rows])
    lines.append(f"Item 7, speed: median {np.median(seconds):.2f} s per "
                 f"photo, range {seconds.min():.2f}-{seconds.max():.2f} s, "
                 f"plus {load_seconds:.1f} s to load the model once")

    # %% Item 9: what each refusal threshold would remove, and what it costs.
    abs_mm, _ = errors(rows, "straight")
    per_photo = abs_mm.reshape(-1, 2).max(axis=1)
    scores = np.array([row["straight"]["confidence"] for row in rows])
    tilts = np.array([row["straight"]["tilt_deg"] for row in rows])
    edges = np.array([row["straight"]["marker_edge_px"] for row in rows])
    lines.append("Item 9, refusals (worst error per photo, in mm):")
    for name, values, limit, refuse in (
            ("confidence below", scores, limits[0], scores < limits[0]),
            ("tilt above", tilts, limits[1], tilts > limits[1]),
            ("marker edge below", edges, limits[2], edges < limits[2])):
        kept = per_photo[~refuse]
        kept_text = f"{kept.mean():.2f}" if kept.size else "n/a"
        lines.append(
            f"- {name} {limit}: refuses {refuse.sum()} of {len(rows)} "
            f"({100 * refuse.mean():.0f} %); observed {values.min():.3f}"
            f"-{values.max():.3f}; mean worst error kept {kept_text} mm")

    text = "\n".join(lines)
    print(text)
    os.makedirs("results", exist_ok=True)
    with open("results/accuracy.md", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
