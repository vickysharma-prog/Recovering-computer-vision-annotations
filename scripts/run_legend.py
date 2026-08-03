"""
Batch-run the dialog legend parser over a folder of screenshots.

Usage:
    python scripts/run_legend.py [input_glob]

Default input: data/fixtures/screenshots/*.png  (plus data/fixtures/sample_screenshot.png)
Outputs an annotated image per screenshot to results/legend/ and prints a
per-image summary table of the recovered (color, shape) markers.

Drop full-resolution screenshots into data/fixtures/screenshots/ and run this.
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.decompose import ScreenshotDecomposer
from src.legend import parse_legend

OUT_DIR = "results/legend"

# Distinct BGR colors for drawing row boxes.
_DRAW = {
    "red": (40, 40, 220), "orange": (40, 130, 240), "yellow": (40, 220, 230),
    "green": (60, 200, 60), "cyan": (220, 220, 40), "blue": (230, 90, 40),
    "magenta": (220, 60, 220), "grey": (150, 150, 150), "unknown": (80, 80, 80),
}


def annotate(dialog_rgb: np.ndarray, entries: list) -> np.ndarray:
    """Draw labeled boxes on each detected legend row + template strip."""
    dialog_bgr = cv2.cvtColor(dialog_rgb, cv2.COLOR_RGB2BGR)
    scale = max(1, int(round(420 / max(dialog_bgr.shape[1], 1))))
    canvas = cv2.resize(
        dialog_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    for e in entries:
        cx, cy = int(e.cx * scale), int(e.cy * scale)
        col = _DRAW.get(e.color, (80, 80, 80))
        r = max(6, int(5 * scale))
        cv2.rectangle(canvas, (cx - r, cy - r), (cx + r, cy + r), col, 1)
        cv2.putText(
            canvas, f"{e.row}:{e.color}/{e.shape}", (cx + r + 3, cy + 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA,
        )

    # Template strip on the right.
    if entries:
        T = entries[0].template.shape[0]
        strip = np.full((canvas.shape[0], T + 8, 3), 255, np.uint8)
        y = 4
        for e in entries:
            t = (e.template * 255).astype(np.uint8)
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            if y + T <= strip.shape[0]:
                strip[y:y + T, 4:4 + T] = t
                y += T + 4
        canvas = np.hstack([canvas, strip])
    return canvas


def main() -> None:
    args = sys.argv[1:]
    if args:
        paths = sorted(glob.glob(args[0]))
    else:
        paths = sorted(glob.glob("data/fixtures/screenshots/*.png"))
        extra = "data/fixtures/sample_screenshot.png"
        if os.path.exists(extra):
            paths.append(extra)

    if not paths:
        print("No screenshots found. Drop PNGs into data/fixtures/screenshots/")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    dec = ScreenshotDecomposer()

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"SKIP (unreadable): {path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = os.path.splitext(os.path.basename(path))[0]

        res = dec.decompose(img_rgb, expect_dialog=True)
        if res.dialog is None:
            print(f"{name}: no dialog detected ({img.shape[1]}x{img.shape[0]})")
            continue

        entries = parse_legend(res.dialog)
        print(f"\n=== {name}  ({img.shape[1]}x{img.shape[0]}, "
              f"dialog {res.dialog.shape[1]}x{res.dialog.shape[0]}, "
              f"{len(entries)} rows) ===")
        for e in entries:
            print(f"  row {e.row:2d}  {e.color:8s} {e.shape:9s} "
                  f"(y={e.cy:.0f}, hue={e.hue})")

        out = os.path.join(OUT_DIR, f"{name}_legend.png")
        cv2.imwrite(out, annotate(res.dialog, entries))
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
