"""
Legend figure: the dialog as found, and what each row parses to.

Shows the per-image marker-to-class mapping, which is the reason the pipeline
parses every dialog separately instead of using one global shape dictionary.
The same marker means different things in different images.

Runs the live path, so the figure always shows current behaviour:
    locate_dialog -> parse_legend -> attach_class_names

Usage:
    python scripts/make_legend_figure.py

Output: results/figures/fig2_marker_to_class.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.legend import parse_screenshot, attach_class_names

ROOT = Path(__file__).parent.parent
SCREENSHOTS = ROOT / "data" / "fixtures" / "screenshots"
OUT = ROOT / "results" / "figures"

KEYS = ["A_felicity_2012", "B_gaillard_2011", "C_northdeer_2010", "D_raccoon_2011"]

# One colour per marker colour name, for the parsed text lines.
INK = {"red": "#cc2222", "yellow": "#b08800", "green": "#118844",
       "cyan": "#0d9caa", "blue": "#2244cc", "magenta": "#bb22aa",
       "grey": "#666666", "unknown": "#999999"}


def _read_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 11),
                             gridspec_kw={"width_ratios": [1, 1.15, 1, 1.15]})
    total_rows = total_named = 0

    for i, key in enumerate(KEYS):
        path = next(iter(SCREENSHOTS.glob(f"{key}.*")), None)
        if path is None:
            raise SystemExit(f"no screenshot for {key}")
        rgb = _read_rgb(path)
        entries, bbox = parse_screenshot(rgb)
        dialog = rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        attach_class_names(dialog, entries)

        named = sum(1 for e in entries if e.class_name)
        total_rows += len(entries)
        total_named += named

        r, c = divmod(i, 2)
        ax_img, ax_txt = axes[r, c * 2], axes[r, c * 2 + 1]

        ax_img.imshow(dialog)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.set_title(f"{key}\ndialog found at {bbox[2]}x{bbox[3]}px",
                         fontsize=10)

        ax_txt.axis("off")
        ax_txt.set_title(f"parsed: {named}/{len(entries)} rows named",
                         fontsize=10, loc="left")
        # Keep the text block readable when a dialog carries 25+ rows.
        step = 1.0 / max(len(entries) + 1, 14)
        size = 9 if len(entries) <= 18 else 7.5
        for j, e in enumerate(entries):
            name = e.class_name if e.class_name else "(name not read)"
            count = f"  [{e.count}]" if e.count is not None else ""
            shape = e.shape or "?"
            ax_txt.text(0.0, 1.0 - (j + 1) * step,
                        f"{e.color}/{shape}  ->  {name}{count}",
                        fontsize=size, family="DejaVu Sans Mono",
                        color=INK.get(e.color, "#333333"),
                        transform=ax_txt.transAxes, va="top")

    fig.suptitle(
        "Marker to class is read per image, not assumed\n"
        f"locate_dialog + parse_legend + attach_class_names on the live path   "
        f"·   {total_named}/{total_rows} rows named across the four dialogs",
        fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "fig2_marker_to_class.png"
    fig.savefig(out, dpi=95, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)  "
          f"named {total_named}/{total_rows}")


if __name__ == "__main__":
    build()
