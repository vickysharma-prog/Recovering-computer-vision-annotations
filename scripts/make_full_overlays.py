"""
Full-resolution overlays: every exported box on the whole photograph.

The summary figures crop to one window per frame so that twenty-five frames fit on a
page. That is useful for judging box *size* and useless for judging box *placement*:
it shows a few hundred boxes out of six thousand, in the one region that was chosen
for being crowded.

This writes the whole photograph, at full resolution, with every box on it, so the
boxes can be inspected anywhere in the frame by zooming in a browser rather than
trusting a crop.

Draws straight from `results/dataset/annotations_full.csv`, so it shows the exported
dataset and not a recomputation of it.

Output goes to `results/figures/full/`, which is local only — twenty-five 15-megapixel
JPEGs do not belong in the repository.

Usage:
    python scripts/make_full_overlays.py
    python scripts/make_full_overlays.py --only 5745 --quality 92

Then open results/figures/full/index.html.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
PAIRS = ROOT / "data" / "fixtures" / "pairs"
DATASET = ROOT / "results" / "dataset" / "annotations_full.csv"
OUT = ROOT / "results" / "figures" / "full"

GREEN = (60, 230, 60)


def draw(img: np.ndarray, g: pd.DataFrame, width: int) -> np.ndarray:
    """Every box, plus a dot at each centre so a box can be told from a marking."""
    out = img.copy()
    for _i, r in g.iterrows():
        cv2.rectangle(out, (int(r.xmin), int(r.ymin)), (int(r.xmax), int(r.ymax)),
                      GREEN, width)
        cv2.circle(out, (int(round(r.x_orig)), int(round(r.y_orig))), 1,
                   (255, 40, 40), -1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="substring of one frame name, to render just that one")
    ap.add_argument("--quality", type=int, default=88, help="JPEG quality")
    ap.add_argument("--width", type=int, default=2,
                    help="box line width in pixels at full resolution")
    args = ap.parse_args()

    if not DATASET.exists():
        raise SystemExit(f"{DATASET} not found. Run scripts/export_dataset.py first.")
    full = pd.read_csv(DATASET)
    e = full[full.exported]
    frames = sorted(e.frame.unique())
    if args.only:
        frames = [f for f in frames if args.only in f]
        if not frames:
            raise SystemExit(f"no exported frame matching {args.only!r}")

    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for name in frames:
        g = e[e.frame == name]
        src = PAIRS / g.image_path.iloc[0]
        img = cv2.imread(str(src))
        if img is None:
            print(f"  {name}: photograph not cached, skipped")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        out_path = OUT / f"{name}.jpg"
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(draw(img, g, args.width), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])

        box = float(g.box_px.iloc[0])
        bird = g.bird_px.iloc[0]
        note = (f"bird {bird:.1f}px &rarr; box {box:.0f}px"
                if bool(g.box_measured.iloc[0])
                else f"not measurable, box {box:.0f}px borrowed")
        cards.append((name, out_path.name, w, h, len(g), note,
                      out_path.stat().st_size // 1024))
        print(f"  {name:34s} {w}x{h}  {len(g):5d} boxes  "
              f"{out_path.stat().st_size // 1024:5d} KB")

    rows = "\n".join(
        f'<a class="card" href="{html.escape(fn)}" target="_blank">'
        f'<img src="{html.escape(fn)}" loading="lazy">'
        f'<div class="cap"><b>{html.escape(nm)}</b><br>{w}&times;{h} &middot; '
        f'{n} boxes &middot; {note} &middot; {kb} KB</div></a>'
        for nm, fn, w, h, n, note, kb in cards)

    (OUT / "index.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>Exported boxes, full resolution</title>
<style>
 body{{font:15px/1.5 system-ui,Segoe UI,sans-serif;margin:0;background:#111;color:#eee}}
 header{{padding:16px 22px;background:#1b1b1b;border-bottom:1px solid #333}}
 h1{{font-size:18px;margin:0 0 6px}} p{{margin:4px 0;color:#bbb;max-width:1100px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:16px;padding:18px 22px}}
 .card{{display:block;text-decoration:none;color:inherit;background:#1a1a1a;border:1px solid #333;border-radius:6px;overflow:hidden}}
 .card img{{width:100%;display:block;background:#000}}
 .cap{{padding:9px 11px;font-size:12.5px;color:#ccc}}
 code{{background:#222;padding:1px 5px;border-radius:3px}}
</style>
<header>
 <h1>Exported boxes, full resolution &mdash; {len(cards)} frames,
     {sum(c[4] for c in cards)} boxes</h1>
 <p><b>Click any frame</b> to open the full-resolution image in a new tab, then zoom
    with ctrl+scroll to inspect boxes anywhere in the photograph.</p>
 <p>Green boxes are exactly what <code>annotations_deepforest.csv</code> contains. The
    red dot at each centre is the recovered annotation position. Every box on the
    photograph is drawn &mdash; nothing is cropped away.</p>
</header>
<div class="grid">
{rows}
</div>
""", encoding="utf-8")

    print(f"\nwrote {len(cards)} overlays to {OUT}")
    print(f"open {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
