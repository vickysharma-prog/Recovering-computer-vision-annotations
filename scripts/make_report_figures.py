"""
Compact detection figures for the mentor-facing report (embedded as data URIs).

Produces small JPEGs so the HTML stays light:
  1. before/after — colour flood vs subtraction, on one frame
  2. dense colony — subtraction recovers a packed colony
  3. ground-truth artifact — a 'no photo coverage' frame
"""
from __future__ import annotations
import os, sys
import cv2, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.align import align
from src.subtract import extract_annotations, dot_candidates
from src.classify import detect_dots
from src.legend import locate_dialog

PAIR_DIR = "data/fixtures/pairs"; OUT = "results/report_fig"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})

def read(k):
    p = os.path.join(PAIR_DIR, k.replace("avian_monitoring/", "")); img = cv2.imread(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

bench = pd.read_csv("data/cache/benchmark.csv")
by = {os.path.basename(r.screenshot_key): r for r in bench.itertuples()}

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=90, bbox_inches="tight", pad_inches=0.15,
                facecolor="white"); plt.close(fig)
    # recompress to JPEG for a small data URI
    img = cv2.imread(path)
    jpg = path.rsplit(".", 1)[0] + ".jpg"
    cv2.imwrite(jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    os.remove(path)
    print(f"  wrote {jpg}  ({os.path.getsize(jpg)//1024} KB)")

# ── 1. before / after ────────────────────────────────────────────────
name = "18May15Camera2-Card5-00762.jpg"          # medium, truth 64
r = by[name]; shot, orig = read(r.screenshot_key), read(r.highres_key)
try:
    bbox = locate_dialog(shot)
except Exception:
    bbox = None
old = detect_dots(shot, exclude=bbox)
res = align(shot, orig); sub = extract_annotations(shot, orig, res)
new = dot_candidates(sub)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for a in ax: a.imshow(shot); a.set_xticks([]); a.set_yticks([])
ax[0].scatter([d.cx for d in old], [d.cy for d in old], s=6, c="#e23b3b",
              alpha=0.5, linewidths=0)
ax[0].set_title(f"OLD  colour thresholds\n{len(old)} detected   (truth {int(r.dots)})  =  {len(old)/r.dots:.0f}x over",
                fontsize=12.5, color="#8a1c1c")
ax[1].scatter([c[0] for c in new], [c[1] for c in new], s=18,
              facecolors="none", edgecolors="#14e0a0", linewidths=1.1)
ax[1].set_title(f"NEW  subtract clean original\n{len(new)} detected   (truth {int(r.dots)})  =  {len(new)/r.dots:.2f}x",
                fontsize=12.5, color="#0b6b52")
fig.suptitle(f"Detection — same frame, two methods   ·   {name}", fontsize=13, y=1.02)
save(fig, "fig_beforeafter.png")

# ── 2. dense colony ──────────────────────────────────────────────────
name = "19June12Camera2-Card7-0216.jpg"          # dense, truth 1050
r = by[name]; shot, orig = read(r.screenshot_key), read(r.highres_key)
res = align(shot, orig); sub = extract_annotations(shot, orig, res)
new = dot_candidates(sub)
fig, ax = plt.subplots(figsize=(8, 5.6)); ax.imshow(shot)
ax.scatter([c[0] for c in new], [c[1] for c in new], s=9,
           facecolors="none", edgecolors="#14e0a0", linewidths=0.7)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"Dense colony — subtraction recovers packed dots\n"
             f"{len(new)} detected   ·   truth {int(r.dots)}   =   {len(new)/r.dots:.2f}x   ·   {name}",
             fontsize=12)
save(fig, "fig_dense.png")

# ── 3. ground-truth artifact ─────────────────────────────────────────
name = "16May15Camera2-Card1-00097.jpg"          # 'no photo coverage'
r = by[name]; shot, orig = read(r.screenshot_key), read(r.highres_key)
res = align(shot, orig); sub = extract_annotations(shot, orig, res)
new = dot_candidates(sub)
fig, ax = plt.subplots(figsize=(9, 5.6)); ax.imshow(shot)
ax.scatter([c[0] for c in new], [c[1] for c in new], s=26,
           facecolors="none", edgecolors="#14e0a0", linewidths=1.3)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"Ground-truth audit — 'No photo coverage for this area'\n"
             f"category_sum says {int(r.dots)} (an ESTIMATE in the text box) · actual dots on image ≈ 0 · we detect {len(new)}",
             fontsize=11.5)
save(fig, "fig_artifact.png")

# ── 4. recompress the classification template-match figure (Josh's ask) ─
src_fig = "results/figures/fig14_template_match_samples_D.png"
if os.path.exists(src_fig):
    img = cv2.imread(src_fig)
    h, w = img.shape[:2]; scale = min(1.0, 1100 / w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    cv2.imwrite(os.path.join(OUT, "fig_classify_D.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 75])
    print(f"  wrote fig_classify_D.jpg ({os.path.getsize(os.path.join(OUT,'fig_classify_D.jpg'))//1024} KB)")

print("done")
