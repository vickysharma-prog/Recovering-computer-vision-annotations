"""
Figures for the November 2026 recovery experiments (docs/recovery_experiments.md).

Every number is read from, or recomputed by, the live pipeline:

* fig19, fig20 read the Kaggle runs pulled by `scripts/kaggle_batch.py pull` into
  results/kaggle/<tag>/<split>/frames.csv, through `scripts/recovery_report.py`.
* fig21 runs `scripts/eval_localisation.py` twice on the hand-labelled frames, once
  with every change switched off and once with the defaults.
* fig22 runs detection, dialog location and legend reading on three frames with each
  change switched off and on.

Usage:
    python scripts/make_recovery_figures.py              # all four
    python scripts/make_recovery_figures.py --skip-eval  # fig21 from the last run's cache
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "results", "figures")
CACHE = os.path.join(ROOT, "results", "kaggle", "fig21_eval.json")

BEFORE, AFTER = "#eb6834", "#2a78d6"          # validated pair (light surface)
SEG = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
OLD_ENV = dict(TIERS="0", OVER_ASSIGN="0", WRONG_PAIR_GUARD="0", LINE_FILTER="0",
               MODAL="dt", DIALOG_SHARE="0", COUNT_GEOM="0", COUNT_OCR="orig")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "axes.spines.top": False,
                     "axes.spines.right": False})


def recovery(tag: str, split: str) -> float:
    from recovery_report import headline, load
    return headline(load(tag, split))["recovery"]


def fig19() -> None:
    steps = [("B0", "b0", "baseline"), ("E2", "e2", "tiers"), ("E3", "e3", "line filter"),
             ("E4", "e4b", "dialog box"), ("E5", "e5b", "marker size"),
             ("E6", "e6", "count column"), ("Final", "final", "+ tier D fix")]
    vals = [recovery(t, "dev") for _, t, _ in steps]
    test = [recovery("b0", "test"), recovery("final", "test")]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3.2, 1]})
    x = np.arange(len(steps))
    a.bar(x, vals, width=0.62, color=AFTER)
    for i, v in enumerate(vals):
        a.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom", color=INK, fontsize=9)
    a.set_xticks(x, [f"{s}\n{d}" for s, _, d in steps], fontsize=8.5)
    a.set_ylim(0, 0.7); a.set_ylabel("dots recovered / survey dots")
    a.set_title("Dev set, one change at a time (3,705 frames)", loc="left", fontsize=10)
    b.bar([0, 1], test, width=0.6, color=[BEFORE, AFTER])
    for i, v in enumerate(test):
        b.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom", color=INK, fontsize=9)
    b.set_xticks([0, 1], ["before", "after"]); b.set_ylim(0, 0.7)
    b.set_title("Sealed test set (877 frames)", loc="left", fontsize=10)
    for ax in (a, b):
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig19_recovery_by_step.png"), dpi=150)
    print("fig19", [f"{v:.3f}" for v in vals], "test", [f"{v:.3f}" for v in test])


def fig20() -> None:
    from recovery_report import load
    groups = [("recovered", None),
              ("lost inside exported frames", None),
              ("frame rejected by count ratio", ("select:",)),
              ("no dialog / legend, alignment, wrong pair",
               ("no dialog", "legend did not parse", "alignment failed", "wrong pair",
                "dialog wrong"))]
    rows = {}
    for label, tag in (("baseline", "b0"), ("final", "final")):
        f = load(tag, "dev"); f = f[f.stage != "screenshot missing"]; tot = f.survey.sum()
        e = f[f.stage == "exported"]
        sel = f[f.stage.str.startswith("select:")].survey.sum()
        other = f[~f.stage.isin(["exported"]) & ~f.stage.str.startswith("select:")].survey.sum()
        rows[label] = [e.kept.sum() / tot, (e.survey.sum() - e.kept.sum()) / tot,
                       sel / tot, other / tot]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    for r, (label, parts) in enumerate(rows.items()):
        left = 0.0
        for k, p in enumerate(parts):
            ax.barh(r, p - 0.004, left=left, height=0.55, color=SEG[k])
            if p > 0.04:
                ax.text(left + p / 2, r, f"{p:.0%}", ha="center", va="center",
                        color="white" if k == 0 else INK, fontsize=9)
            left += p
    ax.set_yticks([0, 1], list(rows)); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("share of survey dots (dev, 3,705 frames)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in SEG],
              labels=[g for g, _ in groups], ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.32), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig20_where_dots_go.png"), dpi=150)
    print("fig20", {k: [round(v, 3) for v in p] for k, p in rows.items()})


def run_eval(env: dict) -> dict:
    out = subprocess.run([sys.executable, "scripts/eval_localisation.py"], cwd=ROOT,
                         env={**os.environ, **env}, capture_output=True, text=True).stdout
    p, r, f1 = map(float, re.search(r"precision ([\d.]+)\s+recall ([\d.]+)\s+F1 ([\d.]+)",
                                    out).groups())
    acc = float(re.search(r"per-dot accuracy, legend ROW\s+([\d.]+)", out).group(1))
    return dict(precision=p, recall=r, F1=f1, classification=acc)


def fig21(skip_eval: bool) -> None:
    if skip_eval and os.path.exists(CACHE):
        res = json.load(open(CACHE))
    else:
        res = {"before": run_eval(OLD_ENV), "after": run_eval({})}
        json.dump(res, open(CACHE, "w"), indent=1)
    keys = ["precision", "recall", "F1", "classification"]
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for off, (lab, col) in zip((-w / 2, w / 2), (("before", BEFORE), ("after", AFTER))):
        vals = [res[lab][k] for k in keys]
        ax.bar(x + off, vals, width=w - 0.02, color=col, label=lab)
        for i, v in enumerate(vals):
            ax.text(i + off, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, keys); ax.set_ylim(0, 1)
    ax.set_title("12 hand-labelled frames, 1,648 dots", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig21_labelled_accuracy.png"), dpi=150)
    print("fig21", res)


def _pair(name: str):
    """Screenshot, original and cached alignment for a dev or labelled frame."""
    import pandas as pd
    from scripts.export_dataset import cached_align
    from scripts.probe_groundtruth import fetch
    s = pd.read_csv(os.path.join(ROOT, "data", "cache", "set_dev.csv"))
    row = s[s.screenshot_key.str.contains(name)].iloc[0]
    cache = json.load(open(os.path.join(ROOT, "data", "cache", "align_dev.json")))
    shot, orig = fetch(row.screenshot_key), fetch(row.highres_key)
    return shot, orig, cached_align(row.screenshot_key, shot, orig, cache)


def _panel(img, title, w=560):
    img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))
    bar = np.full((34, w, 3), 255, np.uint8)
    cv2.putText(bar, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def fig22() -> None:
    import src.legend as L
    import src.subtract as S
    from src.classify import detect_dots_subtract
    rows = []

    # Line filter: a frame with no markers at all, only transect lines.
    shot, orig, res = _pair("08May10Camera2-Card1-5124")
    box = L.locate_dialog(shot)
    pans = []
    for flag, lab in ((False, "line filter off"), (True, "line filter on")):
        S._LINE_FILTER = flag
        dots = detect_dots_subtract(shot, orig, res, exclude=box)
        vis = cv2.cvtColor(shot, cv2.COLOR_RGB2BGR).copy()
        for d in dots:
            cv2.circle(vis, (int(d.cx), int(d.cy)), 7, (0, 0, 255), 2)
        pans.append(_panel(vis, f"{lab}: {len(dots)} dots (frame has no markers)"))
    rows.append(np.hstack(pans))

    # Dialog box: the old ranking picks a column of birds on the beach.
    shot, _o, _r = _pair("18May21Camera2-Card1-4762")
    pans = []
    for flag, lab in ((False, "old dialog box"), (True, "new dialog box")):
        L._DIALOG_SHARE = flag
        b = L.locate_dialog(shot)
        vis = cv2.cvtColor(shot, cv2.COLOR_RGB2BGR).copy()
        if b:
            x, y, w, h = b; cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)
        pans.append(_panel(vis, lab))
    rows.append(np.hstack(pans))

    # Count column: the dialog plainly reads 93 / 70 / 11 / 23 / 10.
    from scripts.export_dataset import read
    lab = json.load(open(os.path.join(ROOT, "data", "labels", "27May12Camera1-Card2-0449.json")))
    shot = read(lab["screenshot_key"])
    pans = []
    for geom, ocr, title in ((False, "orig", "old Count reading"),
                             (True, "comb", "new Count reading")):
        L._COUNT_GEOM, L._COUNT_OCR = geom, ocr
        ents, b = L.parse_screenshot(shot)
        x, y, w, h = b; dlg = shot[y:y + h, x:x + w].copy(); L.attach_class_names(dlg, ents)
        vis = cv2.cvtColor(dlg, cv2.COLOR_RGB2BGR)
        vis = cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        vis = cv2.copyMakeBorder(vis, 0, 0, 0, 170, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for e in ents:
            cv2.putText(vis, f"read {e.count}", (vis.shape[1] - 160, int(e.cy * 2) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2, cv2.LINE_AA)
        pans.append(_panel(vis, title))
        print("fig22 0449", title, [e.count for e in ents])
    h = max(p.shape[0] for p in pans)
    rows.append(np.hstack([cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                              value=(255, 255, 255)) for p in pans]))

    S._LINE_FILTER = L._DIALOG_SHARE = L._COUNT_GEOM = True
    L._COUNT_OCR = "comb"
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 6, 6, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(255, 255, 255)) for r in rows]
    cv2.imwrite(os.path.join(OUT, "fig22_examples.jpg"), np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("fig22 written")


FIG23 = [("17May21Camera1-Card1-4097", "frame the baseline dropped"),
         ("21June15Camera1-Card2-00654", "baseline exported too many boxes"),
         ("18June13Camera2-Card2-0081", "dialog box fixed")]


def fig23() -> None:
    """Exported boxes on the original photograph, baseline run against final run."""
    import pandas as pd
    from scripts.probe_groundtruth import fetch
    cols = ["frame", "image_path", "xmin", "ymin", "xmax", "ymax", "exported"]
    runs = {t: pd.read_csv(os.path.join(ROOT, "results", "kaggle", t, "dev",
                                        "annotations_full.csv"), usecols=cols)
            for t in ("b0", "final")}
    s = pd.read_csv(os.path.join(ROOT, "data", "cache", "set_dev.csv"))
    rows = []
    for name, why in FIG23:
        key = s[s.screenshot_key.str.contains(name)].highres_key.iloc[0]
        img = cv2.cvtColor(fetch(key), cv2.COLOR_RGB2BGR)
        survey = int(s[s.screenshot_key.str.contains(name)].dots.iloc[0])
        boxes = {t: r[(r.frame == name) & r.exported] for t, r in runs.items()}
        allb = pd.concat(boxes.values())
        if allb.empty:
            continue
        # Crop to where the boxes are, so they stay visible at report size.
        x0, y0 = int(max(0, allb.xmin.min() - 150)), int(max(0, allb.ymin.min() - 150))
        x1 = int(min(img.shape[1], allb.xmax.max() + 150))
        y1 = int(min(img.shape[0], allb.ymax.max() + 150))
        pans = []
        for t, col, lab in (("b0", (52, 104, 235), "baseline"), ("final", (214, 120, 42), "final")):
            vis = img.copy()
            for b in boxes[t].itertuples():
                cv2.rectangle(vis, (int(b.xmin), int(b.ymin)), (int(b.xmax), int(b.ymax)), col, 3)
            pans.append(_panel(vis[y0:y1, x0:x1],
                               f"{lab}: {len(boxes[t])} boxes, survey count {survey}", w=700))
        h = max(p.shape[0] for p in pans)
        pans = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                   value=(255, 255, 255)) for p in pans]
        head = np.full((30, 1400, 3), 255, np.uint8)
        cv2.putText(head, f"{name}: {why}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(np.vstack([head, np.hstack(pans)]))
        print("fig23", name, {t: len(b) for t, b in boxes.items()}, "survey", survey)
    cv2.imwrite(os.path.join(OUT, "fig23_boxes_before_after.jpg"), np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fig19(); fig20(); fig21(a.skip_eval); fig22(); fig23()


if __name__ == "__main__":
    main()
