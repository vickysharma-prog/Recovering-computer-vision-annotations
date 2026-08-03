"""
Publish the pipeline's evaluation results to a Comet ML project.

This project has no training loop. Detection and classification are OpenCV, and
the numbers come from evaluation runs rather than epochs, so the usual
DeepForest pattern of handing a CometLogger to trainer.fit does not apply here.

Experiments written:
    project timeline    the measured history, one step per dated milestone.
                        Open this one first: it carries the progression
                        charts, the figures, the milestone table, and the
                        list of approaches that were tried and dropped.
    detection: colour thresholds \
    detection: subtraction        the two detection methods, summary metrics
    classification: previous     \
    classification: current       the two matching configurations

The four method runs share metric names, so selecting them together gives a
comparison table with one row each. They are deliberately single-valued:
logging all 60 frames as steps renders as a scatter of dots and is unreadable
beside a run with a different number of steps. Pass --per-frame when the
distribution is what you actually want to inspect.

Every run carries config.yaml as parameters, so a config can be tied to a
result.

Needs COMET_API_KEY or a gitignored .comet.config. Without credentials it
explains how to set them and exits 0, so it can never break a pipeline or CI.

Usage:
    python scripts/log_to_comet.py
    python scripts/log_to_comet.py --dry-run
    python scripts/log_to_comet.py --clean       # archive earlier runs first
    python scripts/log_to_comet.py --per-frame   # add the noisy detail

Output: runs at comet.com/<workspace>/<project>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

FIGURES = [
    ("01_what_changed", RESULTS / "report_fig" / "fig_improvement.jpg"),
    ("02_detection_before_after", RESULTS / "report_fig" / "fig_beforeafter.jpg"),
    ("03_dense_colony", RESULTS / "report_fig" / "fig_dense.jpg"),
    ("04_ground_truth_artifact", RESULTS / "report_fig" / "fig_artifact.jpg"),
    ("05_classification", RESULTS / "report_fig" / "fig_classify.jpg"),
    ("06_dialog_localization", RESULTS / "figures" / "fig1_localization.png"),
    ("07_marker_to_class", RESULTS / "figures" / "fig2_marker_to_class.png"),
]

BAND_ORDER = {"sparse": 0, "medium": 1, "dense": 2, "?": 3}

# The project's measured history, transcribed from the dated entries in
# docs/progress_report.md. Only numbers that were actually recorded on that
# date appear here; a metric that was not measured at a milestone is left out
# rather than carried forward or guessed.
#
# Logged as one experiment with the milestone index as the step, so the charts
# show the progression instead of only the latest value.
MILESTONES = [
    dict(date="2026-06-30", label="legend module (PR #3)",
         note="per-image marker to class parsing; dialog found as a box, 14/14",
         tests=143),
    dict(date="2026-07-01", label="count-OCR + count-prior",
         note="count-prior w=0.9 later retracted: it gamed the metric it was scored on",
         tests=146),
    dict(date="2026-07-10", label="matching rework",
         note="Lab colour anchoring, background removal, NCC; shape-name boost removed by ablation",
         tests=146, selfrec_d=0.76, selfrec_a=0.83),
    dict(date="2026-07-20", label="ground truth corrected",
         note="dot count is category_sum, not total_birds; benchmark rebuilt as 63 stratified pairs",
         tests=163,
         det_median=8.403, det_sparse=63.51, det_medium=9.148, det_dense=3.56,
         det_symlog=3.071),
    dict(date="2026-07-20", label="difference-based detection",
         note="align.py + subtract.py; annotations as image difference instead of colour thresholds",
         tests=163, align=0.967,
         det_median=1.46, det_sparse=6.07, det_medium=1.42, det_dense=1.01,
         det_symlog=0.73),
    dict(date="2026-07-24", label="saturation floor",
         note="gate low-saturation ink before the marker-size estimate; every band improved",
         tests=164, align=0.967,
         det_median=1.244, det_sparse=2.132, det_medium=1.245, det_dense=1.138,
         det_symlog=0.527),
    dict(date="2026-08-03", label="wired + measured on 41 frames",
         note="subtraction feeds the classifier; classification A/B over 41 frames",
         tests=166, align=0.967,
         det_median=1.244, det_sparse=2.132, det_medium=1.245, det_dense=1.138,
         det_symlog=0.527, cls_mean=0.357, cls_prev=0.263),
]

# Approaches that were tested and dropped, with the number that killed each.
# From docs/learnings.md and the progress report.
ABANDONED = [
    ("OCR on the dialog box", "4% precision, later found to be a downscaled-fixture artifact"),
    ("narrow HSV colour bins", "44% detection; dots vary more than the bins allowed"),
    ("text watermark filter", "removed real birds; colony rows look like text"),
    ("dialog colour clusters by position", "1 of 12 mappings correct (8%)"),
    ("count-prior w=0.9", "retracted: gamed the attributable metric, no real gain"),
    ("colour filtering during detection", "one sparse frame went from 129 detections to 1, true count 9"),
    ("bounding-box fill for chrome", "fixed dense, broke sparse from 0.98x to 9.85x"),
    ("species-aware box sizes", "worse; the real problem was 30px position error"),
    ("training on the full dataset", "0 high-confidence detections"),
]


def _has_credentials() -> bool:
    """True if Comet can find a key, without this script ever reading it."""
    if os.environ.get("COMET_API_KEY"):
        return True
    return any((base / name).exists()
               for base in (ROOT, Path.home())
               for name in (".comet.config", "comet.config"))


def _flatten(d: dict, prefix: str = "") -> dict:
    """config.yaml is nested; Comet wants flat key/value pairs."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        elif isinstance(v, (list, tuple)):
            out[key] = str(v)
        else:
            out[key] = v
    return out


def _params() -> dict:
    cfg = ROOT / "config.yaml"
    if not cfg.exists():
        return {}
    return _flatten(yaml.safe_load(cfg.read_text(encoding="utf-8")))


def _alignment() -> dict:
    """Alignment is a property of the run, not of a method, so it rides along."""
    path = RESULTS / "eval_alignment.csv"
    if not path.exists():
        return {}
    a = pd.read_csv(path)
    out = {"alignment_success_rate": float(a.ok.mean()),
           "alignment_frames": float(len(a))}
    if a.ok.any():
        out["alignment_reproj_px_median"] = float(a[a.ok].reproj.median())
    return out


def build_runs() -> list[dict]:
    """One entry per experiment: name, tags, per-frame series, summary metrics."""
    runs: list[dict] = []
    align = _alignment()

    det_path = RESULTS / "eval_detection.csv"
    if det_path.exists():
        d = pd.read_csv(det_path).copy()
        d["_o"] = d.band.map(lambda b: BAND_ORDER.get(b, 9))
        d = d.sort_values(["_o", "name"]).reset_index(drop=True)
        for label, col, tag in (("colour thresholds", "old_ratio", "old"),
                                ("subtraction", "new_ratio", "new")):
            r = d[col].replace(0, np.nan)
            summary = {
                "detection_ratio_median": float(d[col].median()),
                # |log2 ratio| is 0 for a perfect count and penalises
                # under- and over-detection equally, unlike the raw ratio.
                "detection_symlog_error": float(np.abs(np.log2(r)).median()),
                "frames": float(len(d)),
            }
            for band, g in d.groupby("band"):
                summary[f"detection_ratio_{band}"] = float(g[col].median())
            summary.update(align)
            runs.append({
                "name": f"detection: {label}",
                "tags": ["detection", tag],
                "series": [("detection_ratio", d[col].tolist())],
                "labels": d.name.tolist(),
                "summary": summary,
                "assets": [det_path, RESULTS / "eval_alignment.csv"],
                "figures": False,
            })

    ab = RESULTS / "classify_ab"
    old_f, new_f = ab / "ab_old.csv", ab / "ab_new.csv"
    if old_f.exists() and new_f.exists():
        old, new = pd.read_csv(old_f), pd.read_csv(new_f)
        merged = old.merge(new, on="name", suffixes=("_old", "_new"))
        improved = int((merged.agree_new > merged.agree_old).sum())
        worse = int((merged.agree_new < merged.agree_old).sum())
        for label, df, tag in (("previous matching", old, "old"),
                               ("current matching", new, "new")):
            df = df.copy()
            df["_o"] = df.band.map(lambda b: BAND_ORDER.get(b, 9))
            df = df.sort_values(["_o", "name"]).reset_index(drop=True)
            summary = {
                "classification_agreement_mean": float(df.agree.mean()),
                "classification_agreement_median": float(df.agree.median()),
                "frames": float(len(df)),
                "frames_improved_vs_previous": float(improved),
                "frames_worse_vs_previous": float(worse),
            }
            for band, g in df.groupby("band"):
                if band == "?":          # frame is not in the 63-pair benchmark
                    continue
                summary[f"classification_agreement_{band}"] = float(g.agree.mean())
            runs.append({
                "name": f"classification: {label}",
                "tags": ["classification", tag],
                "series": [("classification_agreement", df.agree.tolist())],
                "labels": df.name.tolist(),
                "summary": summary,
                "assets": [old_f, new_f],
                "figures": False,
            })

    return runs


MILESTONE_METRICS = {
    "det_median": "detection_ratio_median",
    "det_sparse": "detection_ratio_sparse",
    "det_medium": "detection_ratio_medium",
    "det_dense": "detection_ratio_dense",
    "det_symlog": "detection_symlog_error",
    "align": "alignment_success_rate",
    "tests": "tests_passing",
    "cls_mean": "classification_agreement_mean",
    "selfrec_d": "classification_selfrecovery_D",
    "selfrec_a": "classification_selfrecovery_A",
}


def log_timeline(Experiment, project: str, workspace: str | None,
                 params: dict) -> None:  # noqa: C901
    """One experiment holding the whole measured history, step = milestone."""
    exp = Experiment(project_name=project, workspace=workspace,
                     auto_metric_logging=False, auto_param_logging=False,
                     log_git_patch=False)
    exp.set_name("project timeline (measured history)")
    exp.add_tags(["timeline", "history", "opencv"])
    exp.log_parameters(params)

    for step, m in enumerate(MILESTONES):
        for key, metric in MILESTONE_METRICS.items():
            if key in m:
                exp.log_metric(metric, float(m[key]), step=step)
        exp.log_other(f"step_{step}", f"{m['date']}  {m['label']}")

    exp.log_table("milestones.csv", pd.DataFrame([
        {"step": i, "date": m["date"], "milestone": m["label"],
         "detection_ratio_median": m.get("det_median"),
         "detection_ratio_sparse": m.get("det_sparse"),
         "tests": m.get("tests"), "what_changed": m["note"]}
        for i, m in enumerate(MILESTONES)]))
    exp.log_table("abandoned_approaches.csv", pd.DataFrame(
        ABANDONED, columns=["approach", "why it was dropped"]))
    for name, path in FIGURES:
        if path.exists():
            exp.log_image(str(path), name=name)
    for doc in ("README.md", "docs/progress_report.md", "docs/learnings.md"):
        p = ROOT / doc
        if p.exists():
            exp.log_asset(str(p), file_name=Path(doc).name)
    exp.end()
    print(f"  logged project timeline ({len(MILESTONES)} milestones)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="bird-annotation-recovery")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be logged and exit")
    ap.add_argument("--clean", action="store_true",
                    help="archive existing runs in the project first")
    ap.add_argument("--per-frame", action="store_true",
                    help="also log every frame as a step; useful for digging "
                         "into the distribution, too noisy for a shared view")
    args = ap.parse_args()

    runs = build_runs()
    if not runs:
        print("No eval results found. Run scripts/eval_detection.py and "
              "scripts/eval_alignment.py first.")
        return 1

    params = _params()
    if args.dry_run:
        print(f"{len(runs)} experiments, {len(params)} parameters each\n")
        for r in runs:
            pts = sum(len(v) for _, v in r["series"])
            print(f"  {r['name']:38s} {pts:3d} points, "
                  f"{len(r['summary'])} summary metrics")
            for k in sorted(r["summary"]):
                print(f"      {k:38s} {r['summary'][k]:.4f}")
        return 0

    if not _has_credentials():
        print("No Comet credentials found, so nothing was uploaded.\n"
              "Get a key from comet.com (Account settings -> API keys), then use\n"
              "either of these. Both keep the key out of the shell history and\n"
              "out of git.\n\n"
              "  1. Write .comet.config in the project root (already gitignored):\n"
              "       [comet]\n"
              "       api_key = your-key\n"
              "       workspace = your-workspace\n\n"
              "  2. Or set the environment variable in your own terminal:\n"
              '       PowerShell : $env:COMET_API_KEY = "your-key"\n'
              "       bash       : export COMET_API_KEY=your-key\n\n"
              "Re-run with --dry-run to see the numbers without uploading.")
        return 0

    from comet_ml import Experiment

    if args.clean:
        from comet_ml.api import API
        api = API()
        ws = args.workspace or api.get_default_workspace()
        try:
            existing = api.get(ws, args.project) or []
        except Exception:
            existing = []
        for e in existing:
            api.archive_experiment(e.id)
        if existing:
            print(f"archived {len(existing)} earlier run(s)")

    log_timeline(Experiment, args.project, args.workspace, params)
    per_frame = args.per_frame

    for r in runs:
        # log_git_patch would upload every untracked file, which here means the
        # local image fixtures. Git metadata (commit, branch) is kept, since
        # that is what ties a run to the code that produced it.
        exp = Experiment(project_name=args.project, workspace=args.workspace,
                         auto_metric_logging=False, auto_param_logging=False,
                         log_git_patch=False)
        exp.set_name(r["name"])
        exp.add_tags(r["tags"] + ["opencv", "no-training", "benchmark-63-pairs"])
        exp.log_parameters(params)

        # Step is the frame index, ordered sparse then medium then dense, so
        # the chart reads left to right as the images get more crowded.
        if per_frame:
            for metric, values in r["series"]:
                for i, v in enumerate(values):
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        exp.log_metric(metric, float(v), step=i)
        exp.log_metrics(r["summary"])
        exp.log_other("frame_order", "sparse, then medium, then dense")

        for path in r["assets"]:
            if path.exists():
                exp.log_table(str(path))
        if r["figures"]:
            for name, path in FIGURES:
                if path.exists():
                    exp.log_image(str(path), name=name)
            for doc in ("README.md", "docs/progress_report.md",
                        "docs/learnings.md"):
                p = ROOT / doc
                if p.exists():
                    exp.log_asset(str(p), file_name=Path(doc).name)
        exp.end()
        print(f"  logged {r['name']}")

    print(f"\n{len(runs) + 1} experiments in project '{args.project}': "
          f"the timeline, plus {len(runs)} method runs.\n"
          "Open 'project timeline' for the history, or select the method runs "
          "together for the comparison view.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
