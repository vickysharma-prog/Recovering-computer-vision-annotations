"""
Publish the current pipeline results to a Comet ML project.

This project has no training loop: detection and classification are OpenCV, and
the numbers come from evaluation runs rather than from epochs. So this logs one
Comet experiment per evaluation, reading the CSVs the eval scripts already
write. Nothing in src/ changes.

What each run records:
  parameters : every value in config.yaml, flattened to section.key
  metrics    : detection ratio per density band, alignment success and error,
               classification agreement before and after the matching rework
  assets     : the per-image eval CSVs, so a number can be traced to a frame
  images     : the report figures

Needs COMET_API_KEY in the environment. Without it the script explains how to
set one and exits 0, so it never breaks a pipeline or CI run.

Usage:
    python scripts/log_to_comet.py
    python scripts/log_to_comet.py --project bird-annotation-recovery --dry-run

Output: a run at comet.com/<workspace>/<project>
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
    ("fig_improvement", RESULTS / "report_fig" / "fig_improvement.jpg"),
    ("fig_beforeafter", RESULTS / "report_fig" / "fig_beforeafter.jpg"),
    ("fig_dense", RESULTS / "report_fig" / "fig_dense.jpg"),
    ("fig_artifact", RESULTS / "report_fig" / "fig_artifact.jpg"),
    ("fig_classify", RESULTS / "report_fig" / "fig_classify.jpg"),
    ("fig1_localization", RESULTS / "figures" / "fig1_localization.png"),
    ("fig2_marker_to_class", RESULTS / "figures" / "fig2_marker_to_class.png"),
]


def _has_credentials() -> bool:
    """True if Comet can find a key, without this script ever reading it."""
    if os.environ.get("COMET_API_KEY"):
        return True
    return any((base / name).exists()
               for base in (ROOT, Path.home())
               for name in (".comet.config", "comet.config"))


def _flatten(d: dict, prefix: str = "") -> dict:
    """config.yaml is two levels deep; Comet wants flat key/value pairs."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        elif isinstance(v, (list, tuple)):
            out[key] = str(v)
        else:
            out[key] = v
    return out


def collect() -> tuple[dict, dict, list[Path]]:
    """Read the eval outputs. Returns (params, metrics, assets)."""
    params: dict = {}
    metrics: dict = {}
    assets: list[Path] = []

    cfg = ROOT / "config.yaml"
    if cfg.exists():
        params.update(_flatten(yaml.safe_load(cfg.read_text(encoding="utf-8"))))

    det = RESULTS / "eval_detection.csv"
    if det.exists():
        d = pd.read_csv(det)
        assets.append(det)
        metrics["detection/frames"] = len(d)
        metrics["detection/old_ratio_median"] = float(d.old_ratio.median())
        metrics["detection/new_ratio_median"] = float(d.new_ratio.median())
        for band, g in d.groupby("band"):
            metrics[f"detection/old_ratio_{band}"] = float(g.old_ratio.median())
            metrics[f"detection/new_ratio_{band}"] = float(g.new_ratio.median())
        # Symmetric error: |log2(ratio)| is 0 for a perfect count and treats
        # over- and under-detection alike, unlike the raw ratio.
        for col in ("old_ratio", "new_ratio"):
            r = d[col].replace(0, np.nan)
            metrics[f"detection/symlog_{col}"] = float(np.abs(np.log2(r)).median())

    ali = RESULTS / "eval_alignment.csv"
    if ali.exists():
        a = pd.read_csv(ali)
        assets.append(ali)
        metrics["alignment/frames"] = len(a)
        metrics["alignment/success_rate"] = float(a.ok.mean())
        if a.ok.any():
            metrics["alignment/reproj_px_median"] = float(a[a.ok].reproj.median())

    ab_dir = RESULTS / "classify_ab"
    old_f, new_f = ab_dir / "ab_old.csv", ab_dir / "ab_new.csv"
    if old_f.exists() and new_f.exists():
        old, new = pd.read_csv(old_f), pd.read_csv(new_f)
        assets += [old_f, new_f]
        metrics["classification/frames"] = len(new)
        for tag, df in (("old", old), ("new", new)):
            metrics[f"classification/{tag}_agreement_mean"] = float(df.agree.mean())
            metrics[f"classification/{tag}_agreement_median"] = float(df.agree.median())
            for band, g in df.groupby("band"):
                if band == "?":       # frame is not in the 63-pair benchmark
                    continue
                metrics[f"classification/{tag}_agreement_{band}"] = float(g.agree.mean())
        merged = old.merge(new, on="name", suffixes=("_old", "_new"))
        metrics["classification/frames_improved"] = int((merged.agree_new > merged.agree_old).sum())
        metrics["classification/frames_worse"] = int((merged.agree_new < merged.agree_old).sum())

    return params, metrics, assets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="bird-annotation-recovery")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--name", default=None, help="run name shown in the dashboard")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be logged and exit")
    args = ap.parse_args()

    params, metrics, assets = collect()
    if not metrics:
        print("No eval results found. Run scripts/eval_detection.py and "
              "scripts/eval_alignment.py first.")
        return 1

    if args.dry_run:
        print(f"{len(params)} parameters, {len(metrics)} metrics, "
              f"{len(assets)} assets, "
              f"{sum(1 for _, p in FIGURES if p.exists())} figures")
        for k in sorted(metrics):
            print(f"  {k:44s} {metrics[k]}")
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
    # log_git_patch would upload every untracked file, which here means the
    # local image fixtures: megabytes of data that add nothing to a run. The
    # git metadata (commit, branch) is kept, since that is what ties a run to
    # the code that produced it.
    exp = Experiment(project_name=args.project, workspace=args.workspace,
                     auto_metric_logging=False, auto_param_logging=False,
                     log_git_patch=False)
    if args.name:
        exp.set_name(args.name)

    exp.log_parameters(params)
    exp.log_metrics(metrics)
    for path in assets:
        exp.log_asset(str(path), file_name=path.name)
    for name, path in FIGURES:
        if path.exists():
            exp.log_image(str(path), name=name)
    for doc in ("README.md", "docs/progress_report.md", "docs/learnings.md"):
        p = ROOT / doc
        if p.exists():
            exp.log_asset(str(p), file_name=Path(doc).name)

    exp.add_tags(["opencv", "no-training", "benchmark-63-pairs"])
    exp.end()
    print(f"logged {len(params)} parameters and {len(metrics)} metrics to "
          f"project '{args.project}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
