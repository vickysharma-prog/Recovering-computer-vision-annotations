"""
Recovery report for one Kaggle batch run, optionally against a baseline.

The question every experiment answers: of the dots the survey says were drawn, how
many did the pipeline recover, and where did the rest go? Frames are bucketed by the
stage that lost them, in pipeline order, so the buckets add up to the whole set.

Usage:
    python scripts/recovery_report.py b0
    python scripts/recovery_report.py e1 --vs b0
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGES = [  # (bucket, substring of excluded_reason), in pipeline order
    ("screenshot missing", "not cached"),
    ("error", "error:"),
    ("select: nothing detected", "nothing detected"),
    ("select: under-detects", "under-detects"),
    ("select: over-detects", "over-detects"),
    ("select: no usable count", "no usable count"),
    ("no dialog", "no dialog"),
    ("dialog wrong (named)", "dialog box wrong"),
    ("alignment failed", "alignment rejected"),
    ("wrong pair", "wrong pair"),
    ("legend did not parse", "legend did not parse"),
]


def load(tag: str, split: str) -> pd.DataFrame:
    f = pd.read_csv(os.path.join(ROOT, "results", "kaggle", tag, split, "frames.csv"))
    s = pd.read_csv(os.path.join(ROOT, "data", "cache", f"set_{split}.csv"))
    s["frame"] = s.screenshot_key.map(lambda k: os.path.splitext(os.path.basename(k))[0])
    f = f.drop(columns=["band"], errors="ignore").merge(
        s[["frame", "year", "band", "weight"]], on="frame", how="left")
    reason = f.excluded_reason.fillna("").astype(str)
    f["stage"] = "exported"
    for name, key in reversed(STAGES):
        f.loc[reason.str.contains(key, regex=False), "stage"] = name
    f["survey"] = f.survey_count.fillna(0)
    f["kept"] = f.dots_exported.fillna(0)
    return f


def funnel(f: pd.DataFrame) -> pd.DataFrame:
    total = f[f.stage != "screenshot missing"].survey.sum()
    rows = []
    for name in [s for s, _ in STAGES if s != "screenshot missing"] + ["exported"]:
        g = f[f.stage == name]
        if g.empty:
            continue
        lost = g.survey.sum() - g.kept.sum()
        rows.append(dict(stage=name, frames=len(g), survey_dots=int(g.survey.sum()),
                         recovered=int(g.kept.sum()), lost_pct=round(100 * lost / total, 1)))
    return pd.DataFrame(rows)


def recovery(f: pd.DataFrame, by: str) -> pd.DataFrame:
    f = f[f.stage != "screenshot missing"]
    g = f.groupby(by).agg(frames=("frame", "size"), survey=("survey", "sum"),
                          kept=("kept", "sum"),
                          yield_frames=("kept", lambda k: int((k > 0).sum())))
    g["recovery"] = (g.kept / g.survey).round(3)
    return g


def headline(f: pd.DataFrame) -> dict:
    # 7% of the manifest's screenshot keys are not in the bucket (HTTP 404, any
    # extension), so nothing can be recovered from them; they are reported, not scored.
    missing = int((f.stage == "screenshot missing").sum())
    f = f[f.stage != "screenshot missing"]
    w = f.weight.fillna(1)
    return dict(frames=len(f), screenshot_missing=missing,
                frames_with_dots=int((f.kept > 0).sum()),
                survey_dots=int(f.survey.sum()), recovered=int(f.kept.sum()),
                recovery=round(f.kept.sum() / f.survey.sum(), 3),
                corpus_weighted=round((w * f.kept).sum() / (w * f.survey).sum(), 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--vs", default=None, help="baseline tag to compare against")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    pd.set_option("display.width", 200)

    f = load(a.tag, a.split)
    print(f"== {a.tag} / {a.split}")
    print(pd.Series(headline(f)).to_string())
    print("\n-- where dots are lost, pipeline order\n" + funnel(f).to_string(index=False))
    print("\n-- by density band\n" + recovery(f, "band").to_string())
    print("\n-- by year\n" + recovery(f, "year").to_string())
    if "pairing" in f:
        print("\n-- title-bar pairing check\n" + f.pairing.value_counts(dropna=False).to_string())

    if a.vs:
        b = load(a.vs, a.split)
        m = f[["frame", "kept", "stage"]].merge(b[["frame", "kept", "stage"]], on="frame",
                                               suffixes=("", "_base"))
        print(f"\n== {a.tag} vs {a.vs}")
        print(pd.DataFrame([headline(b), headline(f)], index=[a.vs, a.tag]).to_string())
        d = m.kept - m.kept_base
        print(f"\nframes gained dots {(d > 0).sum()}, lost dots {(d < 0).sum()}, "
              f"unchanged {(d == 0).sum()};  net {int(d.sum()):+d} dots")
        print("worst losses:\n" + m.assign(delta=d).nsmallest(10, "delta").to_string(index=False))


if __name__ == "__main__":
    main()
