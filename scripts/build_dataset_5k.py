"""
Package the final 5,000-frame run as a DeepForest dataset in results/dataset_5k/.

Reads the merged Kaggle output of `kaggle_batch.py pull --tag final` for both splits.
The CSVs are gzipped because the plain DeepForest file is about 100 MB, GitHub's
per-file limit; pandas and DeepForest read `.csv.gz` directly. `image_path` is
relative to data/fixtures/pairs, the same root_dir as results/dataset/.

    annotations_deepforest.csv.gz   image_path, xmin, ymin, xmax, ymax, label
    annotations.csv.gz              the same boxes with split, tier and legend row
    frames.csv                      every frame and why it was or was not exported

Tier A boxes come from frames whose detected count is within the selection band and
whose legend was read. B, C and D are lower confidence; see docs/recovery_experiments.md.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "dataset_5k")
KEEP = ["image_path", "xmin", "ymin", "xmax", "ymax", "label", "frame", "split", "tier",
        "legend_row", "species_resolved", "class_name", "x_orig", "y_orig", "box_px"]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ann, frames = [], []
    for split in ("dev", "test"):
        base = os.path.join(ROOT, "results", "kaggle", "final", split)
        a = pd.read_csv(os.path.join(base, "annotations_full.csv"), low_memory=False)
        ann.append(a[a.exported].assign(split=split))
        frames.append(pd.read_csv(os.path.join(base, "frames.csv")).assign(split=split))
    a = pd.concat(ann, ignore_index=True)[KEEP]
    for c in ("xmin", "ymin", "xmax", "ymax", "x_orig", "y_orig"):
        a[c] = a[c].round(1)
    a[KEEP[:6]].to_csv(os.path.join(OUT, "annotations_deepforest.csv.gz"), index=False)
    a.to_csv(os.path.join(OUT, "annotations.csv.gz"), index=False)
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(OUT, "frames.csv"), index=False)
    print(f"{len(a)} boxes on {a.image_path.nunique()} photographs")
    print(a.groupby(["split", "tier"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
