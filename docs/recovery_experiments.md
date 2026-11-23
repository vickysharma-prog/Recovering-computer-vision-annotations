# Recovery experiments, November 2026

The goal of this round was to recover more correct annotations from the screenshots.
Training a model was not the target. Every change was measured on the same frames
against a fixed baseline, and the final configuration was scored once on a sealed test
set that no change was ever tuned on.

## Data

5,000 frames drawn from the manifest: the 1,076 frames of the August scaled run plus
3,924 new ones, balanced across 7 survey years and 3 density bands (about 242 per cell).
They were split into 4,000 dev frames and 1,000 sealed test frames.

- 295 dev frames and 123 test frames have no screenshot in the S3 bucket (HTTP 404 for
  every file extension). They are left out of every denominator below.
- Runs happen on Kaggle: 5 CPU kernels, 4 export processes each, alignment cached after
  the first run. A full dev run takes about an hour.

Recovery means dots exported divided by the survey's dot count (`category_sum`).
"Surely false" counts dots exported above the survey count on a frame. They cannot all
be real, so the figure is a floor on false dots.

## Results

### Sealed test set (877 frames with a screenshot)

|                         | Baseline | Final  |
|-------------------------|----------|--------|
| Recovery                | 0.322    | 0.502  |
| Corpus-weighted         | 0.313    | 0.585  |
| Dots recovered          | 87,868   | 137,219|
| Frames giving any dots  | 279      | 642    |
| Surely-false dots       | 3,998    | 3,925  |

### Dev set, one change at a time (3,705 frames)

| Step | Change | Recovery | Surely false |
|------|--------|----------|--------------|
| B0 | baseline | 0.339 | 15,219 |
| E1 | wrong-pair guard | | 1 wrong frame removed |
| E2 | confidence tiers | 0.402 | |
| E3 | over-detecting frames judged on assigned dots, line filter | 0.473 | 25,470 |
| E4 | dialog box ranked by column share | 0.513 | 28,136 |
| E5 | marker size from the median blob area | 0.491 | 14,238 |
| E6 | Count column found from the markers, cleaned digit OCR | 0.484 | 8,373 |
| Final | all of the above, plus the tier D fix | 0.550 | 13,321 |

E5 and E6 lower the raw count because they remove false dots. Capped at the survey
count per frame, recovery goes 0.325, 0.477 (E6), 0.538 (final).

### Hand-labelled frames (12 frames, 1,648 dots)

|                | Baseline | Final |
|----------------|----------|-------|
| Precision      | 0.315    | 0.588 |
| Recall         | 0.630    | 0.749 |
| F1             | 0.420    | 0.658 |
| Classification | 0.703    | 0.780 |

The E5 and E6 methods were chosen while looking at these frames, though no constant was
fitted to them. The sealed test set is the out-of-sample check.

Placement error should only be read on the 2 frames labelled blind (`0115`, `0401`).
The other 10 were seeded from the old pipeline's own dots, so any change to dot centres
shows up as error there. On the blind frames placement improved: 2.1 to 1.8 px and 2.6
to 2.0 px.

## What each change does

**E1, wrong-pair guard** (`scripts/export_dataset.py`). Some screenshots are filed
against the wrong original: `17May11Camera1-Card3-0043` shows photo `543`. A frame is
dropped when the title-bar photo number disagrees with the file name and fewer than 60%
of SIFT matches are inliers. On 30 frames checked by eye this caught 10 of 13 wrong pairs
with no false positive. About 2% of frames are wrong pairs. Title OCR lives in
`src/title_check.py`.

**E2, confidence tiers** (`scripts/export_dataset.py`). A rejected frame used to give
nothing. Every exported dot now carries a tier:

- A: count ratio in band, legend read
- B: under-detects; only dots the legend accepts
- C: over-detects; judged again on assigned dots
- D: no dialog or no legend, but the ratio is in band; position only, labelled `Bird`

**E3, line filter** (`src/subtract.py`). Transect lines are drawn in the markers' own
colours, and their fragments passed as markers. Dots within 6 px of a component at least
40 px long and 6 times longer than wide are dropped. On the labelled frames, false
positives fell from 2,260 to 1,742 while 3 real dots were lost. `08May10Camera2-Card1-5124`
has no markers at all, yet 228 line fragments were exported from it before this filter.

**E4, dialog box** (`src/legend.py`). Sea and sand pass the flat-grey test, and a
colony on them can line up into a longer column than the real legend. Panels are now
ranked by the share of their coloured blobs that sit on the column. This added 43,836
dots on dev.

**E5, marker size** (`src/subtract.py`). The distance-transform peak measured the
stroke half-width of outline markers, about 5 times too small. The median blob area is
within 10% of the true size on 10 of 12 labelled frames. With fewer than 5 blobs it falls
back to the distance transform.

**E6, Count column** (`src/legend.py`). The Name|Count divider was taken by index, and
an extra gridline on the left shifted it onto the names. It is now the first gridline
right of the markers. The digit reader also misses clean digits, so a second reader
(borders and gridlines removed, three page-segmentation modes) fills the rows the first
leaves blank.

This also explains why moving the divider broke classification twice before. With the
old reader on the new strips, `5745` falls from 0.788 to 0.544. With the second reader it
stays at 0.778, and `0449` rises from 0.469 to 0.655.

## Measured and dropped

- Texture just outside the original's footprint as a wrong-pair signal: it reads 0.47
  to 0.78 on correct pairs as well.
- Legend glyph size as the marker size: off by 1.6 times, from 0.35 to 2.6 times
  depending on the frame.
- Lower percentiles of blob area as the marker size: they pick up noise on sparse frames.
- The cleaned digit reader on its own: where both readers return a number they disagree
  on 16% of rows, split evenly when checked by eye.

## Open

- **Dense colonies of thin outline markers** lose about 11,000 real dots under E5.
  Fused blobs pull the median above one marker. Only 2 of the 12 labelled frames are
  dense, so no estimator can be checked yet. A few more dense frames need hand labels.
- Count OCR still reads 74.5% of rows, and read counts sum exactly to the survey on 21%
  of frames.
- Tier D has no legend check and carries the most noise. Drop it where precision
  matters.
- The line filter misses some transect lines. On `18May11Camera1-Card4-0467` the final
  run still exports rows of boxes along two lines, because the line broke into pieces
  shorter than 40 px.

## Dataset

`results/dataset_5k/`, built by `scripts/build_dataset_5k.py` from the final run on both
splits: 742,690 boxes on 3,380 photographs. `image_path` is relative to
`data/fixtures/pairs`, as in `results/dataset/`. The files are gzipped because the plain
DeepForest CSV is about 100 MB, GitHub's limit for one file.

| file | contents |
|------|----------|
| `annotations_deepforest.csv.gz` | `image_path, xmin, ymin, xmax, ymax, label` |
| `annotations.csv.gz` | the same boxes with `split`, `tier`, `legend_row`, `species_resolved` |
| `frames.csv` | every frame and why it was or was not exported |

| split | tier A | B | C | D |
|-------|--------|---|---|---|
| dev   | 411,752 | 60,150 | 45,435 | 88,134 |
| test  | 99,183 | 12,420 | 10,268 | 15,348 |

## Figures

`scripts/make_recovery_figures.py` draws fig19 to fig23 from the runs above and reruns
`eval_localisation.py` for fig21.

## Reproducing

```bash
python scripts/kaggle_batch.py snapshot
python scripts/kaggle_batch.py push --tag final --split dev --chunks 5 --align data/cache/align_dev.json
python scripts/kaggle_batch.py pull --tag final --split dev --chunks 5
python scripts/recovery_report.py final --vs b0
```

The winning settings are the defaults. Each step can be switched off with its flag to
rebuild the ablation: `TIERS`, `OVER_ASSIGN`, `WRONG_PAIR_GUARD`, `LINE_FILTER`,
`MODAL=dt`, `DIALOG_SHARE`, `COUNT_GEOM`, `COUNT_OCR=orig`.
