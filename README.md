> Developed for [Google Summer of Code 2026](https://summerofcode.withgoogle.com/) under the DeepForest project (WeeCology).

# Recovering Bird Annotations from Historical Airborne Imagery

Between 2010 and 2021, surveyors counted birds in Gulf of Mexico aerial photographs using a point-counting tool. The tool drew a coloured dot on the photograph for every bird and saved a screenshot. It never saved the coordinates. What remains is 18,304 screenshots with the annotations baked into the pixels, and the clean high-resolution photographs they were drawn on.

This project reads the dots back out. It finds each dot, works out which class it belongs to, and maps it onto the original photograph as DeepForest training data.

![the problem](results/cell1_study_images.png)

Left: the screenshot, with the counting tool's dots and its dialog drawn over the photograph. Right: the same photograph, clean. Everything the pipeline needs is on the left, including the legend that says what each marker means.

Data source: `twi-aviandata.s3.amazonaws.com` (Gulf of Mexico avian monitoring, post-Deepwater Horizon).

## Status

The pipeline runs from the images alone. Survey counts are never an input; the CSV is used only to check the output.

| | Measured | On |
|---|---|---|
| Registration success | **96.7%**, 0.38px median reprojection error | 60 pairs |
| Detection precision, on frames the pipeline accepts | **0.74–1.00** | 541 hand-labelled dots |
| Dot placement error | **~1.5px** median | 541 hand-labelled dots |
| Classification, per dot | **0.544** | 215 matched dots |
| Tests | **167** passing, CI on Python 3.10 and 3.11 | |

**Detection is finished.** The important part is not a single accuracy figure. It is that every image now carries a check on whether its detections can be trusted, before anything is built from them.

A frame that reports far more dots than the image contains cannot be accurate, whatever is done downstream. That check needs no labels and no survey data, so it runs on every image. Pass rates were measured on the 63-pair benchmark and applied to the band distribution of all **18,252 screenshots**: about **48% of images pass, and they hold roughly 72% of the corpus's dots around 2 million annotations.** That is far more than training a DeepForest model needs, so the recovered set is large enough for the next stage to begin. See [Choosing frames](#choosing-frames).

Detection is scored against **hand-labelled dot positions**, not counts. The survey gives a count per image and never a coordinate, so until those labels existed there was no way to tell a pipeline that found the right dots from one that found the right *number* of dots in the wrong places. 541 dots across 8 frames are labelled in `data/labels/` and scored by `scripts/eval_localisation.py`.

Classification is the current focus.

![project progress](results/report_fig/fig_timeline.jpg)

Each point on the left is a measured change, not a re-tuning. Sparse frames were the worst case throughout: 63.51× over-detection under colour thresholds, 6.07× once the clean original was subtracted, 2.13× after the saturation floor. The dashed line on the right is what the same matching scores when it is tested against templates cut from its own pixels, which is why both numbers are reported.

![which frames are worth using](results/report_fig/fig_improvement.jpg)

Left: every frame sits under the ceiling its reported count implies, and the ones near 1× sit close to it. Right: the same frames ordered by that ratio, showing recall holding steady while precision splits. Both panels are drawn from `results/eval_localisation.csv` by `scripts/make_improvement_figure.py`, so they cannot fall behind the code.

## What the data looks like

Two weeks went into measuring the data before any pipeline code existed: 533K files mapped in the S3 bucket, 49,204 CSV rows across 60 columns analysed, 25 images downloaded and measured by hand.

![dataset analysis](results/cell2_forensic_analysis.png)

Laughing Gull is the most common species at 855K birds, not Brown Pelican. The median image holds 61 birds and 65% hold fewer than 100, so the four large study images used early on were never representative. 18 annotators, 7 years, 5 Gulf Coast states.

![dot properties](results/cell3_dot_properties.png)

Measured from 1,199 dots: median diameter 8.0px, circularity 0.57, six distinct colours. Circularity of 0.57 is what makes shape matching hard, because a dot at this size is barely a shape at all. These measurements set the colour-path thresholds and the template size.

## How it works

Every screenshot raises two separate questions, handled by different modules.

| Question | Modules |
|---|---|
| Where are the dots, and how many? | `align.py` → `subtract.py` |
| Which class is each dot? | `legend.py` → `classify.py` |

```mermaid
flowchart TB
    subgraph IN["Inputs"]
        SS["Screenshot<br/>dots baked into pixels"]
        ORIG["Clean original photograph"]
    end

    subgraph L["legend.py: read the dialog"]
        LOC["locate_dialog<br/>find the panel as a box"]
        PAR["parse_legend<br/>colour, shape, 24x24 template"]
        OCR["attach_class_names<br/>Tesseract + fuzzy match"]
        LOC --> PAR --> OCR
    end

    subgraph D["align.py + subtract.py: where the dots are"]
        SIFT["SIFT + RANSAC registration<br/>returns failure, not a bad warp"]
        DIFF["chromatic image difference"]
        SPLIT["distance transform<br/>splits merged dots"]
        SIFT --> DIFF --> SPLIT
    end

    subgraph C["classify.py: which class each dot is"]
        LAB["Lab colour anchoring<br/>to this image's palette"]
        NCC["NCC template match<br/>within the colour group"]
        LAB --> NCC
    end

    SS --> LOC
    SS --> SIFT
    ORIG --> SIFT
    SPLIT --> LAB
    OCR --> LAB
    NCC --> OUT["dot positions + class labels"]
    OUT --> MAP["map to original<br/>export DeepForest CSV"]

    CSV["Survey CSV"] -.->|"validation only,<br/>never an input"| OUT
```

### Reading the dialog

The counting tool leaves a floating panel somewhere in the frame listing every class, its marker and its count. `locate_dialog` finds that panel as a box, wherever it sits, and everything outside it stays aerial.

![dialog localisation](results/figures/fig1_localization.png)

Marker to class is a **per-image** mapping, not a global one, so the pipeline parses each dialog separately and cuts a 24×24 template from each row's own glyph.

![marker to class](results/figures/fig2_marker_to_class.png)

Compare the four dialogs and the reason is plain: a red plus means BRPE wbn in image B and BRPE bird in image D, and a circle means "site" in one image and "chick" in another. A global shape dictionary would get these wrong. Tesseract reads the class names, fuzzy-matched against the 98 species codes, which resolves 80 of the 82 rows shown. Regenerate with `python scripts/make_legend_figure.py`.

### Finding the dots

The clean original still exists, so a dot is whatever is present in the screenshot and absent from the photograph. `align.py` registers the two with SIFT and RANSAC, and returns a refusal instead of a bad warp when the match is poor. `subtract.py` takes the difference.

A colour threshold tests something different: whether a pixel falls inside a fixed band. Leaves, water glints and red map labels fall inside it too.

![before and after](results/report_fig/fig_beforeafter.jpg)

Dense colonies are the hardest case. Overlapping dots merge into a single blob, so a distance transform splits that blob back into individual markers.

![dense colony](results/report_fig/fig_dense.jpg)

Measured against hand-labelled dot positions, on the frames the pipeline accepts:

| frame | dots labelled | precision | recall | placement |
|---|---|---|---|---|
| `18May15…00825` | 10 | **1.00** | 1.00 | 0.04px |
| `19May18…00620` | 186 | **0.82** | 0.81 | 0.02px |
| `18June21…06389` | 103 | **0.74** | 0.53 | 1.26px |

Recall and placement hold up on every frame, accepted or not: 0.53 to 1.00, and about 1.5px median. What changes between frames is how much of the surroundings is detected alongside the markers, which is what the next section is about.

### Choosing frames

Not every image is worth using, and which ones can be worked out from the image alone.

The check is arithmetic. Correct detections can never outnumber the dots actually on the image, so precision is capped at `dots present / dots reported`. A frame reporting seven times what it should hold cannot exceed 14% precision, however it is filtered afterwards. Nothing about that needs labels.

Verified against hand labels on three frames spanning two density bands and 10 to 186 dots:

| ratio reported / present | precision |
|---|---|
| 1.00 | **1.00** |
| 0.99 | **0.82** |
| 0.62 | **0.74** |
| 5.63 | 0.14 |
| 7.35 | 0.07 |
| 14.22 | 0.04 |

Applying the benchmark's per-band pass rates to the band distribution of all 18,252 screenshots: **about 48% of images pass, holding roughly 72% of the corpus's dots.**

Frames that fail are mostly sparse scenes over textured ground. A mangrove colony with nine real markers returned 128 detections, because at this resolution a leaf and a marker are the same handful of pixels. Five ways of filtering those out were measured and none separated them; that is written up in [what did not work](#what-did-not-work). Excluding those frames costs little, because sparse images hold only 6% of the corpus's dots.

### Assigning classes

`classify.py` has to separate classes that share a colour, such as BRPE nest, bird and chick, all drawn in red. Each dot is matched to the colours that image's dialog actually uses, in Lab space, then matched by normalised cross-correlation against the templates cut from the dialog rows of that colour.

![classification](results/report_fig/fig_classify.jpg)

Each row shows the dialog marker, the template cut from it, and six aerial patches picked at random from everything assigned to that class. The figure is generated by `scripts/make_classify_figure.py` from the live pipeline, so it always shows current behaviour.

One caveat to read it correctly: this frame is a legacy fixture with no clean original beside it, so **detection here is the colour path, not subtraction**, which the figure's own title strip records. The bottom rows are the rare classes, and their patches are red map text and bare vegetation. Those are detection false positives reaching the matcher rather than matching errors, and subtraction removes most of them on frames that have an original to subtract.

Two measurements disagree, and both are reported:

- **Legend self-recovery, 76–83%.** A legend glyph is shrunk to aerial scale and pushed back through matching to see whether it recovers its own class. This flatters the method, because the template and the test glyph come from the same pixels.
- **Per-class count agreement on real aerial dots, 0.36.** Assigned counts per class compared against the counts read from the dialog, over 41 frames. Previous matching scored 0.26.

Both numbers appear here so that neither is mistaken for the other. The gap between them is how much of the 76–83% comes from testing the method against glyphs cut from its own templates.

## The benchmark

An earlier four-image set was small enough to overfit, and it used the wrong ground truth. Reading the counting tool's own *Total Count* field settled which column is correct: the dot count is `category_sum`, the sum of the per-class columns, not `total_birds`. `total_birds` excludes chicks and undercounts by up to 57%.

The current benchmark is **63 stratified pairs**: 7 years × 3 density bands × 3 images, across 40 colonies, with dot counts from 7 to 2,037. Density band means the number of annotation dots on the image, not the number of colours: sparse is 5–50, medium 51–300, dense 301 and above. Sampling by band keeps the dense tail in, because that is where detection was always weakest. Detection and alignment are scored on the 60 of those 63 whose screenshot and original are both cached locally.

### The ground truth needs checking too

One dense frame looked like a bad regression, 450 dots down to 9. Opening it showed why.

![ground truth artifact](results/report_fig/fig_artifact.jpg)

It is a "No photo coverage for this area" frame. There are no dots on it at all, only survey polygons and a text box. Its `category_sum` of 450 is a written estimate. The old detector scored 288 by counting polygon lines.

## Limitations

- **Classification is the weak half.** 0.544 per dot. Where two classes share a colour their templates score almost level — WHIB site against WHIB adult scored 0.548 to 0.540 on the same dot — and the labels then swap wholesale. Measured at 19 dots for that pair, plus 7 SNEG and 5 TRHE of the same kind.
- **The legend does not parse on 12% of frames.** Seven of 60 benchmark frames find no dialog at all, on some of which it is plainly visible. Classification is impossible there whatever the matcher does, and it is also what stands between the frame check and being fully self-contained.
- **The dense band has no hand labels.** Frame selection is verified on sparse and medium only. Dense holds 55% of the corpus's dots, so this is the first gap to close.
- **The accepted set skews dense** — 89% of dense frames pass the check against 28% of sparse. A training set built from it will see more crowded scenes than empty ones, and species that only appear on sparse frames will be under-represented.
- **Count OCR reads about 60–65%** of the Count column. The digits are around 10px tall. Class names read well at full resolution; the counts do not.

## Repository structure

```
src/legend.py       find the dialog, parse each row to marker, colour, shape, template, class, count
src/align.py        register the clean original onto the screenshot
src/subtract.py     annotations as image difference; turn ink into dot candidates
src/classify.py     Lab colour anchoring, then NCC shape matching within a colour
scripts/            benchmark builders, evaluation harnesses, figure generators
tests/              166 tests
notebook/           the earlier Colab prototype, stages 3 to 7
docs/learnings.md   what went wrong and why
```

`src/decompose.py` and `src/detect.py` are kept for reference and are not part of the live pipeline. `decompose.py` split each screenshot at roughly half its width, which discarded much of the aerial and the birds in it; `legend.locate_dialog` replaced it. `detect.py` needs CSV counts as an input, which breaks the rule that the pipeline works from the image alone.

## Quick start

```bash
pip install -r requirements.txt          # full
pip install pytest numpy opencv-python scikit-image PyYAML scipy   # tests only

pytest tests/ -q                         # 166 passing

python scripts/build_manifest.py         # survey counts, ~1.4 MB
python scripts/build_benchmark.py --per-cell 3   # the 63 pairs, ~472 MB
python scripts/eval_detection.py         # old vs new vs category_sum
python scripts/eval_alignment.py         # registration success rate
```

The tests need no data and no network; the fixtures they use are in the repo. Everything below `pytest` downloads from a public bucket, so no credentials are involved, but the pairs are large because the clean originals are about 7 MB each. `build_benchmark.py --dry-run` prints the selection and the download size without fetching anything. Both scripts cache, so an interrupted run can be repeated.

Class-name OCR needs the Tesseract binary, not just `pytesseract`. On Windows: `winget install UB-Mannheim.TesseractOCR`. The pipeline skips OCR and keeps parsing if Tesseract is missing.

All tunable parameters live in `config.yaml`.

## About this work

Built with guidance from the DeepForest maintainers and past contributors, who walked me through the open-source contribution process as well as the research. Before writing pipeline code I spent two weeks studying the data, and I would do that again: roughly 40% of the time on this project has been measurement rather than building, and almost every parameter in `config.yaml` traces back to something measured rather than guessed.

The pipeline has been through several detector versions, four training experiments, and a number of approaches I tested and dropped. I wrote up each dropped approach with its root cause, which turned out to matter: two of them had been abandoned for the wrong reason. OCR is the clearest case. I rejected it at 4% precision, but that test ran on a fixture 2.3× smaller than a real screenshot, and at full resolution it works.

The lesson I keep relearning is to check what is actually being measured before trusting the number. Detection was scored against the wrong CSV column for weeks, and the four-image test set was small enough that tuning against it looked like progress.

## The earlier prototype

`notebook/prototype_v1.ipynb` is a 23-section prototype that runs in Colab on a T4 GPU in about 45 minutes. Stages 3 to 7 still live there and have not yet been rewritten as modules: coordinate mapping, SAM 3 validation, DeepForest training and export.

![coordinate mapping](results/cell8_mapping.png)

Stage 4 is the one that matters next: dots read off the screenshot, placed onto the clean original. SIFT homography puts them within about 0.5px; the height-ratio fallback used when SIFT fails is roughly 30px out, and that error is what limited training.

Its numbers were measured differently from the ones above and are not comparable. Detection there was scored at 70.8% on 30 random images, against `total_birds` and using CSV counts as a pipeline input. Both of those are now known to be wrong: `total_birds` is the wrong column, and feeding the pipeline the answer it is meant to produce is not a fair test. The 63-pair benchmark replaced it.

What the prototype did establish still holds. Training on 920 SIFT-mapped annotations at 0.5px accuracy improved on pretrained DeepForest by 29% in max score, while training on all 3,851 annotations including entries with about 30px position error made the model worse. Position accuracy matters more than the amount of data, which is why registration quality is gated rather than assumed.

Details: [docs/training_analysis.md](docs/training_analysis.md).

## What did not work

| Approach | Result |
|---|---|
| Narrow HSV colour bins | 44% detection; dots vary more than expected |
| Text watermark filter | Removed real birds; colony rows look like text |
| Uniform `scale_x` for mapping | 23–39% horizontal stretch; use `scale_y` for both axes |
| Species-aware box sizes | Worse, because positions were the real problem |
| Training on the full dataset | 0 high-confidence detections |
| Colour filtering during detection | Cut one sparse frame from 129 detections to 1, against a true count of 9 |

Five more were measured against the hand labels, trying to separate real markers from background on the frames that over-detect. None of them worked, and they are recorded so the ground is not covered twice:

| Approach | Result |
|---|---|
| Raise the saturation floor | False positives are *more* saturated than true ones, 204 against 178. Moving the floor from 60 to 140 left precision flat at 0.29 and discarded 30% of the real markers. |
| Minimum blob area | The direction reverses between frames — real markers are larger on three, smaller on two. |
| Elongation | Real and false medians are both 1.00 on five frames of six. |
| Legend template match score | Reversed on one frame, where false positives scored 1.000 and real markers 0.218. |
| A learned filter over all four | Logistic regression with leave-one-frame-out validation gained 0.012 F1 while recall fell from 0.597 to 0.177. Its raw and per-frame-normalised weights take opposite signs, which is what a feature carrying no consistent signal looks like. |

The conclusion is that these frames cannot be cleaned by filtering what has already been detected, which is why the pipeline excludes them instead.

Full list: [docs/learnings.md](docs/learnings.md).
