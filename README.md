> Developed for [Google Summer of Code 2026](https://summerofcode.withgoogle.com/) under the DeepForest project (WeeCology).

# Weecology : Recovering Bird Annotations from Historical Airborne Imagery

Between 2010 and 2021, surveyors counted birds in Gulf of Mexico aerial photographs using a point-counting tool. The tool drew a coloured dot on the photograph for every bird and saved a screenshot. It never saved the coordinates. What remains is **18,304 screenshots** carrying 2.81 million bird observations baked into the pixels, and the clean high-resolution photographs they were drawn on.

This project reads them back out. It finds each dot, determines its species and category from the image's own legend, maps it onto the original photograph, and exports DeepForest-ready training data.

![the problem](results/cell1_study_images.png)

Left: the screenshot, with the counting tool's dots and its dialog drawn over the photograph. Right: the same photograph, clean. Everything the pipeline needs is on the left, including the legend that says what each marker means.

Data source: `twi-aviandata.s3.amazonaws.com`, provided by [The Water Institute](https://thewaterinstitute.org/) (Gulf of Mexico avian monitoring, post-Deepwater Horizon).

## Status

The pipeline runs from the images alone. Survey counts are never an input; the CSV is used only to check the output.

| | Measured | On |
|---|---|---|
| Registration success | **96.7%**, 0.38px median reprojection error | 60 pairs |
| Detection precision, on frames the pipeline accepts | **0.30–1.00** per frame | 7 hand-labelled frames |
| Dot placement error | **0.65px** median | 1,648 hand-labelled dots |
| Classification, per dot | **0.781** | 691 of 885 dots, 7 frames |
| Legend class names read | **0.904** | 218 legend rows |
| Exported dataset | **118,270 boxes** across 413 photographs | |
| Model improvement, mAP@50 | **0.036 → 0.087** | 60 held-out photographs |
| Tests | **236** passing, CI on Python 3.10 and 3.11 | |

**Detection is finished.** The important part is not a single accuracy figure. It is that every image now carries a check on whether its detections can be trusted, before anything is built from them.

A frame that reports far more dots than the image contains cannot be accurate, whatever is done downstream. That check needs no labels and no survey data, so it runs on every image. Pass rates were measured on the 63-pair benchmark and applied to the band distribution of the **18,252 screenshots that carry any dots** (of 18,304 in total): about **48% of images pass, and they hold roughly 72% of the corpus's dots, close to 2 million annotations.** That is far more than training a DeepForest model needs. See [Choosing frames](#choosing-frames).

Both stages are scored against **hand-labelled dot positions**, not counts. The survey gives a count per image and never a coordinate, so until those labels existed there was no way to tell a pipeline that found the right dots from one that found the right *number* of dots in the wrong places. 1,648 dots across 12 frames are labelled in `data/labels/` and scored by `scripts/eval_localisation.py`.

The classification figure quotes seven of those twelve: the frames that both pass the check above and carry a class per dot. Quoting all twelve would mix in five frames the pipeline rejects, which measures a configuration nobody runs.

**The pipeline is complete end to end.** Recovered dots are mapped onto the original photographs, sized into boxes measured from each frame's own birds, and exported as a DeepForest dataset of **118,270 annotations across 413 photographs**. Fine-tuning DeepForest's bird model on that data improves it on held-out imagery.

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
| Where are the dots, and how many? | `align.py` → `subtract.py` → `select.py` |
| Which class is each dot? | `legend.py` → `classify.py` |
| Where does it belong on the original? | `mapping.py` → `birdsize.py` |

```mermaid
flowchart TB
    subgraph IN["Inputs: the pipeline reads nothing else"]
        direction LR
        SS["Screenshot<br/>dots baked into pixels"]
        ORIG["Clean original photograph"]
    end

    subgraph L["legend.py: what does each marker mean?"]
        direction TB
        LOC["locate_dialog<br/>find the panel as a box, anywhere"]
        PAR["parse_legend + attach_class_names<br/>colour, shape, 24x24 template, name, count"]
        LOC --> PAR
    end

    subgraph D["align.py + subtract.py: where are the dots?"]
        direction TB
        SIFT["SIFT + RANSAC registration<br/>refuses rather than return a bad warp"]
        DIFF["chromatic difference + distance transform<br/>what was added, split back into single dots"]
        SIFT --> DIFF
    end

    SEL{"select.py: is this frame worth using?<br/>precision ceiling from detected / reported"}

    subgraph C["classify.py: which class is each dot?"]
        direction TB
        LABC["LAB colour anchoring<br/>offset measured per legend row"]
        NCC["NCC template match, capacity per row<br/>no match means no class"]
        LABC --> NCC
    end

    MAPN["mapping.py + birdsize.py<br/>place on the original, size the box per frame"]
    OUT["DeepForest CSV<br/>118,270 boxes on 413 photographs"]

    SS --> LOC
    SS --> SIFT
    ORIG --> SIFT
    DIFF --> SEL
    SEL -->|"accepted, ~48% of images"| LABC
    SEL -.->|"rejected"| DROP["frame excluded"]
    PAR --> LABC
    NCC --> MAPN
    MAPN --> OUT
    SURVEY["Survey CSV"] -.->|"validation only,<br/>never an input"| OUT
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

Measured against hand-labelled dot positions, on the seven accepted frames that carry labels:

| frame | dots labelled | precision | recall |
|---|---|---|---|
| `18May15…00825` | 10 | **1.00** | 1.00 |
| `17May10…5745` | 372 | **0.76** | 0.79 |
| `18June21…06389` | 103 | **0.74** | 0.53 |
| `19May18…00620` | 178 | **0.72** | 0.74 |
| `18May11…0027` | 335 | 0.46 | 0.63 |
| `18May15…426` | 193 | 0.42 | 0.51 |
| `27May12…0449` | 215 | 0.30 | 0.40 |

Median placement error is **0.65px** across all of them. Placement holds everywhere; what changes between frames is how much of the surroundings is detected alongside the markers.

Most of that loss is map ink. On `0027` a painted transect line runs the width of the frame and is detected along its length, and on `06389` the painted words "Rabbit Area" are detected as markers. Both are drawn in the same palette colours as the dots, so colour cannot separate them. Geometry could: a line is collinear and text is a regular run of characters. That is written up in [what did not work](#what-did-not-work), because the one filter tried for it removed real birds.

### Choosing frames

Not every image is worth using, and which ones can be worked out from the image alone.

The check is arithmetic, and it lives in `src/select.py`. Correct detections can never outnumber the dots actually on the image, so precision is capped at `dots present / dots reported`. A frame reporting seven times what it should hold cannot exceed 14% precision, however it is filtered afterwards. Nothing about that needs labels.

The band has to be two-sided. Under-detection is equally disqualifying, and a frame returning a single detection for 19 dots passed a one-sided cut while being useless. One quality target sets both edges: `q ≤ ratio ≤ 1/q`, with `q = 0.6`.

Verified against hand labels on three frames spanning two density bands and 10 to 186 dots:

| ratio reported / present | precision |
|---|---|
| 1.00 | **1.00** |
| 0.99 | **0.82** |
| 0.62 | **0.74** |
| 5.63 | 0.14 |
| 7.35 | 0.07 |
| 14.22 | 0.04 |

Applying the benchmark's per-band pass rates to the band distribution of the 18,252 screenshots carrying dots: **about 48% of images pass, holding roughly 72% of the corpus's dots.**

Frames that fail are mostly sparse scenes over textured ground. A mangrove colony with nine real markers returned 128 detections, because at this resolution a leaf and a marker are the same handful of pixels. Five ways of filtering those out were measured and none separated them; that is written up in [what did not work](#what-did-not-work). Excluding those frames costs little, because sparse images hold only 6% of the corpus's dots.

### Assigning classes

`classify.py` has to separate classes that share a colour, such as BRPE nest, bird and chick, all drawn in red. **The class identity is the legend row, not the name read off it** — one dialog carries two rows that both OCR as `ad`, and keying on the name merges them.

Four steps decide where a dot goes.

**Colour picks the candidate rows**, using the palette that image's own dialog uses, in Lab space. A legend glyph is drawn crisply on a white table cell; the same marker in the aerial is a few pixels of thin stroke over vegetation at a quarter of the resolution, so its measured colour drifts. That drift is corrected **per row, not per frame**. On `5745` the frame-wide median difference is `a = −12.5` while row 0 needs `−29.4` and row 2 needs `−11.0`. The glyph explains it: a thin asterisk mixes with whatever it sits on, where a filled circle keeps its own colour. On `00620` two rows of the same species and the same colour need `L = −84` and `L = −16`.

**A template match ranks them.** Normalised cross-correlation against the 24×24 template cut from each candidate row, with colour agreement folded into the score, so a dot sitting on a row's colour outranks one that barely reached the margin.

**Each row is limited to the Count the dialog states.** A dot whose best row is full goes to its next best row instead; a dot with no candidate row left is scored again against every row that still has space. Without this, populous `site` rows stay half empty while empty `bird` rows fill up, which is a label swap that per-class counts cannot see.

**A dot matching no row keeps no class.** That is the valid/invalid split Josh asked about, and the pipeline already makes it.

![classification](results/report_fig/fig_classify_17May10Camera2-Card1-5745.jpg)

Each row shows the dialog's own marker, the template cut from it, and six aerial patches drawn at random from everything assigned to that class. `scripts/make_classify_figure.py` generates it from the live pipeline, so it always shows current behaviour, and the title strip records which detection path ran.

The rings carry the verdict. Green means a hand label at that point agrees with the class the pipeline gave; red means it disagrees; white means no label sits there. **White is not evidence of an error.** Hand labelling is not exhaustive — 90 of this frame's 386 detections have no label — so an unringed patch is unverified rather than wrong.

Read that way, the figure says two things at once. The populous rows are right: `WHIB site` at 84 dots and `WHEG site` at 40 are green almost throughout, and the white rings among them sit on markers identical to the green ones. The two rows holding two dots each are wrong on every dot, and **both of them match the dialog's stated count exactly**. That is the clearest argument in the repository for why counts prove nothing about labels.

The whole frame, with every detected dot drawn in the colour of the class it was given, beside what the dialog states, what the pipeline assigned, and what the labels say:

![the frame classified](results/figures/fig_classify_17May10Camera2-Card1-5745.jpg)

`LAGU sit` reads 150 in the dialog, 148 from the pipeline and 139 by hand; `ROSP site` 62, 62 and 60; `WHEG site` 40, 40 and 40. The 19 dots with no class are mostly ink that is not a marker.

Per-dot accuracy, on the seven frames that both pass frame selection and carry a class per dot:

![accuracy per frame](results/figures/fig_accuracy_by_frame.png)

**0.781 over 885 dots.** Dropping `0449`, where the dialog's counts are misread, gives 0.842. `06389` reads 1.000 because it has two classes, which is worth saying rather than quoting.

One older measurement is kept here so neither is mistaken for the other. **Legend self-recovery scores 76–83%**: a legend glyph is shrunk to aerial scale and pushed back through matching to see whether it recovers its own class. That flatters the method, because the template and the test glyph come from the same pixels. The gap to 0.781 is how much of it comes from testing the method against itself.

### Mapping onto the original, and sizing the box

Recovered coordinates are positions on a screenshot, which is the photograph shrunk into a window. `mapping.py` places them on the original using the registration transform, at sub-pixel accuracy.

![mapped onto the original](results/figures/fig_mapping_to_original.png)

*Dots read off the screenshot, placed on the clean full-resolution photograph.*

The survey recorded a point per bird and never an extent, so no box size can be read off the data. `birdsize.py` measures it from the imagery, per frame, because eleven years of surveys flew focal lengths from 28mm to 300mm and the same species spans 10px on one photograph and 41px on another.

![boxes on birds](results/figures/fig_box_closeup.png)

Around each dot it cuts a patch from the clean original, compares lightness against the patch's own median so a pale bird on grass and a dark bird on sand both register, and measures the long side of the resulting shape. Dividing known species body lengths by the measured size gives **1.3 to 4.0 cm per pixel**, which is what these surveys fly, and the camera EXIF predicts that spread independently from its focal lengths and pixel pitches.

### The dataset

```
results/dataset/          25 benchmark frames        6,420 boxes
results/dataset_scaled/  413 frames drawn fresh    118,270 boxes
```

Every decision in this pipeline was checked against those 25 benchmark frames, so numbers measured there are optimistic by construction. The scaled run took 1,197 candidates stratified across seven survey years and three density bands and put them through the same pipeline untouched.

**The quality figures did not move at eighteen times the size.** Species resolution read 0.648 on 25 frames and 0.650 on 413; box size was measurable on 96% of frames before and 98% after.

Every row carries `frame` and `legend_row`, which together form a stable key, so correcting one legend row fixes every dot on it.

![every exported frame](results/figures/fig_box_per_frame.png)

*Every exported frame drawn on one page. A sample that happens to contain the easy cases will confirm whatever it is shown, so the figure draws all of them.*

### Training

Fine-tuning DeepForest's bird model on 349 recovered photographs, 60 held out, three epochs:

```
                  pretrained    fine-tuned
mAP@50              0.036         0.087       2.4x
best F1             0.288         0.333       both at 0.20
per frame           40 of 60 improved, 15 worse, 5 level
```

![training result](results/figures/fig17_e4_before_after_01343.png)

*A held-out photograph holding 1,611 birds. Pretrained DeepForest finds 542; after training on the recovered annotations, 1,268.*

![training result, second frame](results/figures/fig18_e4_before_after_0296.png)

*The more informative frame. Here the fine-tuned model drew **fewer** boxes than pretrained, 828 against 1,103, and still found nearly twice as many birds, 373 against 200. It is not drawing more, it is drawing better.*

Precision and recall rise together at three of the four thresholds, which distinguishes better boxes from merely more of them, and the gain is spread across the test set rather than carried by a few frames. Scoring is against the recovered annotations on photographs the model never trained on. An mAP@50 of 0.087 is a low base: these birds are 16 to 54 pixels on photographs over 5,000 pixels wide, which is the small-object regime where mAP is punishing.

Full detail in [`docs/training_analysis.md`](docs/training_analysis.md).

## The benchmark

An earlier four-image set was small enough to overfit, and it used the wrong ground truth. Reading the counting tool's own *Total Count* field settled which column is correct: the dot count is `category_sum`, the sum of the per-class columns, not `total_birds`. `total_birds` excludes chicks and undercounts by up to 57%.

The current benchmark is **63 stratified pairs**: 7 years × 3 density bands × 3 images, across 40 colonies, with dot counts from 7 to 2,037. Density band means the number of annotation dots on the image, not the number of colours: sparse is 5–50, medium 51–300, dense 301 and above. Sampling by band keeps the dense tail in, because that is where detection was always weakest. Detection and alignment are scored on the 60 of those 63 whose screenshot and original are both cached locally.

## Repository structure

```
src/legend.py       find the dialog, parse each row to marker, colour, shape, template, class, count
src/align.py        register the clean original onto the screenshot
src/subtract.py     annotations as image difference; turn ink into dot candidates
src/select.py       which frames to trust, from the reported-to-detected ratio alone
src/classify.py     Lab colour anchoring per row, NCC shape matching, capacity per class
src/mapping.py      screenshot pixels to original photograph pixels
src/birdsize.py     measure bird size per frame to set the box
scripts/            benchmark builders, evaluation harnesses, figure generators
tests/              236 tests
data/labels/        1,648 hand-labelled dots across 12 frames
results/dataset*/   the exported DeepForest annotations
blog/               the six-part write-up of this project
notebook/           Colab notebooks for the training runs
docs/learnings.md   what went wrong and why
```

`src/decompose.py` and `src/detect.py` are kept for reference and are not part of the live pipeline. `decompose.py` split each screenshot at roughly half its width, which discarded much of the aerial and the birds in it; `legend.locate_dialog` replaced it. `detect.py` needs CSV counts as an input, which breaks the rule that the pipeline works from the image alone.

## Quick start

```bash
pip install -r requirements.txt          # full
pip install pytest numpy opencv-python scikit-image PyYAML scipy   # tests only

pytest tests/ -q                         # 236 passing

python scripts/build_manifest.py         # survey counts, ~1.4 MB
python scripts/build_benchmark.py --per-cell 3   # the 63 pairs, ~472 MB
python scripts/eval_localisation.py      # per-dot precision, recall, placement, class
python scripts/eval_alignment.py         # registration success rate
```

`eval_localisation.py` is the gate. `eval_detection.py` compares counts, and a count
cannot see whether a dot sits on a marker.

To redraw the figures in this file:

```bash
python scripts/make_classification_figures.py
python scripts/make_classify_figure.py --pair 17May10Camera2-Card1-5745.jpg --rows 8
```

The tests need no data and no network; the fixtures they use are in the repo. Everything below `pytest` downloads from a public bucket, so no credentials are involved, but the pairs are large because the clean originals are about 7 MB each. `build_benchmark.py --dry-run` prints the selection and the download size without fetching anything. Both scripts cache, so an interrupted run can be repeated.

Class-name OCR needs the Tesseract binary, not just `pytesseract`. On Windows: `winget install UB-Mannheim.TesseractOCR`. The pipeline skips OCR and keeps parsing if Tesseract is missing.

All tunable parameters live in `config.yaml`.

## About this work

Roughly 40% of the time on this project went on measurement rather than building, and almost every parameter in `config.yaml` traces back to something measured rather than guessed.

Every dropped approach was written up with its root cause, which turned out to matter: two were abandoned for the wrong reason and later reopened. OCR is the clearest case, rejected at 4% precision on a test fixture 2.3× smaller than a real screenshot, and working at full resolution.

The recurring lesson is to check what is actually being measured before trusting the number. Detection was scored against the wrong CSV column for weeks, and a count metric reading a healthy 1.24× was hiding a bug that deleted a quarter of all real markers.

Built with guidance from the DeepForest maintainers at Weecology, and with thanks to [The Water Institute](https://thewaterinstitute.org/) for the archive and for their input in [discussion #6](https://github.com/vickysharma-prog/Recovering-computer-vision-annotations/discussions/6).

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

Five more were measured while building the classification stage:

| Approach | Result |
|---|---|
| Read the dialog's own **Total Count** and use it as a budget | Correct on 11 of 25 frames, absent on 8, OCR noise on 6. Where it reads it is accurate, and 11 of 25 is too thin to cap classes with. |
| Give **every** row with an unreadable count a capacity of zero | Recovers the frame it was built for and destroys frames whose Count column barely reads. One went from 0.516 to 0.065. Only rows parsed *below* the last counted row are safe to zero. |
| Discard a per-row colour offset that looks contaminated by a neighbouring row | Saved one dot on one frame and cost six on another. Pooled 0.789 to 0.779. A wrong offset only ever adds a candidate, and the template score handles it. |
| Let the species matcher accept two edits instead of one | `nest` becomes the species code `BNST`. Coverage bought with a wrong species is a regression. |
| Take the Name\|Count divider from the first gridline right of the marker | The geometry is right and the fix broke classification twice, 0.861 to 0.634 and 0.861 to 0.600. Moving the name boundary changes the parsed `class_name`, and row identity is keyed on that. It is a classification change wearing a legend-parsing disguise, and it has to be scored on `eval_localisation.py` rather than on name coverage. |

Full list with root causes: [docs/learnings.md](docs/learnings.md).

## Documentation

| | |
|---|---|
| [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) | Current state, start here |
| [`docs/learnings.md`](docs/learnings.md) | Every dead end, with its root cause |
| [`docs/training_analysis.md`](docs/training_analysis.md) | The training runs in full |
| [`docs/labelling_findings.md`](docs/labelling_findings.md) | How the ground truth was built |
| [`docs/legend_findings.md`](docs/legend_findings.md) | Legend parsing and OCR |
| [`blog/`](blog/) | The six-part write-up |

## Read more

**Blog series** — the full story of this project in six posts, from the data forensics through to the trained model:
**[vickysharma.hashnode.dev](https://vickysharma.hashnode.dev/)**

**Open source contributions** — my pull requests and issues across DeepForest and other organisations:
**[github.com/search?q=author:vickysharma-prog](https://github.com/search?q=author%3Avickysharma-prog&type=pullrequests&s=created&o=desc)**

## License

MIT. See [LICENSE](LICENSE).
