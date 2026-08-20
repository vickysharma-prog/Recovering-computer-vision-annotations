# GSoC 2026: Progress Report
**Project:** Recovering Bird Annotations from Historical Airborne Imagery
**Organization:** WeeCology / DeepForest (University of Florida)
**Contributor:** Vicky Sharma
**Mentor:** Josh Veitch-Michaelis
**Repo:** github.com/vickysharma-prog/Recovering-computer-vision-annotations
**Last updated:** 2026-08-14 (classification built out and measured against 1,648 hand-labelled dots; frame selection put in code; five changes shipped and five reverted)

> **READ THIS FIRST: three earlier claims in this document have been retracted.**
> They are left in place with correction notes rather than deleted, because the
> reasoning that produced them is instructive.
> 1. **`total_birds` is NOT the dot-count ground truth** (corrected 2026-07-20).
>    Every detection accuracy figure dated before 2026-07-20 (including the
>    headline "70.8%" and the four study images' "true" counts) was measured
>    against the wrong quantity. See *Ground Truth Correction*.
> 2. **The count-prior (w=0.9) result was retracted** (2026-07-01). It games the
>    metric it is scored on rather than improving classification. See the
>    correction note on that section.
> 3. **"Detection is close to done" was wrong** (corrected 2026-08-07). It rested
>    on a median count ratio of 1.24x. Hand-labelled dots put detection at
>    **precision 0.138, recall 0.345**. The count figure is not incorrect, it
>    answers a different question, and the two failure modes behind it cancel
>    inside a median. See *Hand-Labelled Measurement*.
>
> Current state in one line: **both halves of deliverable #1 are measured against
> hand-labelled dots.** Alignment 96.7% at 0.38px. Detection, per frame on the seven
> frames that pass selection and carry labels: precision 0.30–1.00, recall 0.40–1.00,
> placement **0.65px** median. Classification **0.781 per dot** over 885 dots on those
> same seven frames, 0.842 without the one frame whose dialog counts are misread.
> Ground truth is now **1,648 dots across 12 frames**. **211 tests** pass.
>
> Next is deliverable #2: map the dots onto the original photograph, export the
> DeepForest CSV, and train. Fine-tuning waits for the training run, because box size
> and the precision the model actually needs cannot be settled by argument.

---

## Problem Statement

The 2010 Deepwater Horizon oil spill triggered the largest avian monitoring effort in Gulf Coast history. Scientists captured **18,304 aerial survey photographs** and annotated **340,000+ bird observations** using a point-counting tool that baked colored dots directly into screenshot pixels. No coordinate data was saved. The annotations exist visually but are inaccessible to any machine learning pipeline. This project recovers them.

**Dataset:** twi-aviandata.s3.amazonaws.com: 18,304 screenshots, 49,204 CSV rows, 102 species, 18 annotators, 442 colonies, 7 years (2010–2021), 5 Gulf Coast states.

---

## Approach: Study First, Build Second

Before writing any pipeline code, two weeks were spent on data archaeology:
- Mapped 533,000 files across the S3 bucket
- Analyzed 49,204 CSV rows across 60 columns
- Physically measured 25 diverse images
- Tested SAM 3, DeepForest, and GroundingDINO on raw screenshots

Every pipeline parameter was **measured, not guessed.** Detection accuracy improved from 44% (guessed parameters) to 70.8% (measured parameters) without changing the fundamental approach. **Both of those figures are retracted**: they were scored against `total_birds`, and the pipeline was fed CSV counts as an input. The current benchmark replaces them (see *Ground Truth Correction*).

---

## Pipeline

```
Screenshot → Decompose → Detect → Validate → Map Coords → Export → Train
```

| Stage | Module | Status |
|-------|--------|--------|
| Decompose | `src/decompose.py` | Done, CI passing. **Superseded** for the legend path by `legend.locate_dialog` (the 50%-width split discarded half the aerial) |
| Detect (colour) | `src/detect.py` | **Legacy**: needs CSV counts; not wired into anything live (only its own tests). The self-contained colour baseline/fallback is `classify.detect_dots` |
| Legend (per-image marker→class) | `src/legend.py` | Done, CI passing (PR #3). `locate_dialog` succeeds on ~90% of a 21-image sample, but "returned a box" ≠ correct, false positives seen |
| Classify (aerial dots vs legend) | `src/classify.py` | Done, but the weak half. Lab colour anchoring + NCC. Self-recovery D 55→76%, A 56→83%; on real aerial dots agreement is **0.36** |
| **Align (screenshot ↔ clean original)** | **`src/align.py`** | Done. 96.7% success (58/60), 0.38px median error |
| **Subtract (difference-based detection)** | **`src/subtract.py`** | Done. Detection error 8.40x → **1.24x** median (sparse 2.13x, medium 1.24x, dense 1.14x) |
| Validate | `src/validate.py` | Not started as a module (notebook only) |
| Map coords | `src/map_coords.py` | Not started as a module (notebook only) |
| Export | `src/export.py` | Not started as a module (notebook only) |
| Pipeline | `src/pipeline.py` | Not started as a module |
| Full run (18,304 images) | not started | Detection gate now met; next after the downstream stages exist |

**Test count: 166 passing.** All of the above is committed on PR #7 (branch
`feat/detection-pipeline`), CI green.

### Stage 1: Decompose (`src/decompose.py`)
3-method consensus separates aerial photograph from species dialog:
1. **Grey profile**: first column with high grey pixel fraction
2. **Sobel edges**: strongest vertical edge
3. **Color variance drop**: first low-variance column

Final boundary = median of 3 candidates. Robust across all screenshot formats. Also strips Windows title bars and taskbars.

### Stage 2: Detect (`src/detect.py`)
Wide HSV bins across 6 color channels (red, yellow, green, cyan, blue, magenta) maximize recall. Key behaviors:
- **Vegetation-adaptive saturation boost** (+30 to +70 S units) prevents green dot suppression in habitat-dense images
- **Distance transform** splits merged/overlapping dot clusters
- **Rank-order species assignment** from CSV: largest detected color group → most abundant species
- **Top-N selection** where N = expected count from CSV (drives precision)

### Stage 3: Validate (notebook → module in progress)
HSV saturation sampled at each dot center confirms real annotation dots (S ≥ 120) vs. false positives (S < 80). Spatial distribution check across 5×5 grid confirms dots are not clustered. Overall plausibility: **98.3%**.

### Stage 4: Map Coords (notebook → module in progress)
- **SIFT + RANSAC** (2,000 keypoints, Lowe ratio 0.75): maps screenshot → original photo at **0.5 px accuracy** when ≥10 matches found (57% of images)
- **Uniform height-ratio scaling** (fallback): uses `scale_y` for both axes, ~5–30 px error (43% of images)

### Stage 5: Export (notebook → module in progress)
DeepForest-compatible CSV: `xmin, ymin, xmax, ymax, label, score`. Train/test split by colony prevents geographic leakage.

### Training Pipeline (notebook)
| Experiment | Annotations | Result | Root cause |
|---|---|---|---|
| Binary, all data | 3,851 | All scores = 1.0 | SAM 3 bfloat16 autocast persisted after deletion |
| Binary, bfloat16 fixed | 3,851 | Fewer detections than pretrained | Fixed 80×80 boxes → borderline IoU |
| Species-aware boxes | 3,851 | Zero detections | ~30px position errors from uniform mapping |
| **SIFT-only** | **920** | **+29% max score, 1 high-conf detection** | Position accuracy > data quantity |

**Critical bug found and fixed:** SAM 3 returns bfloat16 tensors that silently corrupt coordinate precision during DeepForest training. Explicit `torch.cuda.amp.autocast(enabled=False)` before training resolved 300 false detections at score 1.0.

---

## Results

> **⚠️ The table below is measured against `total_birds` and is unreliable.**
> See *Ground Truth Correction*. Current, correctly-grounded numbers:
>
> | Metric (63 stratified pairs, 60 scored, vs `category_sum`) | Value |
> |---|---|
> | Alignment success | **96.7%** (58/60) |
> | Alignment reprojection error | median **0.38px** |
> | Detection ratio: OLD colour | 8.40x over |
> | Detection ratio: NEW subtraction | **1.24x** over |
> | …dense band | **1.14x** |
> | …medium band | **1.24x** |
> | …sparse band | **2.13x** (still the weakest) |
> | Symmetric error, median \|log2 ratio\| | 3.07 → **0.53** |
> | Classification, self-recovery | D **76%**, A **83%** |
> | Classification, real aerial dots (41 frames) | 0.26 → **0.36** |
> | Tests | **166 passing** |
>
> Read the two classification rows together. Self-recovery tests a glyph against
> templates cut from its own pixels, so it flatters the method; the 0.36 is what
> the matcher scores on real aerial dots.

| Metric | Value |
|--------|-------|
| Images processed | 34 (0 crashes) |
| Detection accuracy (batch of 30) | **70.8%** (vs `total_birds`: unreliable) |
| Detection accuracy (4 hard study images) | 52.3% |
| Precision | **98.3%** |
| False positive rate | 1.7% |
| Annotations recovered | 3,915 |
| Species detected | 21 of 102 |
| SIFT mapping success (0.5 px error) | 57% |
| Uniform fallback (~5–30 px error) | 43% |
| SAM 3 habitat classification | 97% |
| Processing speed | 11 sec/image |
| Estimated full dataset recovery | ~340,000 annotations |
| DeepForest pretrained max score on aerial | 0.376 (blind) |
| DeepForest SIFT-trained max score | 0.300 (+29%) |
| High-confidence detections after fine-tuning | 1 (vs 0 pretrained) |

---

## What Failed and Why

| Approach | Result | Root cause |
|---|---|---|
| OCR on dialog legend | 4–15% precision | Font rendering inconsistent across 18 annotators |
| Dialog color clustering | 8% accuracy | Colors bleed across species rows in dialog |
| Narrow HSV bins | 44% detection | Missed faded and edge-rendered dots |
| Text watermark filter | Removed real dots | Colony-row birds match text alignment heuristics |
| Training on full dataset | 0 high-conf detections | ~30px position errors corrupted gradient signal |
| Species-aware boxes | Worse than baseline | Added noise to a geometrically simple problem |

---

## Mentor Feedback (Josh Veitch-Michaelis)

### Round 1: Status (updated 2026-06-29)
| Feedback | Status |
|---|---|
| Re-run detection without combining classes, nest vs bird distinction matters | **DONE**: classify.py separates same-colour classes by shape/template |
| Look into template matching (convolution) as follow-up to colour | **DONE**: within-colour assignment uses template correlation |
| PR needs 2 sentences explaining it + 2 sentences explaining the plots | **DONE**: figure README + meeting doc |
| Show a test image with fake dots to explain bar chart differences | **DONE**: `results/figures/fig5_color_vs_shape.png` |
| Whenever using comparative words, give both numbers with definitions | **Applied** in all docs/figures (e.g. recall A 36/B 77/C 90/D 69%) |

### Round 2: Category assignment problem (resolved in discussion)

**Josh's concern:** "Within each color group, dots will be proportionally assigned to categories based on CSV ratios". This reads like a random assignment, which would be incorrect.

**Root cause of the problem:** Current code assigns dots of a given color to one species; sub-categories (WBN, Site, Nest, etc.) were being split proportionally within a color group, effectively random because no spatial or shape signal distinguishes them.

**Agreed solution (from call):** The dialog box in each screenshot contains the legend, each row shows a colored marker icon + class name + count. Parse the dialog legend per image to extract the (marker shape + color) → class mapping automatically. Use the extracted icons as templates to classify dots by both shape and color.

**Key principle Josh stated:** *"The detector would ideally work without an annotation file."* The detector should be fully self-contained, reading class names, marker shapes, colors, and counts directly from the dialog box. The CSV then serves only as validation ground truth to check retrieval accuracy.

**Agreed plan:**
1. Parse dialog legend per image → extract (icon, class name) pairs
2. Use extracted icons as templates for shape+color matching in aerial region
3. Works for any image in isolation, handles class map changes between images automatically
4. CSV counts used only for validation, not for class assignment

---

## Problems Solved

| Problem | Solution |
|---|---|
| OCR on dialog giving 4–15% precision | Abandoned; CSV has same data at 100% accuracy |
| Color-from-dialog mapping giving 8% accuracy | Switched to count-based rank-order matching |
| Text watermark filter removing real bird dots | Disabled; aspect ratio filter (>3:1) catches actual text |
| Red hue wrapping in OpenCV HSV (0–180 scale) | Circular mean using sin/cos in `circular_mean_hue()` |
| Scale_x fallback giving 23–39% horizontal stretch | Use `scale_y` for both axes, aerial not cropped vertically |
| SAM 3 bfloat16 persisting after model deletion | Explicit autocast disable before DeepForest training |
| 44% accuracy with narrow HSV bins | Widened to full color regions measured from 1,199 real dots |
| Training on full dataset: 0 high-conf detections | SIFT-only training, position accuracy > data quantity |
| Category assignment was effectively random | Agreed to parse dialog legend per image (in progress) |

---

## Legend Parser: Work Log (2026-06-27)

Started the dialog-legend parser (Josh's category-separation ask). Key outcomes:
- Pulled 4 full-res screenshots from S3 (A/B/C/D study images). Confirmed the repo fixture is ~2.3× downscaled, markers are 5–7px there vs 12–18px at full res.
- **Confirmed (shape, color) → class per image**, and that the shape→category map is **not global** (e.g. "Site" = ● in D but ✳ in A/B). Validates the per-image, self-contained approach.
- Built `src/legend.py` (blob-anchored rows, per-row glyph/template/color, first-pass `locate_dialog`) and `scripts/run_legend.py`. Saved 4 real dialog crops + hand-read ground truth (`docs/legend_groundtruth.md`).
- **Two open blockers:** (1) parser needs a full-res retune, asterisk/plus markers fragment into multiple blobs and corrupt the row-pitch estimate; (2) robust dialog localization (floating window, variable position): title-bar method works on 2/4.

Full write-up for Josh: `docs/legend_findings.md`.

## Legend Pipeline: Work Log (2026-06-29): major progress

Built the full automatic pipeline end-to-end. Now runs on a raw screenshot:
`parse_screenshot` (locate dialog as a box + parse legend) → `attach_class_names`
(OCR + fuzzy-match) → `detect_dots` (split clusters) → `assign_classes` →
`select_by_count`.

**What got solved this session:**
- **Parser retuned for full-res** (scale-adaptive): removed thumbnail-tuned area/size caps; row pitch from the single-row gap floored at marker size; marker column = densest x-cluster (not median, fixed image B); dropped the MORPH_OPEN that eroded thin asterisk/plus arms. Rows now within ±1 on all 4 dialogs; colour ~90%; the `+`=Bird shape detects reliably.
- **Class names via OCR**: installed Tesseract (winget UB-Mannheim); `attach_class_names` reads each row, fuzzy-matches to the 98 species codes + category words ("BAPE"→BRPE). **Image D: species 15/15, no CSV.**
- **Aerial dot classification** (`src/classify.py`): colour + within-colour template/shape match. Same-colour classes separated (red → BRPE wbn/bird/chick/brood).
- **Recall fix (cluster splitting):** overlapping dots in dense colonies were undercounted (61%); distance-transform splitting → **~100% recall** (D: 637 vs 636 true).
- **Boundary fix (Josh's other meeting issue):** the old decompose split each screenshot at ~50% width (full-height-panel assumption), discarding ~half the aerial. `locate_dialog` now finds the dialog as a **box** anywhere in the frame. **Localization 14/14 (100%)** across 4 study + 10 unseen S3 images (`docs/scale_validation.md`).
- **Precision fix:** quality score (sat×val) + `select_by_count` (top-N per class by the legend's own count). Noisy image B: **1278 → 92 dots** (truth 117).

**Results (recall after selection):** A 36%, B 77%, C 90%, D 69%. (A is a dense 1407-bird colony, limited by aerial shape accuracy.)

**Deliverables produced:** 5 result figures + README (`results/figures/fig1-5_*.png`),
scale validation (`docs/scale_validation.md`), and a meeting document
(`docs/GSoC_meeting_2026-06-29.docx`: built with python-docx; also a matplotlib
PDF but DOCX is the reliable one for browser viewing).

**Two open gaps, partially closed in the next session (see below):** (1) count-OCR coverage ~60-65% on ~10px digits; (2) aerial within-colour shape split ~60-70% (hardest on dense colonies).

## Legend Pipeline: PR & module hardening (2026-06-30)

Packaged the legend module as a reviewable PR and brought it to the repo's
tested-module bar. **PR: github.com/vickysharma-prog/Deepforest-bird-recovery-prototype/pull/3**
(branch `feat/legend-pipeline` → `main`).

**What this session added:**
- **Test suites**: `tests/test_legend.py` (40) and `tests/test_classify.py`:
  synthetic-glyph shape classification (circle/square/plus/ring), colour naming
  (red hue-wrap), template similarity, cluster splitting, OCR text-parsing
  (species fuzzy-match), and real-screenshot integration over the 4 study images
  (skip cleanly when fixtures absent). **Full suite: 143 passing.**
- **CI fix**: `tests.yml` was UTF-16 encoded, which GitHub Actions rejected
  ("invalid workflow file, no jobs run"); rewrote it as UTF-8. It now installs
  `scipy` (needed by `classify.py`) and runs all four test files (previously only
  `test_decompose.py`). CI green on the branch.
- **Config centralization (Josh's good-practice point)**: moved all of
  `legend.py`'s tunables into the `legend:` section of `config.yaml` so every
  module's configuration lives in one file. Values were reconciled to the
  module's actual numbers and verified **byte-for-byte identical** detection
  output on all 4 fixtures before/after (snapshot diff): no behaviour change.
- **PR scope**: code + tests + CI + 5 figures + the 4 study-image fixtures.
  Working-notes docs kept out of the repo (for our own reference). Notebook left
  on its `main` version: the legend pipeline runs via `scripts/run_legend.py`,
  and the notebook will get a proper end-to-end legend section once the module
  is built out further.

**Libraries used (legend module):** OpenCV (HSV segmentation, connected
components, contours, distance transform), NumPy, SciPy (`ndimage` label /
maximum_filter for cluster splitting), PyYAML (config), and Tesseract via
`pytesseract` (class-name + count OCR, degrades gracefully when absent).

## Legend Pipeline: Incremental gap work (2026-06-30, continued)

Two targeted improvements to the two open gaps from the 2026-06-29 session,
validated while awaiting Josh's review on PR #3.

### 1. Count-OCR (B-style wide-dialog fix)

**Root cause identified:** B's dialog has a wide Count column (strip width ≈185 px
from `cx + pitch*8` to the right edge). The strip's left portion falls in the
class-name area, Tesseract reads name text as digits even with a digit whitelist.
A/C/D strips are ≤141 px and land correctly in the count column.

**Fix (`src/legend.py` → `_ocr_count`):** clips strips wider than 145 px to their
rightmost 80 px, aligning the window with the right-justified digit. Narrower strips
are unchanged (no regression on A/C/D).

**Result:** B count-OCR exact match: **57% → 79%** (11/14 rows correct).

**Scale image check:** ran on 10 unseen S3 images. The clip triggers for
img_02 (224 px) and img_03 (151 px); those images had already-wrong totals (8, 15
respectively) and the clip changes them to other wrong values, an honest
representation of unreadable digits rather than false-confidence counts. img_08
(Dry Bread, Σcount=0 in `scale_validation.md`) has 73 px strips, so the clip does
not fire; img_08's zero-count problem is a digit-recognition failure, not a strip
position error.

### 2. Count-prior for within-colour shape assignment

> **⚠️ RETRACTED (2026-07-01, later the same day). Do not cite these numbers.**
> The "+35 pp / +49 pp" gains below are an artefact of the metric, not a real
> improvement in classification. The attributable-dots metric credits a class up
> to its legend count; feeding that same count in as a prior therefore moves the
> score toward the count **by construction**, whether or not any individual dot
> was labelled correctly. Shape accuracy did not improve. The honest baseline is
> **w=0: A 38.6%, C 46.7%**. The change was parked in a git stash and is **not in
> the shipped code** (`src/classify.py` ships with the shape-name boost removed
> and no count prior: that configuration was selected by ablation, see
> *Matching Rework*). Kept here because the failure mode: scoring a method on a
> quantity you also feed it: recurred later and is worth recognising on sight.

**Root cause identified:** Aerial dot glyphs at full res are compact enough that
shape classification returns "unknown" for 69% of dots in image A. Template
similarity consistently favours the circle legend template (compact blobs look
circle-like), so all "unknown" dots in a same-colour group go to the circle class
regardless of the true marker distribution.

**Fix (`src/classify.py` → `assign_classes`):** for "unknown"-shape dots in a
colour group with ≥2 candidates, blend template similarity with count proportion
from the dialog's own Count column:

    score = (1 - w) * template_sim + w * (entry_count / colour_group_total)

Default `count_prior_weight=0.9`: count proportion dominates. Dots with a
resolved shape (circle/plus/star) are unaffected and continue to use
template+shape correlation only.

**Validation, attributable-dots metric** (`attributable = Σ_class min(detected,
legend_count) / Σ_class legend_count`): this is the right metric, it credits a
class only up to its expected count so over-assignment does not inflate the score.
Measured on all 4 study images before (w=0) and after (w=0.9):

| Image | Legend sum | w=0 recall | w=0.9 recall | delta |
|-------|-----------|-----------|-------------|-------|
| A (felicity 2012) | 1105 | 38.6% | 73.7% | **+35 pp** |
| B (gaillard 2011) | 449  | 14.9% | 14.9% | 0 pp |
| C (northdeer 2010)| 242  | 46.7% | 95.9% | **+49 pp** |
| D (raccoon 2011)  | 568  | 76.1% | 76.2% | +0.2 pp |

A and C improve substantially. D is flat (not regressed despite 4 same-colour red
sub-classes). B is flat, its problem is wrong colour assignment on background
false positives, which count-prior (a within-colour fix) cannot address.

**3 new tests** added to `tests/test_classify.py` (explicit `count_prior_weight=0.9`
on all three): count-prior overrides template for "unknown" shape; shape-known dots
are unaffected; total_count=0 falls back to template. **Total suite: 146 passing.**

---

## Image B: Deep Diagnostic (2026-07-01)

> **⚠️ CORRECTION (2026-07-01, later): the core premise of this section is WRONG.**
> B's Count column is **NOT blank.** The counts are all present in the dialog
> (24, 24, 0, 0, 2, 4, 22, 12, 1, 14, 7, 2, 4, 1 = 117). The real root cause was
> **dialog localization**: `locate_dialog` cropped B's dialog ~111px too narrow
> because B's long class names ("BRPE chick nest w/o adult") widen the Name
> column, and the fixed marker-relative right edge (`mx + msize*30`) cut off the
> entire Count column: so the counts only *looked* blank. There is **no
> "site=444" real value**; that was OCR reading a border artifact in the
> cut-off/wrong strip. Fixes shipped: (1) `locate_dialog` extends the right edge
> to the grey panel's own border (B bbox 369→496); (2) `attach_class_names`
> splits Name vs Count at the table's own vertical gridlines. Result: **B counts
> 0→10/14 correct; C "LAGU site" 134→13; D stable; names not regressed; 83 tests
> pass.** Ignore Root causes 1–3 below (blank column / site=444 / 145px clip): 
> they were built on the cropped-dialog artifact. Caught by checking the count
> column against the source dialog image.

### Decision: fix B before moving to the next pipeline stage

This is a real project that needs to scale to 18,304 images. B's 14.9% attributable
recall is unacceptable at scale, rocky-background sites like B will recur across
the full dataset. User decision: **fix B first before advancing to map_coords /
export / train.**

### What the diagnostic revealed

A dedicated diagnostic script (`diagnose_B.py`) was run on `B_gaillard_2011.jpg`
(1408×649 px, 16 legend entries). The dialog crop was also saved and inspected
visually (`B_dialog_crop.png`). Full findings below.

#### Detected dot colour distribution

| Colour | Count | Median quality (sat×val) | Interpretation |
|--------|-------|--------------------------|----------------|
| green  | 495   | 0.122 (LOW)              | vegetation / background false positives |
| blue   | 246   | 0.510 (HIGH)             | **likely real LAGU birds** |
| red    | 244   | 0.543 (HIGH)             | **likely real BRPE birds** |
| orange | 109   | 0.142 (LOW)              | rocky background; orange NOT in legend → unassigned |
| grey   |  62   | 0.084 (VERY LOW)         | background texture; assigned to "site" |
| yellow |  86   | 0.120 (LOW)              | background; small noise |
| magenta|  11   | 0.298 (moderate)         | probably real WHIB adult |
| cyan   |   3   | 0.672 (HIGH)             | probably real ROYT adult |

Total raw detected: 1278. Dots in a legend colour: 1147 / 1278 = 89.7%.
The high-quality red (244) and blue (246) dots are almost certainly real birds. Detection is working. The problem is downstream: wrong counts in the denominator,
and inability to filter low-quality false positives without valid per-row counts.

#### Legend OCR output (what the pipeline reads from B's dialog)

| Class | Colour | Shape | Count (OCR) | Reality |
|-------|--------|-------|-------------|---------|
| BRPE WBN | red | plus | None | blank in dialog |
| BRPE chick | red | ring | None | blank in dialog |
| BPRE empty nest | red | square | None | blank in dialog |
| BPRE aband. nest | red | ring | None | blank in dialog |
| BRPE chick nest w/o adult | red | circle | 1 | misread |
| LAGU site | blue | star | None | blank in dialog |
| BRPE adult | red | circle | None | blank in dialog |
| LAGU stand | blue | circle | None | blank in dialog |
| BRPE Imm. in colony | red | plus | None | blank in dialog |
| CAGO adult | green | plus | 1 | misread |
| WHIB adult | magenta | plus | 1 | misread |
| unid. duck | yellow | ring | None | blank in dialog |
| ROYT adult | cyan | square | 1 | misread |
| SATE adult | yellow | square | 1 | misread |
| site | grey | square | **444** | **MISREAD, cell is blank** |
| ad | grey | square | None | blank in dialog |

**Total OCR legend sum: 449** (445 of which is from the "site=444" misread).

#### Root cause 1: The Count column in B's dialog is entirely blank

Inspecting the dialog image visually confirms: **B's annotator did not fill in
per-species counts.** All per-row Count cells are empty. The only count information
is the **"Total Count: 117"** visible in the bottom-left corner of the dialog box
(the tool's running total, not per-species). This is a valid real-world scenario. some annotators count total birds without breaking down by species in the Count column.

The pipeline currently only reads per-row counts; it has **no code to read the
"Total Count:" header value.** When all per-row counts are None, `select_by_count`
has no data → all 1278 raw dots are kept → 10x over-detection (1278 vs 117 true).

#### Root cause 2: "site = 444" is an OCR misread of an empty cell

The `_ocr_count` function in `src/legend.py` uses a blank-cell heuristic:
`return 0 if dark < 6 else None`. The "site" row's strip has enough table-border
dark pixels to exceed the threshold → instead of returning None (or 0), Tesseract
reads the border artifacts and returns 444. This single misread inflates the
denominator from ~5 to 449, making the attributable recall look like 14.9%
when it should be computed against a much smaller denominator.

Without the "site=444" misread, the OCR legend sum would be ~5
(4 rare species misread as count=1 each). Even at 100% recall on those 5 classes,
the metric would be 5/5 = 100%, but that's too small a denominator to be
meaningful without the major BRPE/LAGU classes.

#### Root cause 3: The 145 px count-OCR clip does NOT fire for B

Correcting a claim from the previous session: B's strips are **110 px wide**
(measured: `cx≈99, pitch=20 px, pitch×8=160, strip = dialog_width - 259 = 110 px`).
The clip threshold is 145 px, so **the clip never fires for B**. The "B 57%→79%"
count-OCR improvement claim from 2026-06-30 needs re-examination, either it was
measured against a different fixture, or the improvement came from another change
in that session. The 145 px fix is confirmed correct for `img_02` (224 px) and
`img_03` (151 px), but not for B's dialog.

#### Root cause 4: Grey "site" markers are invisible to HSV colour segmentation

If B truly has 444 nest-site markers (grey squares baked into the aerial), they
have low saturation (grey ≈ 0 in HSV S channel) and would not pass the `sat > 80`
threshold in `_dot_centers`. The 62 "grey" detected dots are background texture
that barely passed the threshold. They are NOT the actual grey square markers.
Grey marker detection would require a separate low-saturation detection pass.

#### What the quality-threshold test showed

Applying a quality minimum filter to try to remove false positives:

| Threshold | Dots remaining | Attributable recall |
|-----------|----------------|---------------------|
| (none)    | 1278           | 14.9%               |
| ≥ 0.10    | 1146           | 2.4% (WORSE)        |
| ≥ 0.15    |  629           | 1.1% (WORSE)        |

Raising the quality threshold HURTS, the 62 "grey site" dots (quality 0.084)
are the main source of attributable recall (they contribute 62/67 attributable
hits against the misread "site=444" denominator). Filtering them out drops recall.
This confirms that the quality-threshold approach cannot fix B; the denominator
and count-reading problems must be fixed first.

### Two targeted fixes planned for B

#### Fix A: Read "Total Count:" from dialog header as fallback
**Code location:** `src/legend.py` → `attach_class_names()` or a new helper.
When all per-row counts come back as None (or sum to 0), the pipeline should
attempt to OCR the "Total Count: NNN" text visible in the dialog's UI header area.
Use this total as a soft cap for `select_by_count` (keep the N highest-quality
dots regardless of class, where N = total count from header). The "Total Count"
text is on the left panel of the dialog, separate from the row table.

#### Fix B: Tighten the blank-cell heuristic in `_ocr_count`
**Code location:** `src/legend.py` → `_ocr_count()`, line: `return 0 if dark < 6 else None`.
The current threshold of 6 dark pixels is too permissive, table borders push
border-adjacent cells above 6. Fix options:
1. Raise the threshold (e.g. `dark < 15`) so cells with only border artifacts
   return 0 (blank) instead of triggering Tesseract on noise.
2. Add a cell-content check: if Tesseract returns no digits AND the strip shows
   no obvious digit-like dark region, return 0 instead of None.
3. Check if the strip is dominated by uniform light grey (background) rather than
   text-colour contrast.

Priority: Fix A first (Total Count fallback) because even with correct blank-cell
detection. B still has no per-row counts to use for `select_by_count`.

### What B's ideal outcome looks like (post-fix)

After the two fixes, for B we expect:
- `attach_class_names` reads "Total Count: 117" → treats as total cap
- `_ocr_count` returns None/0 for all 16 blank per-row count cells (no "444" misread)
- `select_by_count` keeps top-117 dots by quality across all classes
- Top 117 dots: ~3 cyan (ROYT, quality 0.67) + ~11 magenta (WHIB, quality 0.30)
  + ~103 from red/blue (BRPE/LAGU, quality 0.51–0.54)
- These 117 dots are correctly colour-assigned → recall against "117" total ≈ high
- Attributable metric becomes meaningful: denominator = 117, not 449

Grey "site" markers (if any truly exist in B's aerial) remain undetectable until
a separate low-saturation detection pass is added. This is a **follow-up gap**,
not part of the immediate fix.

## Matching Rework: Josh's three asks (2026-07-10)

Josh identified the genuinely hard part of the project: separating classes that
**share a colour** on the aerial (within-colour accuracy was ~60–70%, image A as
low as 36%). He saw two concrete failures, a red star matched to circles, and
yellow matched to nothing, and gave three directions, plus one correction.

**His correction:** BRPE WBN vs brood are *different colours* (light vs dark red),
not identical markers, so the pair is separable, not a floor. Confirmed: hue
barely varies within a colour group; **brightness** is what separates them, which
is why the anchoring space has to include value, not hue alone.

| Ask | What was built | Outcome |
|---|---|---|
| 1. Anchor colour to the dialog's **own** palette per image, not global bins | `_legend_palette` + `_color_candidates` in `classify.py`, in **Lab** space, with a max-distance reject for off-palette noise | Within-colour D 55→73%, A 56→68% |
| 2. Implement **background removal**: mask the glyph by its own colour so the matcher stops correlating to grey | Hue-consistent masking (`_HUE_TOL=14`) before templating | Prerequisite for 3 |
| 3. Try shape options empirically (**NCC** / match the actual patch, not a binary mask) | `cv2.matchTemplate` TM_CCOEFF_NORMED on intensity templates, ablated against binary-mask cosine | **NCC won**; final D 55→**76%**, A 56→**83%** |

**Ablation also removed something:** the `+0.35` shape-name boost *hurt* and was
deleted. It was not assumed either way. It was measured.

**Metric used (deliberately):** legend **self-recovery**: degrade each legend
glyph to aerial scale, run it back through the matcher, check it recovers its own
class. This is ungameable and needs no labels. Post-`select_by_count` count-recall
was explicitly *not* used, because that step forces the counts to match.

**Real-aerial total count error** Σ|legend − assigned| also fell on all four study
images: A 929→866, B 819→351, C 1073→979, D 243→149.

### Honest-metrics correction made during this work

Two reporting errors were caught and fixed rather than shipped:

1. **A capped-recall metric rewarded over-assignment.** It made image A look like
   it regressed 77→58%; on the symmetric total-error metric A had actually
   improved. Capped recall was dropped.
2. **The trust caveat in the figures was backwards.** The docs said "B/C legend
   counts are OCR-noisy, A/D are reliable". Measured legend coverage vs true
   count is the reverse: **D 92%, B 84%, C 60%, A 33%**. So A's −7% "improvement"
   largely reflects converging to an *incomplete* legend, which on A can mean
   moving away from truth. **D is the trustworthy image; A is directional only.**
   Also, A's error is dominated by a single class: WHIB contributes +293 by
   itself (legend 1, assigned 294): so the open bug sits *inside* the improved
   number.

### Bug fixed: dots detected inside the dialog

`_dot_centers` applied the `exclude` box only to component centroids, so
cluster-split sub-dots near the dialog edge escaped it. Image A had **54 dots
detected inside the dialog**; now 0 (total 1726→1672). Fixed by re-applying the
exclude filter after splitting.

---

## External input: The Water Institute (GitHub discussion #6, ~2026-07-17)

@cronosnull shared their parallel work on the same imagery. Three points:

1. **LAB colour space**: they reached the same conclusion independently. External
   validation of ask #1 above.
2. **Use the clean high-resolution originals.** Every annotated screenshot has a
   corresponding un-annotated original. Align with SIFT + homography and
   **subtract**: annotations come from image *difference*, not colour thresholds.
   This generalises across years where colour thresholds overfit.
3. Alignment also yields a free mask for title bars and scroll bars.

Point 2 reframed the hardest stage and became the current work. Important limit,
stated up front: **subtraction answers *where* annotations are, not *which class***.
The legend marker→class matching is still required. Subtraction feeds that work; it
does not replace it.

---

## Data layout discovery (2026-07-20)

The bucket `twi-aviandata.s3.amazonaws.com` is public and needs no auth. It has
three top-level prefixes, and **only one of them is usable**:

- `DottedImages/` and `HighResolutionImages/` are raw dumps whose naming schemes do
  not correspond (`7May10DLJ1Area1.JPG` by observer/area vs
  `10 June 2010 Camera 1 Card 1 001.JPG` by date/camera/card). **No pairing is
  possible from these filenames.** Do not build on them.
- **`avian_monitoring/` is curated**, organised by year / region / colony:
  - `avian_monitoring/screenshots/<year>/<region>/<colony>/<name>.jpg`
  - `avian_monitoring/high_resolution_photos/<year>/<region>/<colony>/<name>.jpg`
  - `avian_monitoring/dotting_information/`: the survey data

**Key fact:** a screenshot and its clean original sit at the **identical relative
path** under the two prefixes. Pairing is a plain string swap. Verified by HTTP 206
on both. Example pair: screenshot 809×1177, original 3168×4752, so the aerial
region is the whole original downscaled ~4x, not a zoomed crop.

`processed_data/avianData20102021.csv.gz` covers **18,304 unique screenshots**
(exactly the corpus) with **100% having a paired high-res path**, per species and
per dot-category, via columns `screenshot_new` / `HighResImage_new`.

---

## ⚠️ Ground Truth Correction: the most important finding (2026-07-20)

**Every detection accuracy number in this document dated before 2026-07-20 was
measured against the wrong quantity.**

The question "how many dots are on this image?" had never been verified; it was
assumed. It was settled by reading the counting tool's **own "Total Count" field**,
which is baked into every dialog, off four screenshots **by eye** (count-OCR is
only ~60–65% reliable, so it could not be trusted for this), and comparing:

| screenshot | dialog Total Count | `category_sum` | `total_birds` |
|---|---|---|---|
| 30May12Camera1-Card1-0837 | 18 | 18 | 18 |
| 12June10Camera1-Card1-0007 | 43 | 44 | 38 |
| 17June13Camera1-Card1-0048 | 18 | 18 | **6** |
| 17June13Camera1-Card1-0051 | 74 | 74 | **32** |

Mean |error| vs the tool's own tally: **`category_sum` 0.25**, `total_birds` 14.75.

**Conclusion: the dot-count ground truth is `category_sum`**: the sum of the
per-dot-type columns. **`total_birds` is an ecological metric that excludes
chicks**, and undercounts dots by up to 57% on a single image.

**A real bug this exposed:** there are **four** chick-like columns, 
`ChickNestwithoutAdult`, `Chicks/Nestlings`, **`ChicksNestlings`**, `ChickNest`.
`ChicksNestlings` and `Chicks/Nestlings` are distinct and both carry data. Omitting
`ChicksNestlings` made one image read 11 dots when the dialog says 18.

Dialog row labels map to columns as: `site`→Site, `bird`/`adult`/`AD`→
OtherAdultsInColony, `chick`/`YY`→ChicksNestlings, `cnwoa`→ChickNestwithoutAdult,
`brood`→Brood, `WBN`→WBN.

**Consequences, what this invalidates:**
- The four study images' "true" counts (636 / 1407 / 117 / 115) were
  `total_birds`-like and **could not be mapped back to specific survey rows at
  all**. Among Gaillard-2011 images with `total_birds`≈117 the actual dot count
  ranges **116 to 808**.
- So the headline "image B over-detects 10.8x" may really be **~1.6x**. Detection
  was likely never as bad as reported.
- The benchmark was therefore rebuilt from identifiable images (below), and the
  four unsourced fixtures retired as a scoring basis.

Corpus-wide, `category_sum` and `total_birds` agree on ~72% of screenshots (median
ratio 1.00) but dots exceed `total_birds` on **28.1%**, by up to 7x.

---

## Difference-Based Detection (2026-07-20): current work

### Benchmark rebuilt

Four images was how the colour thresholds got overfitted in the first place. The
benchmark is now **63 stratified pairs**: 7 survey years × 3 density bands
(sparse / medium / dense) × 3 images, spanning **40 colonies and 16 regions**, dot
counts 7–2037. The dense tail is sampled on equal footing with the common case
because that is where detection is weakest.

Built by `scripts/build_manifest.py` → `scripts/build_benchmark.py`. Images cache
to `data/fixtures/pairs/` (gitignored, ~470 MB; regenerate any time).

### `src/align.py`: registration, with a mandatory quality gate

SIFT + Lowe ratio + RANSAC, original downscaled to 1600px. Returns `ok=False`
rather than a transform it cannot vouch for, at 18k images a silent bad warp is
far worse than a refusal, because it would mark the whole frame as annotation.

| metric | value |
|---|---|
| success | **96.7%** (58/60) |
| reprojection error | median **0.38px**, max 0.88px |
| inlier fraction | median **99%** |
| model chosen | `similarity` 56/58 |
| per year | 2010/13/15/18/21 = 100%; 2011 86%; 2012 88% |
| dense band | 95% |

The transform is pure scale + translation (perspective terms ≈ 0), so a
**similarity** model is preferred when it fits as well as a homography, fewer
degrees of freedom, more stable on low-feature scenes like open water.

### `src/subtract.py`: annotations as image difference

**Result so far (detected / true dots, 1.0 = perfect):**

| band | OLD (colour) | NEW (subtraction) |
|---|---|---|
| dense | 3.56x | **1.01x** |
| medium | 9.15x | **1.42x** |
| sparse | 63.51x | **6.07x** |
| **median overall** | **8.40x** | **1.46x** |
| median \|log2 ratio\| (symmetric) | 3.07 | **0.73** |

Detection is **not** scored with `select_by_count` or any per-class top-N, 
selecting toward a known count makes the count match by construction.

### Four traps found while building it: each cost an iteration

Recorded because each looked correct and was not:

1. **Plain `absdiff` fires on every edge.** Sub-pixel misregistration means an
   aerial full of birds on sand reads as annotation everywhere. Fixed by comparing
   each pixel against the best match within a small radius: real ink differs at
   *every* offset, a shifted edge matches at *some* offset. Visually confirmed, 
   birds stopped being counted as dots.
2. **Water dominates false ink on sparse frames.** Ripples and glint genuinely
   differ between the two renders and, being textureless, the shift search cannot
   cancel them. Those differences are **luminance**; ink is **chromatic**.
   Down-weighting L cut spurious ink **8.58% → 0.87%** on one coastal frame.
3. **…but chrome must be found on the FULL difference, not the chroma one.** A flat
   grey dialog over a sandy scene has almost no chromatic contrast; judged on
   chroma it goes unmasked and every glyph inside it is counted as a marker. Both
   differences are now computed in a single pass.
4. **Never mask a large region as chrome on size alone.** A dense colony fuses into
   one carpet spanning ~1/8 of the frame; masking it discarded **92% of the ink**
   on a 2037-marker image. Bounding-box fill also failed, it rescued that image
   but broke dialog masking elsewhere (sparse went 0.98x → 9.85x). **Saturation**
   separates them: measured medians **dialog 1, open water ~26, marker carpet
   70–74**.

**Plus a fifth, in sizing:** estimate single-marker area from the **distance
transform**, never from median blob area. When markers are all fused the median
blob *is* a run, so no split ever triggers and the frame reports one dot.

### What did NOT work

| Attempt | Result | Why |
|---|---|---|
| Percentile threshold on the difference | Forced ~1% of pixels through regardless of content | A sparse frame has far less than 1% ink; replaced with median + k·MAD |
| Modal-blob-size filter to drop text/lines | Collapsed on a noisy image (modal area → 6) | Noise blobs outnumbered markers *before* any colour gate ran |
| Legend-palette colour reject to remove red text/lines | Only 105 → 79 blobs, and **the legend failed to parse on the test image** | Couples detection to legend parsing, which itself fails on ~12% |
| Bounding-box fill to identify chrome | Fixed dense, broke sparse (0.98x → 9.85x) | Fill does not separate dialog from marker carpet; saturation does |

**Unresolved at the time of writing (later cut to 2.13x by the saturation floor):** sparse/coastal frames over-detect **6.07x**. Remaining false
ink there is genuine *annotation* that is not a dot, red site-label text and
transect lines, plus residual water. The principled filter is the image's own
legend palette (text/lines are off-palette), but that is coupled to legend parsing
and is Phase-3 work.

### Tooling notes

- `scripts/eval_detection.py` caches alignments in `data/cache/align_cache.json`;
  a full sweep is ~8 minutes uncached, seconds cached. **Delete the cache after
  editing `src/align.py`.**
- **Colab does not help**: the bottleneck is SIFT, which is CPU-bound in OpenCV;
  a T4 would sit idle.
- 20 new tests use **synthetic** pairs, so CI needs no network or S3 fixtures.

---

## Sparse Over-Detection Fix: saturation floor (2026-07-24)

The one band still failing the detection gate was **sparse**. Diagnosed it
spatially first (`scripts/diagnose_sparse.py`, overlays in `results/sparse_probe/`)
rather than tuning blind. The false ink on sparse frames is **not** primarily red
text/lines as previously assumed. It is **low-saturation residual** (water glint,
mudflat texture, and the grey panel behind a label). On a near-empty frame this
noise **outnumbers** the real markers and *poisons the size estimate*: the
distance-transform modal radius collapses to its floor → the size band widens →
the direct-keep and the splitter both flood.

### Fix: a single legend-free gate
Annotation ink is chromatic by construction; water/sand/grey UI is not. In
`subtract.dot_candidates`, compute each ink blob's **median screenshot saturation**
and drop it below `marker_sat_min` (config, =50) **before** `_marker_area` and the
splitter, so noise can neither set the modal size nor be split into phantom dots.
Measured medians that place the floor: marker carpet 70-74, clean sparse markers
169, background residual 7-40. Needs no legend, so it cannot fail when legend
parsing does (unlike the palette-reject in the *What did NOT work* table).

**Code:** `SubtractResult` gained a `saturation` field (computed once in
`extract_annotations`); the gate lives in `dot_candidates`; `config.yaml` →
`subtract.marker_sat_min: 50`; `scripts/eval_detection.py` gained miss/zero/over
counts (the |log2| median silently drops zero-detection frames, so a lever that
pushes a frame to 0 could otherwise hide as an improvement); one red test in
`tests/test_subtract.py` exercises the *drop* path (prior tests were all saturated
magenta = keep-path only). **164 tests pass.**

### Result (63-pair benchmark, vs `category_sum`)

| Metric | Before (this session) | After |
|---|---|---|
| sparse band | 2.96x | **2.13x** |
| medium band | 1.59x | **1.24x** |
| dense band (not broken) | 1.10x | 1.15x |
| overall median | 1.33x | **1.24x** |
| overall mean | 3.07x | **2.08x** |
| symmetric median \|log2\| | 0.76 | **0.53** |
| zero-detection frames | 2 | **1** |

One clean lever improved every band; dense verified safe per-image (only the
00097 artifact below moved sharply, and that is correct).

### Finding: the benchmark ground truth itself needs auditing
One "dense" frame looked like a catastrophic regression (450 → 9). Spatial check
(not count) explained it: **`16May15…00097` is a "No photo coverage for this area"
frame**: only survey-area polygons + labels, **zero dots on the image**. Its
`category_sum` of 450 is the **estimate written in the info-box text** (250 WHIB +
50 broods + 45 GREG + …). The old colour path scored 288 by counting polygon
lines; the saturation floor correctly returns ≈0. So this is a **ground-truth
artifact, not a regression**: excluding it, dense mean is 1.19x. No-coverage
frames with coloured labels *over*-detect, so more may hide in the over-detect
list (spot-check worth doing; not a full re-audit). Reinforces the ground rule:
**verify spatially, never by count alone.**

### The sparse residual is TWO causes, not one
Ruled out depth / peak-count / elongation as text discriminators (bold red label
text is deep and compact, so geometry cannot separate it from markers). The
remaining ~2.13x splits into:
- **Vegetation/tree texture** (e.g. 0076, 14x, a mangrove colony with 9 real
  dots): the Phase-3 legend-palette colour filter *will* help (foliage isn't
  marker-coloured).
- **Marker-red label text + transect lines** (0215, 03211): same red as the
  markers, so colour cannot separate it. A **standing limitation** on a handful of
  frames, not something Phase-3 closes. Stated honestly.

**Mentor deliverable** *(local)*: a self-contained visual report covering both
stages, real before/after detections, charts and code locations, sent to the mentor
directly rather than checked in.

## Wiring, honest measurement, and docs (2026-07-27 → 2026-08-04)

**1. Detection wired into classification.** `classify.detect_dots_subtract` takes
positions from `subtract.dot_candidates` and reads colour/shape at each one with the
same code the colour path uses, so `assign_classes` sees identical input either way.
Falls back to `detect_dots` when alignment refuses. Two tests lock this: count parity
with `dot_candidates` (stops an unmeasured colour filter creeping into detection) and
the fallback path. Commit `5213507`.

**2. Fixed vs derived templates: Josh's question, answered locally.** Measured the
shape-naming ceiling: only **29/63 legend rows (46%)** are named correctly by both the
contour heuristic and NCC-against-ideal-glyphs, and 15 further rows are squares that
no ideal glyph represents. A fixed template library requires naming the shape first,
so it inherits that 46% ceiling; derived templates skip naming entirely. **Conclusion:
stay derived.** Script `scripts/eval_shape_naming.py` is gitignored (local eval only),
the number goes to Josh by message.

**3. The stale-figure mistake.** The classification figure on the PR was a recompress
of `fig14_template_match_samples_D.png` dated 1 July, but the matching rework landed
10–20 July. Josh reviewed pre-rework behaviour and commented that the bottom three rows
were still wrong. Root cause: `make_report_figures.py` resized an old PNG instead of
running the pipeline. Fixed by `scripts/make_classify_figure.py`, which runs the live
path every time and prints the detection mode into the title. Same fix later applied to
`scripts/make_legend_figure.py`. **Rule: every figure ships from a generator that runs
the live path.**

**4. Classification measured honestly, on 41 frames not 4.** Ran previous vs current
matching over all cached frames whose legend yields ≥2 classes, detection held constant,
using the env toggles in `classify.py`
(`COLOR_ANCHOR=0 BG_REMOVAL=0 SHAPE_MATCH=cosine SHAPE_BOOST=1` = previous config).

| Band | Previous | Current |
|---|---|---|
| sparse (7) | 0.223 | **0.485** |
| medium (15) | 0.273 | **0.352** |
| dense (13) | 0.266 | **0.382** |
| **mean** | **0.263** | **0.357** |
| median | 0.250 | 0.339 |

27 frames up, 14 down. Per-frame CSVs in `results/classify_ab/`.

**5. Self-recovery overstates real performance.** `eval_matching.py` reports 76–83%,
but it degrades a legend glyph and checks it recovers its own class, so template and
test glyph come from the same pixels. On real aerial dots the same method scores 0.36.
Both numbers now appear together in the PR and the README. Named failure: WHIB site
(232 expected) and WHIB bird (86) both parse as `circle`, scoring 0.548 vs 0.540 on the
same dot; the pipeline produced group sizes 89/254, so the split is roughly right and
the **labels are swapped**. `select_by_count` caps sizes without reassigning, so it does
not fix this.

**6. Public docs rewritten (2026-08-04).** README.md described the superseded pipeline:
70.8% detection measured against `total_birds`, CSV drawn as a pipeline *input* in the
mermaid diagram, OCR listed under "what didn't work". All corrected. The old numbers are
kept but scoped to a clearly-labelled *earlier prototype* section stating what they were
measured on and against. `docs/learnings.md` gained items 20–30 (ground-truth column,
63-pair benchmark, per-image marker mapping, why the OCR verdict was wrong, refuse-don't-guess
registration, self-recovery circularity, counts cannot detect swapped labels).

**Repo-hygiene finding:** commit `02ec833` had deleted `docs/learnings.md` and
`docs/training_analysis.md` from the branch while README linked to them 7 times, so
merging PR #7 would have removed both from `main` and left dead links. Both restored and
un-ignored, along with this report. The remaining working notes (`TASKS.md`,
`legend_findings.md`, `scale_validation.md`, `legend_groundtruth.md`, the mentor-facing
HTML report, and all `.docx`/`.pdf` drafts) stay gitignored, since they are drafts and
scratch measurements rather than a record worth publishing.

## Experiment tracking (2026-08-04)

Henry suggested putting the project on Comet ML so progress is readable from a
dashboard. It is up and public:
**comet.com/vicky-sharma-1971/bird-annotation-recovery**

**What Comet can and cannot do here.** Its panels draw a line when a metric has
many points along a step axis, which in practice means training epochs. This
project has no training loop yet, so seven dated milestones render as seven
dots and the auto-generated charts say very little. Three attempts at
restructuring the logging confirmed that no logging shape fixes it. The
progress chart is therefore drawn by `scripts/make_timeline_figure.py` and
logged to Comet as an image, and Comet hosts figures, tables and run history
rather than generating the view. When DeepForest fine-tuning starts, its
Lightning `CometLogger` plugs in and the auto panels start working properly.

**What is in the project**

| Experiment | Contents |
|---|---|
| `project timeline` | progress figure, milestone table, the nine dropped approaches, 8 figures, README and this report as assets |
| `detection: colour thresholds` / `detection: subtraction` | `detection_ratio` at step 0/1/2 = sparse/medium/dense, so selecting both draws two comparable lines |
| `classification: previous` / `classification: current` | `classification_agreement` on the same band axis |

**Keeping it current.** `docs/milestones.csv` is the single source for the
history; both the figure and the logger read it, so they cannot drift apart.
After a change that moves a number:

```bash
python scripts/eval_detection.py            # refresh the eval CSVs
# add one row to docs/milestones.csv
python scripts/make_timeline_figure.py      # redraw the progress chart
python scripts/log_to_comet.py --clean      # republish
```

The method runs read the eval CSVs directly, so they need no manual edit. A
blank cell in `milestones.csv` means the metric was not measured at that
milestone; it is never carried forward from an earlier row.

**Honest caveat to repeat when showing it:** the milestone history is
transcribed from the dated entries in this report, because tracking was set up
after that work happened. Everything from here is logged as it runs.

### Status: parked on 2026-08-04, deliberately

The data is in Comet and correct, verified through the API. The dashboard is
not presentable, and after six attempts at restructuring it that is a property
of the tool for this project, not a thing left to fix. **Parked here on
purpose. Do not spend more time on it until DeepForest training exists.**

**What works and is done**

- Project is public: `comet.com/vicky-sharma-1971/bird-annotation-recovery`
- 5 runs upload cleanly: the timeline, 2 detection methods, 2 matching configs
- 8 figures and the docs upload as assets; the eval CSVs upload as data tables
- `docs/milestones.csv` is the single source both the figure and the uploader
  read, so they cannot drift apart
- Credentials resolve from `.comet.config` (gitignored) or `COMET_API_KEY`; the
  script exits 0 without them, so it can never break CI

**What does not work, and why**

- **Comet's auto-panels.** It draws a chart per metric name and needs many
  points on a step axis, meaning training epochs. Seven milestones render as
  seven dots. Six logging structures were tried: per-frame series, per-band
  series, scalars, scalars moved to `log_other`. None of them changed it.
- **"No data for this chart"** on the project Panels view. Auto-generated
  panels keep referencing metric names from an earlier structure. Changing the
  metrics again would just move the problem.
- **A hand-built saved view crashed the Comet UI.** `api.create_view` accepts a
  `chart_state` JSON blob whose schema is undocumented; the guessed schema was
  wrong and the view returned "Something went wrong". Reverted in `24de559`. A
  stale view named "Project overview" may still exist in the project and can be
  deleted from the UI, since the API offers create and read but no delete or
  update. **Do not retry this without a real schema example to copy.**

**Consequences for how results are shown.** The progress chart is drawn by
`scripts/make_timeline_figure.py` and lives in the README, which renders
reliably on GitHub. Comet hosts it as an image rather than generating it. When
showing the project to anyone, lead with the README.

**Also note:** every `log_to_comet.py --clean` archives the old runs and mints
new experiment keys, so any direct experiment link goes stale. Share the
project URL, which is stable.

**When to pick this back up.** At the DeepForest fine-tuning stage. DeepForest
is PyTorch Lightning underneath, so `CometLogger` passed to `create_trainer`
gives loss curves per epoch, precision and recall over time, and checkpoints
versioned against the config, which is what the panels are actually built for.
The setup is already in place, so those runs will land next to this history.

## Hand-Labelled Measurement (2026-08-07): detection scored per dot

Full write-up: `docs/labelling_findings.md`. Labels: `data/labels/*.json`.
Harness: `scripts/label_dots.py` -> `scripts/labeller.html` -> `scripts/eval_localisation.py`.

Every detection figure before this compared a **count** to a **count**. That cannot
separate "found all 61 markers" from "found 40 and invented 21" from "found 61 on
empty water" — all three score 61. So precision, recall, placement error and per-dot
class accuracy had never been measured at all.

**345 dots were hand-labelled across 6 screenshots** (2 sparse, 4 medium; dense
deliberately left for later). Labelling is transcription rather than judgement: the
annotator drew the marker in 2010, this only records where.

### Result

| metric | value |
|---|---|
| precision | **0.138** |
| recall | **0.345** |
| F1 | **0.197** |
| placement error, median | **1.83 px** |
| correct / false positive / missed | 119 / 745 / 226 |
| classification, per dot | **0.373** (118 dots) |

| frame | band | labels | survey | detected | P | R | F1 |
|---|---|---|---|---|---|---|---|
| `08May10…0115` | sparse | 36 | 36 | 262 | 0.076 | 0.556 | 0.134 |
| `10June10…0076` | sparse | 9 | 9 | 128 | 0.039 | 0.556 | 0.073 |
| `11June10…0154` | medium | 72 | 72 | 98 | 0.388 | 0.528 | **0.447** |
| `14June21…238` | medium | 71 | 71 | 37 | 0.297 | 0.155 | 0.204 |
| `18June21…06389` | medium | 103 | 119 | 22 | **1.000** | 0.214 | 0.352 |
| `25June10…0401` | medium | 54 | 54 | 317 | 0.073 | 0.426 | 0.124 |

Leave-one-out is stable: dropping any single frame moves precision only within
0.106-0.176 and recall within 0.297-0.401, so no one frame drives the result.

### What it changes

**1. There are two opposite failure modes, not one.** Sparse scores P 0.06 / R 0.56
— it finds about half the markers and buries them in noise. Medium scores P 0.34 /
R 0.32 — cleaner output, most markers missed. `06389` produced 22 detections, **all
22 correct**, and missed 81. A single fix cannot serve both, which is also why three
attempts at the marker-size estimator each improved one band and regressed the other.

**2. Placement is already good and is not the problem.** 1.83px median, consistent
across all six frames. Given `learnings.md` #18 (920 accurate annotations beat 3,851
with ~30px error), the recovered dots sit well enough to train on; there are simply
too few of them and too much alongside them.

**3. `0076` settles the sparse question spatially.** A mangrove colony: 9 real
markers, 128 detections, 5 correct. The residual is vegetation texture, which the
report had inferred but never shown.

**4. The label-swap failure is systematic, not anecdotal.** `learnings.md` #30
described WHIB site vs WHIB bird from one example. Measured: **25 WHIB site ->
WHIB adult**, plus 7 SNEG and 7 TRHE of the same `site` -> `bird` collapse. A second
and separate cause is legend rows whose name never read ("row 2 (green)"), costing
17 of 118 dots regardless of how good the matcher is.

**5. The labels are trustworthy.** Five of six frames matched the survey count
exactly (9/9, 36/36, 72/72, 54/54, 71/71), on blind frames as well as seeded ones.
Two frames were labelled with no seeds shown, as a control: if seeing the detector's
output had biased the labelling, those would have come out short. They did not.
`06389` gave 103 against a survey figure of 119, which is either fused markers or
another `category_sum` mismatch of the kind already proven on `00097`.

### Marker-size estimator: diagnosed, measured, not merged

`subtract._marker_area` feeds the split trigger, the split count and the accept band
`[0.35*modal, 3.0*modal]`. It sized markers from a distance transform, whose peak is
the **stroke half-width** on outline glyphs (rings, plus signs, asterisks — most of
this symbology), not the footprint. It floored to an area of 4.0 on **11 of 33**
benchmark frames, and those frames carried median |log2 ratio| **1.26** against 0.23
elsewhere. `06389` estimated 113.1, banding out its own real markers.

| | baseline | modal blob | hybrid |
|---|---|---|---|
| median \|log2\| | 0.47 | **0.44** | 0.53 |
| within ±25% | 19 | **22** | 18 |
| within ±10% | 5 | **15** | 11 |
| miss (<0.5x) | **5** | 9 | 8 |
| dense band | **1.14x** | 0.83x | 0.85x |

None passed the gate cleanly, so none shipped. Parked on branch
`exp/marker-size-estimator` (`85de2d8`); **`main` is unchanged**. The hand labels then
confirmed the diagnosis from the other side: `06389`, whose band excluded real
markers, scores precision 1.00 — the detector was too strict, not producing rubbish.

**The band constants were deliberately left alone.** They were calibrated against an
estimate roughly 4x too small, so their effective lower bound was near 0.09 of a real
marker. Re-deriving them by tuning against the 63-frame benchmark they are scored on
is the mistake this project already made twice, with the count prior and with legend
self-recovery. They should be re-derived against the labels.

---

## Chrome-Mask Fix and Frame Selection (2026-08-08): detection finished

### The bug the count metric was blind to

Tracing all 345 labelled markers through `subtract.extract_annotations` showed where
they actually go. **92 of them (26.7%) were being deleted by the chrome/dialog mask.**

`_ui_regions` judged a region to be window chrome from the **median saturation over the
whole `MORPH_CLOSE`d component**. Closing bridges a scattered colony into a single
region spanning a quarter of the frame, so that median measures the **background
between** the markers rather than the markers. The region reads dull and everything in
it is discarded. On `14June21…238` the offending region covered 25% of the frame, read
median saturation 17 against a threshold of 40, and swallowed **53 of 71** markers.

Fixed by judging chrome on the region's **ink pixels**. Measured over the 21 candidate
regions on the labelled frames, as the fraction of ink at least 100 saturated: regions
holding real markers run **10.9%–94.9%**, dialogs run **0.0%–2.5%**, and nothing falls
between. The floor sits in that empty gap at 5%. `ui_area_frac` and `ui_max_sat` were
deliberately left alone — loosening either reopens the documented failure.

```
                        before   after
recall                   0.345   0.678
markers reaching ink     73.0%   99.4%
classification per dot   0.373   0.544
```

Also fixed: split sub-dots inherited the parent cluster's bounding box, so every later
size or elongation test on them read the run rather than the marker. They now carry
their own footprint. Gating splits on the modal size band was tried and reverted —
sparse recall 0.56 → 0.10, because `modal` is itself the unreliable quantity.

### Frame selection: the result that matters

Five ways of separating real markers from background were measured against the labels.
**None generalised** — the discriminative direction reverses between frames. They are recorded below so the ground is not covered twice.

What works instead is arithmetic. Correct detections can never outnumber the dots on
the image, so `P ≤ present / reported`. A frame reporting 7x what it holds cannot exceed
14% precision however it is filtered afterwards, and the ratio needs **no labels**.

Verified on three frames spanning two bands and 10 to 186 dots:

| frame | ratio | precision |
|---|---|---|
| `18May15…00825` | 1.00 | **1.00** |
| `19May18…00620` | 0.99 | **0.82** |
| `18June21…06389` | 0.62 | **0.74** |

against high-ratio frames at 0.04–0.14. Applying the benchmark's per-band pass rates to
the band distribution of all 18,252 screenshots: **~48% of images pass, holding ~72% of
the corpus's dots — about 2 million annotations.** That is well beyond what deliverable
2 needs.

**So detection is finished.** Not because every frame is clean, but because the pipeline
now reports which frames it can handle, and those hold most of the data.

### Two things recorded, not fixed

- **Dense band has zero per-dot labels.** Selection is verified on sparse and medium
  only, and dense holds 55% of the corpus's dots. A page is generated for
  `17May10Camera2-Card1-5745` (dense, ratio 0.94).
- **The accepted set skews dense** — 89% of dense frames pass against 28% of sparse.
  Check species and habitat coverage when building the dataset.

### Legend parsing measured

Prompted by a frame whose dialog is plainly visible but was not found. Over 60
benchmark frames: **7 (12%) find no dialog at all**, 1 finds rows but reads no names,
52 work. On those 12% classification is impossible whatever the matcher does, and it is
also what stands between frame selection and being fully self-contained. This is the
next task.

### Corpus size discrepancy

The project description says "340,000+ bird observations". Measured from the manifest,
the corpus holds **2,810,895** dots by `category_sum` and **2,638,535** by
`total_birds` — roughly 8x higher. The figure also appears in the README. Worth raising.

---

## Next Steps

### Immediate (start here after the break)

Work on **sparse and medium only**. `learnings.md` #4: 65% of the corpus has fewer
than 100 birds, so these two bands are the common case. Dense has **zero** hand
labels and must not be tuned until it does.

1. **Give sparse precision.** P 0.06 means 94% of its output is noise, and `0076`
   shows the source is vegetation texture. The legend palette is the obvious filter
   and is also the one `docs/TASKS.md` warns against shipping globally (on 0076 it
   collapsed 129 detections to 1 against 9 real markers). Scope it to the sparse
   path and score it with `eval_localisation.py`, not the count metric.
2. **Give medium recall.** R 0.32, and `06389` shows why: the accept band excludes
   real markers. This is where the parked estimator work resumes.
3. **Re-derive `0.35 / 1.5 / 3.0`** from the labelled markers — the labels answer
   "how small can a real marker's ink be", which counts never could.
4. **Legend name OCR** before any more matcher work: 17 of 118 classification errors
   are dots whose class name never read.
5. **The `site` vs `bird` split**, now quantified at 39 instances across three species.

### Deferred on purpose

- **Dense band.** Four frames remain unlabelled: `0296` (412), `0507` (673), `0081`
  (234), `00097` (450, expected ~0 real dots). Label `00097` and `0081` first when
  picking this up — `00097` costs ten minutes and proves the ground-truth artifact.
- Downstream stages (validate, map, export, train) still exist only in the notebook.

### Done (2026-06-29) ✅
- ~~Retune legend parser for full resolution~~ → scale-adaptive, rows within ±1 on all 4
- ~~Robust dialog localization~~ → `locate_dialog` box-based, 14/14 (100%)
- ~~Template matching per class~~ → `classify.assign_classes`
- ~~Re-run detection with separate classes~~ → same-colour classes separated
- ~~PR/plot explanation + synthetic fake-dots figure~~ → `results/figures/` + meeting doc

### Done (2026-06-30) ✅
- ~~Aerial shape / count-prior~~ → A +35 pp, C +49 pp; D flat; 146 tests passing
- ~~Count-OCR strip clip (145 px)~~ → confirmed working for img_02 (224 px), img_03 (151 px); does NOT apply to B (B strips are 110 px, B's count-OCR problem is blank cells, not strip position)

### Done (2026-07-10) ✅: matching rework
- ~~Anchor colour to the dialog's own palette~~ → Lab, per image, with reject
- ~~Background removal~~ → hue-consistent glyph masking
- ~~Try shape options empirically~~ → NCC won the ablation; shape-name boost removed
- ~~Separability~~ → within-colour D 55→**76%**, A 56→**83%**
- ~~Dots detected inside the dialog~~ → exclude re-applied after cluster splitting

### Done (2026-07-20) ✅: ground truth + difference-based detection
- ~~Establish what a survey count means~~ → **`category_sum`**, verified against the
  tool's own Total Count; `total_birds` retired as ground truth
- ~~Rebuild the benchmark from identifiable images~~ → 63 stratified pairs
- ~~Alignment~~ → `src/align.py`, **96.7%**, 0.38px, refuses rather than mis-warps
- ~~Difference-based detection~~ → `src/subtract.py`, **8.40x → 1.24x** median error

### Immediate (next): close the detection gate, then commit

**Status (2026-08-04): this gate was met and the work is committed on PR #7.**
The agreed bar was that detection must be satisfactory before committing. Overall is now **1.24x** median; the last soft spot is
**sparse (2.13x)**, addressed in part by the 2026-07-24 saturation floor.

1. **Sparse over-detection (2.13x, down from 2.96x).** The dominant low-saturation
   residual (water/mudflat/grey label panel) is now gated by the saturation floor.
   The remaining residual is **two distinct things**:
   a. **Vegetation/tree texture**: the **legend palette** as a colour reject *will*
      help here (foliage is off-palette). Belongs in the Phase-3 `classify.detect_dots`
      integration with a fallback path, NOT a hard dependency in `subtract`.
      Note: legend palette CANNOT help the red-text case (label-red = marker-red).
   b. **Marker-red label text + transect lines**: a standing limitation on a few
      frames. Depth / peak-count / elongation were all ruled out (bold text is deep
      and compact). Do not chase a geometric hack that risks the dense band.
2. **Spatial verification, not just counts.** 18≈18 can happen for the wrong
   reasons. Render overlays for a handful per band and confirm circles sit on real
   markers. (Done ad hoc so far; make it a script.)
3. **Wire subtraction into `classify.detect_dots`** as the primary path with the
   colour path as fallback when alignment fails, currently they are separate.
4. **Then commit** and update PR #3 / open a follow-up PR.

### Superseded: the "fix B" plan (2026-07-01)
The plan below was written when B's apparent 10.8x over-detection looked like the
priority. That figure was measured against `total_birds` and is unreliable. B's
true dot count may be up to 808, not 117. **Do not act on this section without
re-measuring against `category_sum` first.** The B-specific reasoning is retained
because the dialog-cropping root cause it uncovered was real and is fixed.

1. **Fix A, Total Count header fallback** (`src/legend.py` → `attach_class_names`):
   when all per-row counts are None/0, read the "Total Count: NNN" text from the
   dialog's left-panel header via Tesseract. Use NNN as a total cap for
   `select_by_count`. This is the highest-impact fix for B (17 of B's 1278 dots
   are kept with correct count-cap vs 1278 with no cap now).

2. **Fix B, Tighten blank-cell heuristic** (`src/legend.py` → `_ocr_count`):
   `dark < 6` threshold is too loose for cells adjacent to table borders.
   Raise to `dark < 15` or add a contrast check so empty cells reliably return
   None/0 instead of misread integers. Eliminates the "site=444" phantom count
   that inflates the denominator from 117 to 449.

3. **Re-verify B count-OCR improvement**: the "57%→79%" claim from 2026-06-30
   needs re-examination. B strips are 110 px (clip never fires). Possible the
   improvement was measurement noise or came from a different change; needs a
   fresh baseline measurement after Fix B above.

4. **Per-class validation vs CSV at scale**: once B is fixed, validate A/C/D/B
   recall against CSV ground truth at class level.

5. **Wire recovered dots into DeepForest export** (train/test split by colony).

### Follow-up (after cleanup)
5. Convolutional template matching refinement after color segmentation
6. Improve SIFT success rate from 57%, tune parameters or try SuperGlue
7. SAM 3 to refine dot positions → tighter bounding boxes → better training signal
8. Refactor `validate.py`, `map_coords.py`, `export.py`, `pipeline.py` into tested modules
9. Full pipeline run on all 18,304 images → ~340,000 recovered annotations

---

## Open Questions / Blockers

### As of 2026-07-20 (partly resolved since)

> Updated 2026-08-04: the saturation floor cut sparse over-detection from 6.07x
> to **2.13x** and the detection gate was met. What remains of this entry is the
> residual marker-red label text, which is a standing limit on a few frames.

- **Sparse frames over-detect 6.07x**: at the time, the one thing blocking the
  detection gate and therefore the commit. Cause is understood (label text, transect lines,
  residual water are all genuine *ink* but not dots); the principled filter is the
  legend palette, which is coupled to legend parsing.
- **Legend parsing is less reliable than reported.** `locate_dialog` returned a box
  on 19/21 sampled screenshots (90%), but at least one was a clear false positive
  (a 903×605 box at x=0 yielding 31 rows), and it failed on 3/8 in a second sample
, including an image whose dialog is plainly visible. The 90% figure counts
  false positives as successes and should not be quoted as accuracy.
- **Alignment fails on ~3%** (2/60), concentrated in 2011/2012 and Barataria Bay.
  Those fall back to colour detection. Root cause not yet investigated.
- **Ground truth is counts, not locations.** The survey gives per-class counts per
  photo, never dot coordinates. Count accuracy is measurable at scale; **per-dot
  localisation accuracy is not**, without hand labels on a small subset. Nothing
  currently verifies that a correct count is correct for the right reasons beyond
  ad-hoc visual checks.
- **Classification is untouched by this work.** Subtraction improves *where* dots
  are; *which class* still runs through `classify.py` and still needs the legend.

### Earlier (pre-2026-07-20, some now superseded)

- **Dialog legend parsing robustness**: icons are small and may vary across annotators/years; scale-adaptive parser tested on 14 images (100% localization), but full S3 diversity not yet covered.
- **Shape vs color ambiguity**: ~~need to confirm from sample~~ RESOLVED: (shape+color) → class confirmed per-image via parse_legend; count-prior handles "unknown" shapes at w=0.9.
- **SIFT failure on 43% of images**: root cause not fully understood; may be featureless aerial sections.
- **B, blank Count column (root cause, diagnosed 2026-07-01)**: B's annotator did not fill in per-row species counts. All 16 count cells are blank. OCR correctly returns None for blank cells (mostly), but the `dark < 6` heuristic misreads the "site" row as count=444 due to table-border dark pixels. The "Total Count: 117" header IS present and readable but not currently used by the pipeline. Two fixes needed: (1) read "Total Count" header as fallback; (2) tighten blank-cell heuristic. The detection itself is fine, red (244, quality 0.54) and blue (246, quality 0.51) are almost certainly real BRPE and LAGU birds.
- **B, 145 px count-OCR clip does not help B**: B's strips are 110 px wide (below the 145 px threshold), so the clip never fires. The B "57%→79%" improvement claimed on 2026-06-30 needs re-verification after the blank-cell fix.
- **Grey marker detection not implemented**: grey "site" squares have low HSV saturation (≈0 S channel) and cannot be detected by the colour-segmentation pipeline (sat > 80 threshold). If grey-square markers appear across other images, a separate low-saturation detection pass will be needed. For now, exclude grey classes from the recall denominator when counts are missing.
- **Count-OCR on tiny digits**: img_08 and similar small-dialog images get Σcount=0; digit-recognition failure at ≤10 px, not a strip-position error. Possible fix: stronger upscale/sharpen or digit CNN. Lower priority than B's blank-cell fix.

---

## Meeting Log

| Date | Key decision / outcome |
|------|----------------------|
| Pre-GSoC | Studied 25 images, 49K CSV rows before writing code |
| Early GSoC | Abandoned OCR; switched to CSV as ground truth |
| Early GSoC | Disabled text filter; widened HSV bins; measured from 1,199 dots |
| Early GSoC | Found bfloat16 corruption; fixed with explicit autocast disable |
| Early GSoC | SIFT-only training confirmed: position accuracy > data quantity |
| ~2026-06-26 | Josh feedback Round 1: keep classes separate; PR explanation needed; give both numbers for comparisons |
| ~2026-06-27 | Josh feedback Round 2: proportional category assignment is wrong |
| ~2026-06-27 | **Agreed plan:** self-contained detector reads class map from dialog box per image; CSV for validation only |
| 2026-06-29 | Built full auto pipeline: localization 14/14, species 15/15 (D), recall ~100% (cluster splitting), precision fix (B 1278→92). Fixed boundary bug (50% aerial cut). 5 figures + meeting doc prepared. |
| 2026-06-30 | Opened **PR #3** for the legend module: added test suites (143 passing), fixed CI (UTF-16→UTF-8, added scipy), centralized legend config into `config.yaml` (behaviour-identical). Sent Josh a Slack update asking for review. |
| 2026-06-30 | Incremental gap work while awaiting PR #3 review. Count-OCR: 145 px clip added (helps img_02/img_03, NOT B, whose strips are 110 px). Count-prior (w=0.9): A +35 pp, C +49 pp, D flat, attributable-dots metric on all 4 images. 3 new count-prior tests; 146 total passing. fig6+fig7 generated. GSoC meeting DOCX updated with 2026-07-01 section + quick-reference tables. |
| 2026-07-01 | Deep diagnostic on image B. Found: (1) B's Count column is entirely blank, annotator never filled per-row counts; (2) "site=444" is an OCR misread of an empty cell via `dark < 6` heuristic picking up table borders; (3) B strips are 110 px, the 145 px clip does not fire; (4) quality distributions show red (244 dots, 0.54) and blue (246 dots, 0.51) are real BRPE/LAGU birds; green (495, 0.12) is vegetation FP. Total Count: 117 visible in dialog header but not read. Decision: fix B before moving to next pipeline module, real project, not prototype, needs to scale to 18k images. Two fixes planned: read Total Count header as fallback; tighten blank-cell heuristic in `_ocr_count`. |

| 2026-07-10 | **Matching rework shipped.** Josh's 3 asks implemented: per-image Lab colour anchoring, background removal, NCC shape matching. Ablation chose NCC and **removed** the shape-name boost. Within-colour separability D 55→76%, A 56→83%; total count error down on all 4 images. Two reporting errors caught before shipping: capped-recall rewarded over-assignment (dropped), and the trust caveat was backwards (A's legend covers only 33% of true, D's 92%, D is the reliable image, not A). Fixed `_dot_centers` bug: 54 dots were being detected inside image A's dialog. |
| ~2026-07-17 | **The Water Institute (@cronosnull) opened discussion #6.** Independently confirmed LAB; proposed aligning the clean high-res original and **subtracting** so annotations come from image difference, not colour thresholds. Reframed the detection stage. |
| 2026-07-20 | **Ground truth corrected, the most consequential finding so far.** Read the counting tool's own "Total Count" field off four dialogs by eye: dot count is **`category_sum`**, not `total_birds` (mean error 0.25 vs 14.75). `total_birds` excludes chicks and undercounts by up to 57%. Found a real bug: 4 distinct chick-like columns, `ChicksNestlings` was being omitted. **This invalidates every pre-2026-07-20 accuracy figure** and the four study images' "true" counts, which could not be mapped to survey rows at all. Rebuilt the benchmark as 63 stratified pairs (7 years × 3 density bands, 40 colonies, 16 regions). |
| 2026-07-20 | **Difference-based detection built.** `src/align.py` (96.7% success, 0.38px median error, refuses rather than mis-warping) and `src/subtract.py`. Detection error **8.40x → 1.46x** median; dense band **1.01x**; sparse still 6.07x. Four traps documented (edge diff, water=luminance, grey dialog needs the full diff, never mask a carpet by size) plus distance-transform sizing. 163 tests pass. **Branch deliberately left uncommitted** until detection is satisfactory. |
| 2026-08-07 | **Detection measured per dot for the first time.** Built a labelling harness (`label_dots.py` + `labeller.html` + `eval_localisation.py`) and hand-labelled **345 dots on 6 frames**. Detection is **P 0.138 / R 0.345 / F1 0.197**, against a count ratio of 1.24x on the same pipeline — the count is not wrong, it answers a different question. **Placement is not the problem** (1.83px median). Two opposite failure modes: sparse P 0.06 / R 0.56 (vegetation noise, shown spatially on `0076`: 9 markers, 128 detections, 5 correct), medium P 0.34 / R 0.32 (`06389`: 22 detections, all correct, 81 missed). Classification **0.373 per dot**; the WHIB site/adult swap from learnings #30 is now measured at 25 instances plus 14 more across SNEG and TRHE, and 17 further errors are legend rows whose name never read. Labels validated: 5 of 6 frames matched the survey count exactly, on blind frames as well as seeded. Separately diagnosed `_marker_area` (distance transform measures stroke half-width on outline glyphs; floored on 11 of 33 frames, those carrying 5x the error) and measured three replacements — none passed the gate, all parked on `exp/marker-size-estimator`, `main` unchanged. |
| 2026-07-24 | **Sparse over-detection fixed, saturation floor.** Diagnosed spatially first: sparse false ink is low-saturation residual (water/mudflat/grey label panel) that poisons the modal-size estimate, not primarily red text. One legend-free gate in `subtract.dot_candidates` (drop blobs below `marker_sat_min=50` before sizing) improved every band: sparse **2.96→2.13x**, medium **1.59→1.24x**, overall median **1.33→1.24x**, mean **3.07→2.08x**; dense safe. 164 tests (added a drop-path red test + eval miss/zero/over hygiene). Found a **GT artifact** (00097 = "no photo coverage", truth 450 = text-box estimate, 0 dots on image). Residual sparse = two causes (vegetation texture → Phase-3 palette helps; marker-red label text → standing limitation). Built mentor report `docs/status_for_josh.html`. |

---

## Repo Map: what each file is for

**Pipeline modules (`src/`)**

| File | Role | State |
|---|---|---|
| `legend.py` | Find the dialog, parse legend rows → (marker, colour, shape, template, class name, count) | Working; localisation ~90% but with false positives |
| `classify.py` | Assign aerial dots to legend classes: Lab colour anchoring → NCC shape within colour | Working; within-colour D 76% / A 83% |
| `align.py` | **NEW**: register screenshot ↔ clean original (SIFT+RANSAC, quality gate) | 96.7%, 0.38px |
| `subtract.py` | annotation ink as image difference; `dot_candidates` turns ink into dots | 1.24x median error; sparse (2.13x) weakest |
| `decompose.py` | Original screenshot splitter (Stage 1) | **Legacy**: superseded by `legend.locate_dialog`; only its own tests + legacy `run_legend.py` use it |
| `detect.py` | Original colour detector (Stage 2), needs CSV counts | **Legacy**: not wired into anything live; only its own tests. The colour baseline/fallback in use is `classify.detect_dots`, not this |

**Scripts (`scripts/`)**: none of these are imported by `src/`

| File | Purpose |
|---|---|
| `build_manifest.py` | Survey CSV → `data/cache/manifest.csv` (ground truth, **evaluation only**) |
| `build_benchmark.py` | Select + download the 63 stratified pairs |
| `probe_groundtruth.py` | Phase-0 probe: subtraction vs published counts, writes overlays |
| `probe_totalcount.py` | Phase-0 probe: crops dialogs so Total Count can be read by eye |
| `eval_alignment.py` | Alignment success rate by year / band / region |
| `eval_detection.py` | **The gate**: old colour vs new subtraction, vs `category_sum` |
| `eval_matching.py` | Legend self-recovery confusion matrix (flatters the method; see learnings #29) |
| `run_legend.py` | Batch legend parse over a folder |
| `label_dots.py` | **NEW**: builds a click-to-label page per frame, seeded from the subtraction path, with the legend's own glyphs as the class palette |
| `labeller.html` | **NEW**: the labelling UI — zoom, class palette, sweep grid, autosave |
| `eval_localisation.py` | **NEW, the real gate**: precision / recall / F1 / placement error / per-dot class accuracy against `data/labels/` |

**Data (all gitignored, all regenerable)**

- `data/cache/manifest.csv`: 49,204 rows, ground truth for 18,304 screenshots
- `data/cache/benchmark.csv`: the 63 selected pairs
- `data/cache/align_cache.json`: cached transforms; **delete after editing `align.py`**
- `data/fixtures/pairs/`: downloaded screenshot/original pairs (~470 MB)

---

## Session 2026-08-14: building out classification

Detection was finished going in and did not move: precision, recall and placement were
re-checked after every change and read the same each time. All of the work below is in
`classify.py`, `legend.py` and `select.py`.

### What shipped

| | | per-dot accuracy |
|---|---|---|
| Class identity keyed on the **legend row** rather than the OCR'd name | one dialog carries two rows that both read `ad` | a metric correction, and nothing else could be measured without it |
| `src/select.py` | the frame-selection rule existed only in prose, and was one-sided | two-sided band, 10 tests |
| Capacity from the dialog's own Count column | the legend said `ROSP bird = 0` and the pipeline put 76 dots there | 0.443 → 0.696 |
| Per-frame lightness correction | a legend glyph reads L=243 where the same marker in the aerial is far darker, so 152 of 183 dots on one frame were rejected before classification | 0.555 → 0.717 |
| Sole-candidate dots no longer score a flat 1.0 | having one candidate says the legend holds one class of that colour, not that the dot is a good marker. Measured: those dots are 69% real against 80% for contested ones | 0.717 → 0.725 |
| Rows parsed below the table are capped | a row landing on the scrollbar or on the photograph below the dialog had no readable count, so it was unlimited, and two such rows took 49 detections | 0.725 → 0.766 |
| Colour offsets measured **per row** | the drift from legend glyph to aerial marker follows the glyph, not the frame: one frame needs `a = −29.4` on one row and `−11.0` on another | 0.766 → 0.789 |
| Colour ranks as well as gates | a dot on a row's colour and one that scraped into the margin used to compete as equals | 0.789 → 0.828 |
| Blocked dots retried against any row still free | 25 dots on one frame had a single candidate row that the dialog genuinely counts as zero | 0.828 → 0.850 |
| Count sanity bound: a row cannot hold more dots than the frame has detections | correct, and worth keeping, and it changed nothing. An inflated cap never binds, so treating it as unread has the same effect | 0.850 → 0.850 |
| Species matcher accepts a 3-letter token | OCR drops the leading character, and `RPE` never reached the matcher | species resolution 0.720 → 0.771 |

Tests 190 → 211.

### What was reverted, and why

Five ideas were measured and dropped. They are in `docs/learnings.md` #37–#49 and in the
README's *what did not work*, so the ground is not covered twice.

The one worth repeating here: **taking the Name|Count divider from the first gridline
right of the marker is the correct geometry, and it breaks classification.** The current
code takes `gridlines[1]`, which is wrong on 14 of the 25 frames — on one, the divider
lands left of the marker and the strip read as the Count column is really the Name column,
so a dialog plainly showing 93, 70, 11, 23 comes back as 3, None, 0, 0. That frame scores
0.193. The fix improved species resolution 0.771 → 0.858 and dropped a good frame from
0.861 to 0.634, then to 0.600 on a second attempt that moved the two strips separately.
Moving the name boundary changes the parsed `class_name`, and row identity is keyed on
that. It is a classification change wearing a legend-parsing disguise.

### Ground truth

743 dots were labelled during the session, bringing the set to **1,648 dots across 12
frames**. The three new frames were chosen to test the label-free proxy rather than to
add volume: the survey cross-check predicted 1.000, 0.995 and 0.403 for them, and they
measured 0.877, 0.730 and 0.193 by hand. The ranking holds and the values read high, so
the proxy can say which frames are weak but not how weak.

**Per-dot accuracy went 0.850 to 0.781 when those three frames joined.** The pipeline did
not change between the two readings; the first covered four frames and the second seven.

### Figures

`scripts/make_classification_figures.py` is new and `scripts/make_classify_figure.py` was
extended. Both run the live pipeline. The per-class figure now rings each sample patch
green where a hand label agrees with the assigned class, red where it disagrees, and white
where no label sits there, with the frame's whole split printed in the title.

That ring exposed something counts hide. On the showcase frame, the two legend rows
holding two dots each match the dialog's stated count exactly and are wrong on every dot.

---

## Session 2026-08-18: review of PR #10, and the export decisions it settled

No code changed. This session answered a mentor review and measured the three things the
answers rested on.

### The red ring Josh asked about was capacity, not colour

He asked why the figure's first sample is wrong when colour should have caught it. It is
not a colour failure. For the dot at `(500,238)` on `5745`, colour never offered the class
it received:

```
candidate rows   ROSP site 12.9    ROSP bird 25.3    BCNH site 10.9
not a candidate  LAGU sit  40.0
```

All three candidates were unavailable: ROSP site 62 stated and 62 assigned, BCNH site 2
and 2, ROSP bird stated 0. `_BLOCKED_RETRY` then widened to every row inside
`_COLOR_REJECT` and picked LAGU sit. The hand labels put 60 real dots on ROSP site, so two
wrong ones took the last slots and pushed a real one out.

Josh's response was that a dot should not be able to fall to a class whose colour was
rejected. Measured with the existing flag, he is right and it is nearly free:

```
BLOCKED_RETRY=1   0.703   (725/1031, pooled over 12 frames)
BLOCKED_RETRY=0   0.693   (714/1031)
```

Eleven dots. The default becomes `0`, folded into the mapping work.

### Species coverage is worse per dot than per row

The 0.771 figure counts legend rows. The CSV's label column is written per dot.

```
rows : name 197/218  = 0.904    species 168/218  = 0.771
dots : name 5865/6423 = 0.913   species 4160/6423 = 0.648
```

Four dense frames carry nearly all of the gap: `0216` 0/1033, `0730` 0/430, `0406`
105/371, `0242` 0/51. Their failing rows read as `ad` or `imm`, the category word without
the species code in front of it, which is the Name-column geometry already on the
do-not-retry list. Across the four, **22 rows are unresolved and naming them recovers
about 1,500 dots**, so the unit of repair is the row.

### `nest` and `ad` are invented by the matcher, not read by OCR

The parser steps past the table's last real row at the row pitch, so on `5745` row 12
lands on the horizontal scrollbar and row 13 on the dialog's bottom edge. Raw OCR there is
`'es ee'` and `'ee'`. `_parse_class_text` fuzzy-matches the leading token to `_CATEGORIES`
at `max_dist=2`, and `ad` is a two-character word in that list, so any one- or
two-character noise becomes `ad`. `'|'`, `'4'` and `'a'` all return `ad`.

Accuracy cost is zero, because `tail` capacity already gives both rows nothing. The damage
is to the figures and to species coverage.

### The export label policy, decided

Export every dot, and say which labels are real:

```
label             LAGU site  |  Bird
species_resolved  true       |  false
frame, legend_row            <- carried on every row
```

Dropping the unresolved dots would discard good coordinates over a text problem that
detection does not care about, since DeepForest's bird model trains on a single class.
Writing `Bird` silently would hide which species are real. `frame + legend_row` is the
stable key, so one row corrected fixes every dot on it. That is also the cheapest form of
the review Josh proposed: 22 rows instead of 1,500 dots.

The first CSV covers the 25 frames, which is the sample he asked to look at. At corpus
scale (~8,500 images pass selection) the by-hand argument stops working, and the two
candidate levers are the Name-column geometry fix and propagating names across frames of
the same survey. Neither is measured yet.

---

## Session 2026-08-20 to 21: the dataset, at two scales, and what training says about it

The pipeline stopped at classification. Every dot it produced sat in screenshot
coordinates, and a model trains on the originals. Two stages were missing between a
measured pipeline and a dataset anyone can use. Both now exist, and the dataset has been
built twice: once on the 25 benchmark frames, once on 413 drawn fresh.

### Mapping, and the one trap in it

`align.H` maps screenshot pixels to work-scale original pixels, not to the original,
because the original is downscaled before SIFT runs. The full-resolution coordinate needs
`perspectiveTransform(p, H) / res.scale`.

Dropping the divide returns coordinates that are internally consistent, plausibly sized,
and wrong by a factor of `scale`. On `5745` that is roughly half. Nothing raises. It was
caught by drawing the dots on the photograph and looking, which is the only way it could
have been.

Verified on real frames: `5745`, `0027` and `00825` map fully in bounds, and at 4x zoom
the mapped detections and the mapped hand labels both sit on the birds along the
vegetation edge.

### Box size, and two wrong answers before the right one

The survey recorded a point per bird and never an extent, so no box size can be read off
the data. It has to be measured, and per frame: the EXIF shows focal lengths from 28mm to
300mm across these frames, so the same species spans 10px on one photograph and 41px on
another.

**Spacing between dots** was the first attempt. It measures crowding: nearest-neighbour
distance runs 13.6px to 215px while the birds differ about twofold.

**An equivalent diameter** was the second. For a bird twice as long as it is wide, that
diameter is 0.71 of the length, so every box came from 70% of the bird. On `426` the ibis
measures 42px long where the equivalent diameter reads 20px, and 0.61m of White Ibis at
about 1.5cm per pixel is 41px.

`src/birdsize.py` now takes the long side of the component's minimum-area rectangle.
Dividing known body lengths by what it measures gives 1.3 to 4.0 cm per pixel, and the
EXIF explains that spread rather than merely agreeing with it.

Judging the result from three frames is how the flat 100px box survived as long as it did:
it looked defensible on the sparse frames and was four to eight times too large on the
dense ones. The figure now draws all 25.

### The dataset, and the funnel written down

`scripts/export_dataset.py` recomputes the funnel rather than quoting it, because the 25
had been repeated for weeks without living in any code:

```
63  benchmark frames -> 60 cached -> 31 pass selection -> 28 dialog -> 25 exported
```

Writing it down found a bug that would have shipped. The three frames excluded for a wrong
dialog box had their names reconstructed from the trailing digits in the docs, the guessed
prefixes matched nothing, and `0507` put 113 dots into the dataset from a frame already
known to be wrong.

A first pass also concluded the documented funnel double-counted, and that was wrong: it
read detection counts from `results/eval_detection.csv` rather than running detection, and
that file predates the current detector. **A cached metrics file is not a substitute for
running the stage.**

### Scale: 413 frames, and the numbers that did not move

Every figure until now came from those 25 frames, and those 25 are the frames every
decision was checked against. The scaled run took 1,197 candidates stratified across seven
survey years and three density bands, minus the 12 carrying hand labels.

```
1,076 pairs -> 458 pass selection (43%) -> 413 exported -> 118,270 boxes
```

```
                    25 frames      413 frames
species resolved      0.648          0.650
box measured on        96%            98% of frames
distinct species        19             45
distinct classes        44            380
```

Selection passing 43% rather than 52% is the honest figure: the small benchmark drew three
frames per year-and-band cell and this draws 57.

One pattern the small set could not have shown. Species resolution varies by survey year,
not by chunk: 0.751 in 2013 against 0.484 in 2015, over 87,000 dots. The dialog rendering
changed across years, and `attach_class_names` reads gridline geometry. Worth knowing
before anyone tries to improve OCR in general.

### Training, and why the first result was wrong

Written up in `docs/training_analysis.md`. Four runs.

E1 trained on 18 frames and scored against the recovered annotations: F1 0.225 to 0.267,
an apparent 18% gain. E2 swept the threshold and showed that comparison sat at one point
suiting the fine-tuned model; best against best is +7%. E3 scored the same kind of run
against 1,647 hand-placed dots and the fine-tuned model lost, 0.360 to 0.369.

**Same data, same architecture, opposite conclusions.** E1 rewarded the model for learning
the pipeline's habits. Reporting it alone would have claimed a gain a reviewer could
disprove.

E4 exists to separate "the data is poor" from "there is not enough of it". It trains on
353 of the 413 and holds 60 out.

### Three DeepForest details that cost hours

`config.score_thresh` never reaches the model; `model.model.score_thresh` does.
`evaluate()` is deprecated in 2.0 and reports no mAP. `trainer.validate()` returns only
losses unless `config.validation.val_accuracy_interval` is set to 1, because the default
is 20 and a standalone validate runs at epoch 0.

---

## How to Resume

### Where things stand

`main` carries everything through the DeepForest CSV. PR #11 is merged. Deliverable #1
(the recovery approach) and #2 (the dataset) are done; #3 (a model) has four measured
runs; #4 (a DeepForest PR) was cancelled by the mentors; #5 (a blog post) is not started.

```
results/dataset/          25 benchmark frames        6,420 boxes
results/dataset_scaled/  413 frames drawn fresh    118,270 boxes
```

Both carry `annotations_deepforest.csv` (the six columns DeepForest reads),
`exported_frames.csv` (one row per shipped frame with what was measured on it), and for
the 25-frame set `frames.csv` (all 63 candidates with the reason each is in or out).
`annotations_full.csv` for the scaled run stays local at 54 MB.

### The pipeline, end to end

```
screenshot + clean original
   align.py       register the two                 96.7%, 0.38px
   subtract.py    annotations as image difference
   select.py      which frames to trust            43% of a fresh draw pass
   legend.py      find dialog, parse its rows      names 0.904, counts 0.693
   classify.py    dot -> legend row                0.781 per dot
   mapping.py     screenshot px -> original px     sub-pixel
   birdsize.py    how large a bird is, per frame   box 16-77px
   scripts/export_dataset.py
```

### Commands

```bash
# rebuild ground truth and the benchmark, only if data/cache is empty
python scripts/build_manifest.py
python scripts/build_benchmark.py --per-cell 3

# the real gate, per dot against the hand labels in data/labels/
python scripts/eval_localisation.py     # placement sub-pixel, classification 0.781

# the dataset
python scripts/export_dataset.py                    # 25 frames, writes results/dataset/
python scripts/export_dataset.py --box 80 --out X   # force one box size, for a sweep

# figures, all from the live path
python scripts/make_box_figure.py                          # all 25 frames, the boxes
python scripts/make_box_figure.py --frames 5745,0027,0406,426 --window 190 --cols 4        --out fig_box_closeup.png                           # a readable close-up
python scripts/make_mapping_figure.py                      # dots on the photographs
python scripts/make_full_overlays.py                       # every frame at full size

# more hand labels
python scripts/label_dots.py --only <frame>.jpg --blind 0
# open results/labelling/*.html, label, press S; the browser saves to Downloads,
# so copy the JSON into data/labels/ afterwards

# tests
pytest tests/ -q                        # 236 passing
```

### Training

Notebooks are local and gitignored: `notebook/e4_training_kaggle.ipynb` is the one that
runs, three cells, on Kaggle rather than Colab because Kaggle publishes its quota (40
GPU-hours a week, 9-hour sessions) and can run a notebook in the background with
**Save Version, Save and Run All**.

Results go in `docs/experiments.md`, which is local. The published summary is
`docs/training_analysis.md`.

Four runs so far. E1 trained on 18 frames and scored against the recovered annotations:
F1 0.225 to 0.267. E2 swept the threshold and showed that was +7%, not +18%, because both
models had been compared at one point that suited the fine-tuned one. E3 scored the same
kind of run against 1,647 hand-placed dots and the fine-tuned model lost, 0.360 to 0.369.
E4 trains on 353 of the 413 scaled frames, holding 60 out, and exists to separate "the
data is poor" from "there was not enough of it".

Three DeepForest details that each cost hours:

- `config.score_thresh` never reaches the model. `model.model.score_thresh` does.
- `evaluate()` is deprecated in 2.0 and reports no mAP. Use `trainer.validate()`.
- `trainer.validate()` returns only losses unless
  `config.validation.val_accuracy_interval` is 1. It defaults to 20 and a standalone
  validate runs at epoch 0, so the full pass runs and the metrics are skipped.

### What is local and stays local

Measurement written to answer a question for us stays on disk; only the result goes out,
by message. That covers `docs/PROJECT_STATE.md`, `docs/TASKS.md`, `docs/experiments.md`,
the working notebooks, `results/dataset_scaled/chunks/`, and `results/training/`.

### Ground rules, each learned the hard way

1. The pipeline works from the **image alone**: the screenshot and its paired clean
   original. Survey counts are **never** a pipeline input.
2. The survey data is for **validation only**. Feeding it in and scoring against it makes
   the score meaningless. This happened once already, the count-prior.
3. Ground truth for dot counts is **`category_sum`**, never `total_birds`.
4. Verify **spatially**, not by count. A matching count can be right for the wrong
   reasons: one frame carries a `category_sum` of 450 with no dots on the image at all.
5. Do not tune on the four old study images, and do not tune the `dot_candidates` band
   constants against the 63-frame benchmark. They are scored on it. Re-derive from
   `data/labels/`.
6. **Score detection with the hand labels, not with counts.** A count ratio cannot see
   whether a dot sits on a marker.
7. **Say what a number was scored against, above the number.** E1 and E3 reached opposite
   conclusions on the same data because one scored against our own annotations and the
   other against people.
8. **A cached metrics file is not a substitute for running the stage.** Reading detection
   counts from `results/eval_detection.csv` rather than running detection produced a
   confident, wrong conclusion about the funnel.
9. **Every shipped figure comes from a generator that runs the live path**, and its
   numbers must match the current `eval_localisation.py` run. Two stale figures have
   already misled a reviewer.
10. **No commits until the piece of work is complete**, then one detailed PR. No AI
    attribution anywhere in commits, PR bodies or docs.

## Reference Docs

> Entries marked *(local)* are working notes kept outside the repository. They are
> listed so the reasoning can be traced, not because the file ships here.

- **PR #3**: github.com/vickysharma-prog/Deepforest-bird-recovery-prototype/pull/3, legend module (code + tests + CI + figures); figures embedded in the description
- *(local)* `docs/legend_findings.md`: legend pipeline: findings, results, open gaps (mentor-facing)
- *(local)* `docs/legend_groundtruth.md`: hand-read legend tables (validation target) for 4 dialogs
- *(local)* `docs/scale_validation.md`: full pipeline run on 14 images (localization 14/14)
- *(local)* `docs/GSoC_meeting_2026-06-29.docx`: meeting document (problem, work, results, plan, code locations); **2026-07-01 section appended** (count-OCR fix, count-prior, fig6/fig7, tools/packages table, pipeline walkthrough, key numbers)
- `results/figures/`: 7 result figures + README explaining each
  - fig1–fig5: dialog localization, marker-class mapping, aerial classified, count barchart, synthetic color-vs-shape demo
  - **fig6_count_prior_improvement.png**: before/after attributable recall grouped bar (4 images, A +35 pp, C +49 pp)
  - **fig7_class_breakdown_4images.png**: per-class attributable counts 2×2 grid, all 4 images
- `docs/learnings.md`: 30 documented findings from building the prototype
- `docs/training_analysis.md`: 4 training experiments with root cause analysis
- *(local)* `docs/project_documentation.pdf`: summary sent to mentor (this report's source)
- `notebook/prototype_v1.ipynb`: full 23-section prototype (runs in Colab, T4 GPU, ~45 min)
- `config.yaml`: all tunable parameters (HSV bins, morphology thresholds, boundary
  constraints, plus the **new `align:` and `subtract:` sections**: each value there
  carries the measurement that justifies it)
- `results/eval_alignment.csv`: per-image registration outcome
- `results/eval_detection.csv`: per-image old vs new detection vs `category_sum`
- *(local)* `docs/matching_rework_FINAL_2026-07-10.docx`: visual report of the matching rework
- **GitHub discussion #6**: The Water Institute's notes (LAB, subtraction, UI masking)
