# Hand-labelled dots: the first per-dot measurement of detection

> **The numbers below are the state on 2026-08-07 and have been superseded.** They are
> kept because they are what motivated the chrome-mask fix, and because the diagnosis
> in them is what the fix was built from. Current state: the 2026-08-08 section of `docs/progress_report.md`.
>
> What changed the next day: tracing these 345 markers through `extract_annotations`
> showed 92 of them (26.7%) were being deleted by the chrome mask. Fixing that moved
> recall 0.345 → 0.678 and classification 0.373 → 0.544, and made the two-failure-mode
> reading below partly an artefact of that bug rather than a property of the detector.
> The label set has since grown to 541 dots on 8 frames.

**Date:** 2026-08-07
**Labels:** 345 dots across 6 screenshots, `data/labels/*.json`
**Harness:** `scripts/label_dots.py` → `scripts/labeller.html` → `scripts/eval_localisation.py`
**Result file:** `results/eval_localisation.csv`

---

## Why this was needed

The survey gives a **count** per image and never a coordinate. A count cannot tell
these three apart, and all three score 61:

- found all 61 markers
- found 40 real markers and invented 21
- found 61 markers, none of them on anything

So detection precision, recall and placement error had never been measured, and
neither had per-dot classification accuracy. The reported classification figure of
0.36 was per-class *count* agreement, which a wholesale label swap satisfies.

Everything below is the first measurement that can see those differences.

---

## Headline

| metric | value |
|---|---|
| **precision** | **0.138** |
| **recall** | **0.345** |
| **F1** | **0.197** |
| placement error, median | **1.83 px** |
| correct / false positive / missed | 119 / 745 / 226 |
| classification, per-dot | **0.373** (118 dots) |

**The count metric reads 1.24x median on this same pipeline.** That number is not
wrong, it is answering a different question. Per frame the two failure modes are
opposite and cancel inside a median.

---

## Per frame

| frame | band | seeded | labels | survey | detected | TP | FP | FN | P | R | F1 | err px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `08May10…0115` | sparse | blind | 36 | 36 | 262 | 20 | 242 | 16 | 0.076 | 0.556 | 0.134 | 2.15 |
| `10June10…0076` | sparse | seeded | 9 | 9 | 128 | 5 | 123 | 4 | 0.039 | 0.556 | 0.073 | 1.40 |
| `11June10…0154` | medium | seeded | 72 | 72 | 98 | 38 | 60 | 34 | 0.388 | 0.528 | **0.447** | 1.62 |
| `14June21…238` | medium | seeded | 71 | 71 | 37 | 11 | 26 | 60 | 0.297 | 0.155 | 0.204 | 2.04 |
| `18June21…06389` | medium | seeded | 103 | 119 | 22 | 22 | **0** | 81 | **1.000** | 0.214 | 0.352 | 0.04 |
| `25June10…0401` | medium | blind | 54 | 54 | 317 | 23 | 294 | 31 | 0.073 | 0.426 | 0.124 | 2.24 |

---

## Finding 1: two opposite failure modes, not one

| band | n | precision | recall |
|---|---|---|---|
| sparse | 2 | **0.06** | 0.56 |
| medium | 4 | 0.34 | 0.32 |

- **Sparse frames find about half the markers and bury them in noise.** `0076` is a
  mangrove colony: 9 real markers, 128 detections, 5 correct. This confirms
  spatially what the progress report only inferred — the sparse residual is
  vegetation texture, not primarily red label text.
- **Medium frames are cleaner but miss most markers.** `06389` produced 22
  detections, **every one of them correct**, and missed 81.

One fix cannot address both. This also explains why three attempts at the
marker-size estimator each improved one band and regressed the other.

## Finding 2: placement is already good

Median error **1.83 px**, consistent across all six frames (0.04–2.24). When the
detector finds a marker it lands on it.

This matters because `docs/learnings.md` #18 established that position accuracy
dominates training outcome — 920 SIFT-mapped annotations beat 3,851 with ~30px
error. **Detection's problem is finding markers, not placing them.**

## Finding 3: the label-swap failure is real and systematic

Per-dot classification is **0.373** on 118 matched dots. The errors split into two
distinct causes:

| count | true → predicted | cause |
|---|---|---|
| 25 | WHIB site → WHIB adult | same-colour class swap |
| 7 | SNEG site → SNEG bird | same-colour class swap |
| 5 | TRHE site → TRHE bird | same-colour class swap |
| 2 | TRHE site → WHIB site | same-colour class swap |
| 17 | row 2 (green) → none / BRPE brood | legend name OCR failed |
| 8 | UNSB ad, BRPE imm → none | unassigned |

`learnings.md` #30 predicted the WHIB site/adult swap from a single anecdote. It is
now measured at **25 instances**, and the same pattern appears on two further
species. It is a systematic `site` → `bird`/`adult` collapse within a colour, not a
one-off.

The second cause is different and probably cheaper to fix: when the legend row's
name does not read, every dot of that class is unattributable regardless of how
well the matcher works.

## Finding 4: the labels are trustworthy

**Five of six frames matched the survey count exactly** — 9/9, 36/36, 72/72, 54/54,
71/71. The sixth, `06389`, gave 103 against a survey figure of 119.

This held on blind frames as well as seeded ones, which is the point of the control:
if seeing the detector's output had biased the labelling, blind frames would have
come out short. They did not.

`06389`'s 16-dot gap is either fused markers or another case of `category_sum` not
counting drawn dots, as already proven for `00097` (survey 450, zero dots on the
image).

---

## What was also done, and deliberately not shipped

`subtract._marker_area` was diagnosed and three replacements were measured. The
diagnosis is solid: the distance transform returns the **stroke half-width** on
outline glyphs (rings, plus signs, asterisks), not the marker footprint. It floored
to an area of 4.0 on 11 of 33 benchmark frames, and those frames carried median
|log2 ratio| 1.26 against 0.23 elsewhere.

| | baseline | modal blob | hybrid |
|---|---|---|---|
| median \|log2\| | 0.47 | 0.44 | 0.53 |
| within ±25% | 19 | 22 | 18 |
| within ±10% | 5 | **15** | 11 |
| miss (<0.5x) | **5** | 9 | 8 |
| dense band | **1.14x** | 0.83x | 0.85x |

None passed the gate cleanly. Parked on branch `exp/marker-size-estimator`
(`85de2d8`); **`main` is unchanged**. The hand labels then confirmed the diagnosis
from the other side: `06389`, whose accept band excluded real markers, scores
precision 1.00 — the detector was too strict, not producing garbage.

**The band constants 0.35 / 1.5 / 3.0 were left untouched on purpose.** They were
calibrated against an estimate roughly 4x too small, so their effective lower bound
was near 0.09 of a real marker. Re-deriving them by tuning against the 63-frame
benchmark they are scored on is the mistake this project already made twice — the
count prior and legend self-recovery. They should be re-derived against these
labels instead.

---

## What is needed next

1. **Fix the two bands separately.** Sparse needs precision (vegetation rejection),
   medium needs recall (the size band excluding real markers). Verify each against
   `eval_localisation.py`, not the count metric.
2. **Re-derive the band constants** using the labelled frames, now that "how small
   can a real marker's ink be" is answerable.
3. **Legend name OCR** is worth attention before matcher work: 17 of 118
   classification errors are dots whose class name never read, which no amount of
   shape matching can recover.
4. **The `site` vs `bird` split** is the classification problem, now quantified at
   39 instances across three species.
5. **More labels for the dense band.** Four frames remain unlabelled: `0296` (412),
   `0507` (673), `00097` (450, expected ~0 real dots), `0081` (234). Dense is
   currently unmeasured per-dot.

---

## Reproducing this

```bash
python scripts/label_dots.py --frames 10 --blind 2   # build the pages
# open results/labelling/*.html, label, press S, save into data/labels/
python scripts/eval_localisation.py                  # score it
python scripts/eval_localisation.py --tol 4          # tighter match radius
```

Design choices in the tool that the numbers depend on:

- **Seeds come from the subtraction path only.** The first version used the union of
  both detectors and produced 1571 seeds against 9 real markers on `0076`, which
  costs more to delete than to draw.
- **The pipeline's class guess is stored but never displayed.** Showing a confident
  wrong answer to the labeller would hide exactly the swap this set was built to
  find.
- **Two frames are labelled blind**, as the control described in Finding 4.
- **A sweep grid credits a tile only at 2x zoom or closer**, since a 4px marker is
  not visible below that. `06389` was first labelled at 1x and yielded 100 dots;
  re-swept at 3x it yielded 103.
