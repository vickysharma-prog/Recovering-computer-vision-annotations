# Training analysis

The pipeline recovers bird positions from screenshots where a counting tool baked
coloured dots into the pixels. Training answers the question a CSV cannot: **is the
recovered data good enough to teach a detector?**

A dataset can look correct and be useless. If the coordinates are systematically off, or
the boxes the wrong size, or the labels noisy, a model trained on it learns nothing. No
amount of inspecting the file shows that. Training does.

Every run below reports what it was scored **against**, above the numbers, because that
line decides what they mean.

---

## The setup, held constant

```
model        weecology/deepforest-bird, fine-tuned
tiles        400px, 0.15 overlap
split        by frame, never by dot
classes      one, "Bird"
batch 4      lr 1e-4
```

Split by frame matters. Two dots from the same photograph on either side of the split is
not a held-out test: the model has already seen that background, that colony, that light.

One class, not the 380 the dataset carries. Only three species appear on more than four
frames, and a species confined to one frame can be neither learned nor tested under a
frame-wise split. The species labels stay in `annotations_full.csv` and stay recoverable
through `frame` and `legend_row`.

---

## E1: trained on 18 frames, scored against our own annotations

18 frames train, 7 held out, 15 epochs.

```
held-out 7 frames    pretrained  ->  fine-tuned
  precision            0.205     ->    0.220
  recall               0.250     ->    0.338
  F1                   0.225     ->    0.267
```

Recall gained 35% relative and the gain was spread across six of the seven frames and all
three density bands, rather than sitting on the few frames where the pipeline is least
precise. That looked like a clear result.

It was scored against the recovered annotations themselves. The model was trained on them
and tested on them, so the figure says how far it agrees with our pipeline, not how well
it finds birds.

---

## E2: the same checkpoint, six thresholds

No training. The E1 model re-scored at six confidence cutoffs.

```
thresh   fine-tuned P/R      pretrained P/R      ft F1   pre F1
0.05     0.157  0.354        0.141  0.273        0.217   0.186
0.10     0.220  0.338        0.205  0.250        0.267   0.225
0.20     0.316  0.232        0.304  0.213        0.267   0.250
0.30     0.336  0.093        0.336  0.161        0.146   0.218
0.40     0.451  0.012        0.355  0.080        0.022   0.131
0.50     0.500  0.001        0.336  0.019        0.002   0.035
```

Three things came out of it.

**The fine-tuned model wins at three thresholds, not one.** At 0.05, 0.10 and 0.20 it
leads on precision and recall together. Winning once could be the threshold suiting it;
winning three times is harder to explain away.

**E1's +18% was measured at one point that suited it.** Both models were compared at 0.1,
which sits near the fine-tuned optimum and away from the pretrained one's. Each at its own
best threshold gives 0.267 against 0.250, so **+7%**. Both figures are real and answer
different questions. The conservative one leads.

**Above 0.3 the fine-tuned model is worse**, recall 0.093 against 0.161. Fine-tuning
lowered its confidence: it finds more birds and scores every one of them lower. Training
on labels that contain false positives does this, and it shows as a shifted operating
range rather than as an obvious mistake.

---

## E3: the same experiment, scored against people

The objection to E1 is that it rewards the model for learning the pipeline's habits. The
answer is ground truth the pipeline did not produce: **1,647 dots placed by hand across
12 frames**, mapped onto the photographs and tiled the same way.

Train on the 18 frames carrying no hand labels. Test on the 12 that do.

```
thresh   pretrained P/R      fine-tuned P/R      pre F1   ft F1
0.05     0.103  0.445        0.107  0.544        0.168    0.178
0.10     0.200  0.427        0.205  0.499        0.273    0.290
0.20     0.362  0.377        0.389  0.335        0.369    0.360
0.30     0.403  0.273        0.476  0.096        0.326    0.159

best F1   pretrained 0.369        fine-tuned 0.360
```

**Against independent labels, fine-tuning did not improve the model overall.** Best
against best, the pretrained model is marginally ahead.

At low thresholds the fine-tuned model does lead: recall 0.499 against 0.427 at 0.10, 17%
relative. Its useful range moved down, as E2 found. At 0.30 its recall collapses.

```
E1   against our own annotations    fine-tuned won    0.267 vs 0.250
E3   against hand labels            fine-tuned lost   0.360 vs 0.369
```

Same data, same architecture, opposite conclusions. **E1 alone would have claimed an 18%
gain that a reviewer could have disproved.**

Two caveats, both of which favour us. Ten of the twelve label sets were seeded from the
pipeline's own output for a person to confirm, delete or add. They deleted 107 of 116 on
one frame and added 81 on another, so the review was real, but the labels still lean
toward the pipeline. And labelling is not exhaustive, so a real bird the model finds where
no human label sits counts against precision, equally for both models. The test was
generous and fine-tuning still did not win.

E3 says nothing about classification. It trains a single-class model; the species labels
are unused. Classification is measured separately at 0.781 per dot.

---

## E4: was 18 frames simply too few?

That is the open question. A model pretrained on far more data is hard to move with 18
frames, and E3 cannot separate "the data is poor" from "there is not enough of it".

So the dataset was scaled. 1,197 candidates drawn stratified across seven survey years and
three density bands, minus the 12 frames carrying hand labels, put through the same
pipeline untouched.

```
1,076 pairs downloaded
  458 pass frame selection      43%
  413 exported
118,270 boxes
```

**The quality figures did not move at eighteen times the size.**

```
                    25 frames      413 frames
species resolved      0.648          0.650
box measured on        96%            98% of frames
distinct species        19             45
```

That is worth stating on its own, before any training result. Every earlier number came
from 25 frames, and those 25 are the frames every decision was checked against. They were
not an artefact of the sample they were measured on.

Training on 353 of the 413, holding 60 out, is running. Results will be added here.

---

## Three DeepForest details that cost hours

**`config.score_thresh` never reaches the model.** Setting it leaves
`model.model.score_thresh` untouched, so a run configured for 0.3 silently evaluates at
0.1. Set the attribute on the model.

**`evaluate()` is deprecated in 2.0 and reports no mAP.** Use `trainer.validate()`. Its
signature is `evaluate(csv_file, iou_threshold, root_dir)`, so passing the tile directory
positionally lands it in `iou_threshold` and fails quietly.

**`trainer.validate()` returns only losses unless asked for the metrics.** mAP, precision
and recall are computed only when `(current_epoch + 1) % val_accuracy_interval == 0`. The
default interval is 20 and a standalone validate runs at epoch 0, so the check never
passes: the full pass runs, eight minutes go by, and only the losses come back. Set
`config.validation.val_accuracy_interval = 1`.

And one that did not cost hours because it was checked before acting: small boxes train
normally. `Model.create_anchor_generator` offers sizes down to 8px and looks like the fix
for 16px boxes, but its own defaults raise an assertion and `create_model` never calls it.
torchvision's RetinaNet matches with `allow_low_quality_matches=True`, so every
ground-truth box is assigned its best anchor whatever the IoU. Nothing is dropped.
