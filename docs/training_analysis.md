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

### The training run

349 frames train, 60 held out, three epochs, 2 hours 43 minutes on a T4. Tiling turned
118,270 boxes into 162,598 on 18,467 tiles, because a 0.15 overlap writes a box that
straddles a tile edge into both tiles. Four frames failed to tile and were dropped, all
from the same card.

```
train    140,704 boxes on 349 frames
test      21,894 boxes on  60 frames, 2,494 tiles
```

The 60 held-out frames are not 60 frames of the corpus. All 413 exported frames passed
`select.accept_frame`, so both the training and the test set are drawn from the 43% of
images the pipeline accepts, and both carry that same selection bias. This measures the
model on the frames the pipeline is willing to hand it, not on the archive.

Same test set before and after. Only the model changes.

```
                  pretrained    fine-tuned
mAP                 0.008         0.022
mAP@50              0.036         0.087

score >= 0.05   P   0.115         0.186
                R   0.356         0.414
score >= 0.10   P   0.194         0.270
                R   0.331         0.396
score >= 0.20   P   0.304         0.364
                R   0.274         0.307
score >= 0.30   P   0.356         0.416
                R   0.207         0.122

best F1             0.288         0.333      both at 0.20
```

**mAP@50 rises 2.4 times, from 0.036 to 0.087.** Quote this one first. mAP integrates
over the whole confidence curve instead of fixing a cutoff, so no choice of threshold can
flatter it, and it accumulates over the epoch, so the batch-size difference described
below cannot reach it. E1 moved nothing on mAP from 18 frames. **353 frames answer the
question E4 was asked: 18 was too few.**

The threshold table says the same thing from a second direction. **At three of the four
thresholds, precision and recall both rise.** That matters more than either number alone.
A model that only gained recall would be drawing more boxes and catching a few more birds
by volume. Gaining both means better boxes, not more of them.

```
F1        0.05    0.10    0.20    0.30
before    0.174   0.245   0.288   0.262
after     0.257   0.321   0.333   0.188
```

Best against best is 0.288 to 0.333, a 16% relative gain, with both models peaking at
0.20. The stronger statement is that **the fine-tuned model beats the pretrained model's
best score at two separate thresholds**, 0.321 at 0.10 and 0.333 at 0.20. Winning once
could be one cutoff happening to suit it. Winning at two is harder to explain that way,
which is the same argument E2 rested on.

**The gain is spread across the test set, not carried by a few frames.** Counting per
frame, the fine-tuned model finds more birds on 40 of the 60 held-out frames, fewer on 15,
and the same number on 5. An average that came from three spectacular frames and 57 flat
ones would mean something quite different.

**These are small numbers in absolute terms and a reviewer will say so first.** An mAP@50
of 0.087 is a weak detector, and `map_small` is 0.011. The birds measure 16 to 54 pixels
on the original photographs, which puts nearly all of them in the small-object regime
where mAP is punishing and where a few pixels of box error costs the whole match. The
gain is real and the base is low. Both belong in any sentence that quotes it.

### Recall still collapses above 0.30, at any dataset size

Recall at 0.30 falls from 0.207 to 0.122. E2 and E3 both found this on the 18-frame
model, and twenty times the data did not remove it. Training on annotations that contain
false positives lowers what the model will assert. It finds more birds and scores each one
lower, which moves its useful range down rather than breaking it. Anyone deploying this
checkpoint should run it near 0.10 to 0.20 and not at the default.

That reproduces across two dataset sizes and two test sets, so it is a property of
training on recovered labels, not an accident of one run.

### What it looks like on a photograph

Five held-out frames drawn whole, pretrained beside fine-tuned, both at score 0.10, with a
close-up on the densest part of each. These are the five largest gains of the 60, and the
figure says so rather than passing them off as a random sample.

```
frame                          labelled   pretrained    fine-tuned
21June15Camera1-Card3-01343      1,611    542 / 2,445   1,268 / 2,598
18June13Camera1-Card2-0296       1,014    200 / 1,103     373 /   828
28May12Camera2-Card1-0333          499     71 /   302     149 /   352
25May13Camera2-Card4-1442          493    218 /   948     316 /   893
23May13Camera2-Card1-0730          428     76 /   791     149 /   951
```

Read as birds found out of boxes drawn. The first frame is the largest move, recall 0.34 to
0.79 on 1,611 birds. The second is the more useful one to show: the fine-tuned model drew
**fewer** boxes, 828 against 1,103, and still found nearly twice as many birds. That is the
whole claim in one frame. It is not drawing more, it is drawing better.

```
results/figures/fig17_e4_before_after_01343.png    the largest gain
results/figures/fig18_e4_before_after_0296.png     fewer boxes, more birds
results/e4/                                        the numbers behind both
```

The counts here are lower than the per-frame totals in the tile scan because tiling
duplicates a bird that straddles a tile edge, and a whole photograph counts it once.

Two things in the figures are worth looking at rather than reading past.

**The false positives sit outside the colony.** On `01343` the unmatched boxes cluster in
the surrounding vegetation, not among the birds, in both models. Whatever is costing
precision is texture in the grass, not confusion between neighbouring birds.

**Some of the green boxes are not birds.** In the bottom-right and along the right edge of
`0296`, recovered annotations form neat rows that read as text. That is map ink the
detector picked up because the painted labels share the markers' palette colours, a known
limit of the recovery stage. It is in the training data, and the figure shows it.

### What this result does not say

Every number above is scored against the recovered annotations. The model trained on our
pipeline's output and was tested against our pipeline's output on different frames. It
measures how well the model reproduces the pipeline, which is exactly the circularity E3
was built to expose, and E3 is the run where fine-tuning lost.

Two things keep it from being empty. The test frames are held out, so the model is
reproducing the pipeline on photographs it never saw. And the pretrained bird model is a
real detector trained on far more data, so a 2.4 times gain over it is not trivial.

**One control was missed, then checked separately.** The pretrained model was validated at
batch size 1 and the fine-tuned one at batch size 4, because training set the batch size
and it carried into the validation passes afterwards. mAP accumulates over the epoch and
cannot be affected, but precision and recall were not controlled for it.

So both models were run again over all 2,494 held-out tiles one tile at a time, same code,
same threshold, matched greedily at IoU 0.4.

```
                 Cell C table      independent scan
pretrained   P      0.194               0.195
             R      0.331               0.270
fine-tuned   P      0.270               0.281
             R      0.396               0.363
```

Precision agrees to 0.001 on the pretrained model, which is the figure the batch-size
question was about. Recall reads lower for **both** models, by 0.061 and 0.033, so that gap
comes from a different matching rule and not from batch size. Both measurements put the
fine-tuned model ahead on precision and recall together. The scan makes the gain larger,
F1 0.226 to 0.317, so the 16% in the table is the conservative reading.

E3 already reported the other half of this. Against the hand labels, on the 18-frame
model, fine-tuning lost. The two runs answer different questions and both stand: E3 says
what the recovered data is worth against people, E4 says what more of it is worth against
itself.

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
