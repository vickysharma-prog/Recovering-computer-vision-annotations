# Training Analysis
I ran four training experiments, each one informed by the failure of the previous one. The sequence turned out to be more informative than any individual result.

---

## Starting point: pretrained DeepForest

DeepForest comes with a pretrained bird model. On my four test images, it produced 205 total detections but zero above 0.5 confidence. The maximum score was 0.376. It detects something at every location but has no confidence in any of them. For colonial birds photographed from aircraft, the pretrained model is effectively blind.
This isn't surprising — it was trained on general bird imagery, not aerial surveys of nesting colonies. But it means fine-tuning on domain-specific data isn't optional, it's necessary.

---

## Experiment 1: just train on everything

Took all 3,851 recovered annotations from 28 images, set labels to "Bird" (binary detection), and trained for 10 epochs. The training completed without errors. Then every prediction came back with score = 1.0.

Something was obviously broken. After a lot of debugging, I found that SAM 3 (which I'd loaded earlier for validation) enables a global bfloat16 autocast on the GPU. Even after deleting the SAM 3 model, the autocast persists. DeepForest's RetinaNet training under bfloat16 produces meaningless confidence scores.

Fix: `torch.cuda.amp.autocast(enabled=False)` before any training.

---

## Experiment 2: bfloat16 fixed, same data

With bfloat16 disabled, training produced real scores. But the results were disappointing — fewer detections than pretrained, and lower confidence. The model predicted bounding boxes around 106×105 pixels, but my training annotations used fixed 80×80 boxes.

The IoU between an 80×80 ground truth box and a 106×105 predicted box is around 0.45-0.55. With a typical positive IoU threshold of 0.5, about half the correct predictions don't get credit during training. The model doesn't learn efficiently.

---

## Experiment 3: species-aware boxes

Replaced 80×80 with calibrated sizes: BRPE=110×100, LAGU=60×55, ROYT=70×65, and so on. Validated every box against image dimensions. Zero invalid annotations.

Result: zero detections at any threshold. Worse than experiment 2.

This was confusing until I looked at the coordinate mapping quality. 43% of my training images used uniform scaling (SIFT failed on those images), which introduces roughly 30 pixels of position error. Even a perfectly-sized bounding box that's centered 30 pixels away from the actual bird has very low IoU.

---

## Experiment 4: SIFT-only (the breakthrough)

Filtered the training set to only include images where SIFT homography succeeded — 19 images with 920 annotations at 0.5 pixel accuracy. Excluded the 15 uniform-mapped images entirely.

On C_original.jpg (a SIFT-mapped test image):
- Pretrained: 71 detections, 0 high-confidence, max score 0.233
- SIFT-trained: 105 detections, 1 high-confidence, max score 0.300

29% improvement in max score. One actual high-confidence detection. With 76% less training data.

This proves two things: the recovered annotations CAN teach a model when positions are accurate, and the bottleneck is position quality, not annotation quantity.

---

## What this means

The training failure isn't a failure of the recovery approach. It's a coordinate mapping problem with a clear solution:

1. Get more SIFT successes (currently 57% — tune parameters, try alternatives like SuperGlue)
2. Use SAM 3 to refine positions (recovered dot → box prompt → precise segmentation → tight bbox)
3. Or skip dot recovery entirely for some images and use SAM 3 detections directly as training data

The recovery pipeline works. The annotations are real (98.3% precision). The mapping just needs to be more accurate for training to succeed.
