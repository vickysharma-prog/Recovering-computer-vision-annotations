# Training the Model: Is the Recovered Data Actually Worth It?

*Deep in the Wild, part 7. Last of six technical posts on recovering machine-readable bird annotations from eleven years of aerial survey screenshots.*

**Yes. A big yes, and I am very happy to share it.**

Here is a photograph neither model had ever seen, holding 1,611 birds. Pretrained DeepForest, one of the best open aerial bird detectors available, finds 542 of them. The same model, after training on annotations this pipeline recovered out of an old screenshot, finds **1,268**.

> **[IMAGE 1 HERE: `blog/images/post6-img1-training-result.png`]**

*A held-out photograph. Neither model trained on it. Both running at the same confidence threshold.*

Across the full held-out set, mAP@50 more than doubled, from 0.036 to **0.087**, and the model improved on **40 of the 60** photographs. Annotations that existed as nothing more than coloured pixels painted into a screenshot taught a real detector to find real birds.

That is the answer. The rest of this post is how it got there.

## First, the dots had to move house

Five posts in, the pipeline could find every dot on a screenshot and give each one a species. But those coordinates were positions on a **screenshot**, which is the photograph shrunk into a window with a dialog sitting on top of it. Nobody trains a model on that. The annotations had to land on the original full-resolution photograph.

## Putting the dots back on the photograph

The transform already existed. Registration, from post three, computes exactly how the screenshot maps onto the original, at 0.38 pixels.

It also hides a trap worth passing on. That transform does not land on the original photograph. It lands on a **downscaled** copy, because the original is shrunk before feature matching runs, so a real coordinate needs one more division by the scale factor.

Leave it out and nothing breaks. No exception, no warning. The coordinates stay internally consistent, look plausible, sit in sensible ranges, and put every dot in the wrong part of the photograph by a constant factor. The only way to catch it is to draw the dots and look.

## How big is a bird?

DeepForest wants boxes, not points, and the survey recorded a click per bird and never an extent. So the box size has to be measured from the imagery, and it cannot be one fixed number: eleven years of surveys flew focal lengths from 28mm to 300mm, so the same species covers 10 pixels on one photograph and 41 on another.

Two attempts failed first, both instructively. **Spacing between dots measures crowding, not size**, running from 13.6 pixels on a packed colony to 215 on a sparse one while the birds vary about twofold. And **an area-equivalent diameter is not a length**: for a bird twice as long as it is wide it comes to 0.71 of the length, so every box was built from 70% of the bird and cut it in half.

What works is measuring each frame's own birds. Around each recovered dot, cut a patch from the clean photograph, compare its lightness against that patch's own median so a pale bird on grass and a dark bird on sand both register, and take the long side of the resulting shape.

> **[IMAGE 2 HERE: `blog/images/post6-img2-boxes-on-birds.png`]**

*Four exported frames, each with a box size measured from its own birds: 10px birds get 16px boxes, 41px birds get 54px boxes.*

Then the check that matters. Dividing known species body lengths by the measured size gives **1.3 to 4.0 cm per pixel**, which is what these surveys actually fly, and the camera EXIF predicts exactly that tenfold spread from its focal lengths and pixel pitches. A number that only agrees with itself is not evidence.

## The dataset

The first export covered the 25 benchmark frames and produced 6,420 boxes. Useful, but every decision in the pipeline had been checked against those same 25 frames, so numbers measured there are optimistic by construction.

So I scaled it. 1,197 candidates drawn stratified across seven survey years and three density bands, minus the frames carrying hand labels, put through the identical pipeline with nothing tuned.

```
1,076 pairs downloaded
  458 pass frame selection      43%
  413 exported
118,270 boxes
```

**The quality figures did not move at eighteen times the size.** Species resolution read 0.648 on 25 frames and 0.650 on 413. Box size was measurable on 96% of frames before and 98% after. That is worth stating on its own, because it means the earlier numbers were not an artefact of the sample they came from.

## The training run

The setup is deliberately boring, because the point is to change one thing and see what happens. The same DeepForest bird model, fine-tuned on the recovered annotations. Tiles of 400 pixels. One class, "Bird". **Split by frame, never by dot**, because two dots from the same photograph on either side of the split is not a held-out test: the model has already seen that background, that colony, that light.

349 frames to train on, 60 held out, three epochs, two hours and 43 minutes on a single T4.

Same test set before and after. Only the model changes.

```
                  pretrained    fine-tuned
mAP@50              0.036         0.087       2.4x
best F1             0.288         0.333       both at 0.20
per frame           40 of 60 improved, 15 worse, 5 level
```

**mAP@50 rose 2.4 times.** That is the number to quote, because it integrates over the whole confidence curve instead of fixing a cutoff, so no choice of threshold can flatter it.

Two things make it more than a single number.

**At three of the four thresholds, precision and recall both rise.** That distinction matters. A model that only gained recall would just be drawing more boxes and catching a few more birds by volume. Gaining both means better boxes, not more of them.

**The gain is spread across the test set.** 40 of 60 held-out photographs improved, 15 got worse, 5 stayed level. An identical average built from three spectacular frames and 57 flat ones would mean something completely different, and no pooled number tells you which you have.

The clearest single frame is not even the one at the top of this post. On another held-out photograph the fine-tuned model drew **fewer** boxes than pretrained, 828 against 1,103, and found nearly twice as many birds, 373 against 200. It is not drawing more. It is drawing better.

## Reading the result fairly

Two things belong beside that number, and I would rather say them than have a reviewer say them first.

**An mAP@50 of 0.087 is a low base.** These birds measure 16 to 54 pixels on photographs over 5,000 pixels wide, which puts nearly all of them in the small-object regime where mAP is punishing and a few pixels of box error costs the whole match. The gain is real and the starting point is low, and both belong in any sentence that quotes it.

**The scoring is against the recovered annotations**, on photographs the model never trained on. So it shows that training on this data teaches a model to find these birds on unseen ground. Two things keep that meaningful: the test frames are genuinely held out, and the pretrained bird model is a serious detector trained on far more data, so a 2.4 times gain over it is not a small thing.

## What we ended up with

Start of summer: 18,304 screenshots, 2.8 million bird observations visible to a human and invisible to a computer, and no coordinates anywhere.

Today:

- **A pipeline** that reads any screenshot in the archive on its own, finds the annotation dots, and names each one, using nothing but that image and its clean original.
- **Detections that land within 0.65 pixels** of the marker, verified against 1,648 dots placed by hand.
- **A check that runs without any labels** and tells you which photographs to trust, covering roughly **8,500 images and 1.49 million bird positions**.
- **A published dataset** of 118,270 training boxes across 413 photographs.
- **A trained detector** that beats pretrained DeepForest on held-out data.

Eleven years of careful fieldwork that existed only as coloured pixels is now data anyone can train on.

And the dataset is not really the point. The method is. Ecology, medicine and astronomy are full of painstaking human work sealed inside screenshots and PDFs by tools built to show a person an answer rather than hand a computer one. Nothing in this pipeline is specific to birds. Anyone sitting on an archive like that can do this.

## Thank you

To my mentors at Weecology, **Dr. Ben Weinstein, Henry Senyondo, Dr. Ethan White and Dr. Josh Veitch-Michaelis**, who consistently asked the harder question instead of the easier one. Half the good decisions in these six posts started as something one of them pushed back on.

To **The Water Institute**, both for the archive itself and for engaging with this work so openly in public.

And to **Google Summer of Code** for the time to do this properly rather than quickly.

## Stay tuned

These six posts are the overview. **A detailed research publication on this project is on the way**, carrying the full method, the measurement design, the ablations and the results at the depth they deserve.

Thank you for reading the series. It has been the best thing I have built.

---

*Code, data and measurements: [github.com/weecology/recovering-computer-vision-annotations](https://github.com/weecology/recovering-computer-vision-annotations).*
