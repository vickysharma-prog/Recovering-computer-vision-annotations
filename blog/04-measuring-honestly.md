# Measuring Honestly: Hand-Labelling 1,648 Dots to Test the Detector

*Deep in the Wild, part 5. Fourth of six technical posts on recovering machine-readable bird annotations from eleven years of aerial survey screenshots.*

The last post ended with detection looking healthy. Over-detection had come down from eight times too many dots to about a quarter too many, and every band of the benchmark had improved.

Then I measured it properly, and precision came back at **0.138**.

This post is about that gap, why it existed for months without anyone noticing, and the idea that eventually made detection usable despite it.

## What a count cannot see

Every number in the previous post was a ratio: dots detected over dots the survey recorded. It is the only score available, because the survey gives a count per image and never once gives a coordinate.

Here is the problem. Take an image holding 61 birds. All three of these outcomes score exactly 61:

- found all 61 markers
- found 40 real markers and invented 21
- found 61 things, none of which is a marker

The ratio cannot separate them. It never could. So precision, recall, how far a detection sits from the marker it claims, and whether a dot got the right species were all simply unmeasured. Not measured badly. Not measured at all.

The only way out was to produce coordinates the survey never had.

## Building a labelling tool that can disagree with me

So I built one. `label_dots.py` generates a self-contained web page per screenshot where I click on every bird I can see, and four decisions in it are what make the resulting labels worth anything.

**The page is seeded with the detector's own output**, because deleting wrong dots is far faster than drawing every dot from scratch on a colony of 400.

**Two frames are labelled blind**, with no seeds at all. This is the control. If seeing the detector's guesses had biased me into simply agreeing with it, the blind frames would have come out systematically different.

**The pipeline's class guess is stored in the file but never shown on screen.** Showing a confident wrong answer to the person labelling would hide precisely the error the labels exist to find.

**A sweep grid marks a tile as done only at 2× zoom or closer.** A four-pixel marker is invisible below that. One frame labelled at 1× gave 100 dots; re-swept at 3× it gave 103.

Did it work? Five of the first six frames matched the survey's independent count **exactly**: 9 of 9, 36 of 36, 72 of 72, 54 of 54, 71 of 71. That held on the blind frames as well as the seeded ones, which is the check that mattered.

The set now stands at **1,648 hand-placed dots across 12 screenshots**.

## The number that hurt

Scored against those labels, the first run read:

| | |
|---|---|
| precision | **0.138** |
| recall | 0.345 |
| placement error, median | 1.83 px |

The count metric on the very same pipeline, the same day, read **1.24×**. Close to perfect.

Both numbers are correct. They answer different questions, and per frame the two failure modes are opposite and cancel each other inside a median. Sparse frames found about half the markers and buried them in hundreds of false ones. Medium frames were clean but missed most of the markers. Average those and you get something that looks fine.

One finding was genuinely good news. **Placement error was 1.83 pixels**, and consistent across every frame. When the detector found a marker, it landed on it. The problem was finding markers, not placing them, which matters because position accuracy is what training quality turns on.

## The bug a count could never have found

Then the labels showed me something no count in the benchmark had moved on for months.

Part of the pipeline masks out window furniture, the dialog and title bars, so their pixels are not counted as annotation. It decided whether a region was furniture by measuring the average colour saturation over a smoothed version of that region. On a dense colony, smoothing bridges hundreds of scattered markers into one connected region covering a quarter of the frame, and the average then measures the *background between* the markers rather than the markers.

It was deleting **92 of 345 real markers**, 27% of them, and **53 of 71** on one frame.

Not a single count anywhere in the 63-image benchmark had shifted enough to notice. The frames were over-detecting from other causes at the same time, so deleting a quarter of the real markers moved the ratio toward 1.0 and looked like an improvement.

Measuring the region's actual ink instead of its average separates the two cleanly: marker regions run 11% to 95% saturated ink, dialogs 0% to 2.5%, with nothing in between. Fixing it moved recall from 0.345 to 0.678 immediately.

Where detection stands now, per frame, on the labelled frames the pipeline accepts: precision **0.30 to 1.00**, recall **0.40 to 1.00**, and median placement error **0.65 pixels**.

## The idea that made it usable

Precision still varies enormously between images, and I spent weeks trying to fix that. Five different filters, each meant to separate real markers from background: saturation, blob area, elongation, how well a blob matches the legend glyph, and a learned model combining all four. Every one failed, and they failed in an informative way. The discriminating direction *reverses between frames*. False positives are more saturated than real markers, not less. The learned model bought 0.012 F1 while recall collapsed from 0.597 to 0.177.

So I stopped trying to make every image work, and asked a different question: **which images are worth using at all?**

That turns out to be answerable with arithmetic rather than machine learning. Correct detections can never outnumber the dots actually on the image, and can never outnumber the dots detected. So precision is capped at `dots present / dots reported`. A frame reporting seven times what it holds cannot exceed 14% precision, no matter what is done downstream.

The survey's reported count is known without any labels. So the ceiling is known without any labels too, on all 18,304 images.

> **[IMAGE 1 HERE: `blog/images/post4-img1-frame-selection.jpg`]**

*Left: every frame sits under the ceiling its reported count implies. Right: recall holds steady everywhere; precision is what splits.*

The band has to be two-sided. Under-detecting disqualifies a frame just as thoroughly, and a frame returning a single detection for 19 dots sailed through a one-sided cut while being useless.

Verified against the hand labels, the split is clean. Frames whose reported-to-present ratio sits near 1.00, 0.99 and 0.62 measure precision **1.00, 0.82 and 0.74**. Frames at 5.63×, 7.35× and 14.22× measure **0.14, 0.07 and 0.04**.

Applied to the full archive: about **48% of images pass, and they hold roughly 72% of all the dots**, close to two million annotations. Excluding the rest costs little, because the frames that fail are mostly sparse scenes, and sparse images hold only 6% of the corpus.

Knowing which images to trust turned out to beat making every image work.

## What this stage actually delivered

Detection is finished, and it is honestly measured, which took longer than building it.

The result worth stating plainly: around **8,500 photographs and 1.49 million bird positions** in this archive are now recoverable with detections that land on the marker to within a pixel, and the pipeline can identify which photographs those are without a single human label. Eleven years of fieldwork that existed only as pictures of dots is, from this point, addressable data.

Every dot the pipeline keeps has a position that can be checked. None of them has a species yet.

That is the next post.

---

*Code and measurements: [github.com/weecology/recovering-computer-vision-annotations](https://github.com/weecology/recovering-computer-vision-annotations). Built during Google Summer of Code 2026 with the DeepForest project at Weecology Lab, University of Florida. Mentors: Dr. Ben Weinstein, Henry Senyondo, Dr. Ethan White and Dr. Josh Veitch-Michaelis.*
