# What I Learned Building This Prototype

Things I discovered, got wrong, or figured out the hard way. Each one changed how the pipeline works.

Items 1 to 19 come from the first prototype, which detected dots by colour and took per-species counts from the CSV. Items 20 onward come from the current pipeline, which works from the images alone. Where the two disagree, the later item is the one that holds: several early conclusions turned out to be artifacts of a small test set or a downscaled fixture, and those are marked.

---

## Getting the data right

**1. total_birds is per-row, not per-image.** The CSV has one row per species per image. An image with 538 birds and 7 species has 7 rows. I was picking study images by looking at individual row values instead of summing. That gave me completely wrong images the first time. Fixed by grouping by screenshot path and summing.

**2. DottingAreaNumber is just a counter.** I initially thought multiple screenshots might cover different sub-areas of one original image, which would mean I'd need to stitch them together. Turns out it's just a sequential ID (0, 1, 2...). Each original has exactly one screenshot. Saved me from building an unnecessary merging module.

**3. LAGU dominates, not BRPE.** Laughing Gull is 32.4% of all birds. Brown Pelican gets more press but is only 14.7% in this dataset. Changed which species I optimized for.

**4. Most images are easy.** Median is 61 birds per image. 65% have fewer than 100. Only 2% have over 1000. My four study images (96-1388 birds, 7-13 species) were deliberately hard. They are not representative of what the pipeline typically faces.

---

## Things I tried that didn't work

**5. OCR on the dialog box.** Spent a day on this. Tesseract with upscaling and adaptive thresholding, fuzzy matching species codes with Levenshtein distance ≤ 2. Problem: "THE" matches "TRHE" (Tricolored Heron), "AND" matches "CANG" (Canada Goose), any 3-4 letter word matches something. 4% precision on the best image, 4% on the worst. The CSV has the same information with 100% accuracy, so I stopped. **This conclusion was wrong, see #24.** I had been testing on a fixture that is 2.3× smaller than the real screenshots.

**6. Color analysis from the dialog.** Tried extracting colored regions from the dialog box and matching them to species by vertical position. Assumed colors would appear in the same order as species in the CSV. They don't. 1 out of 12 mappings was correct (8%). Switched to count-based matching instead: match the largest detected color group to the most common species, second-largest to second-most-common, and so on. Not perfect but functional. **Superseded by #23.** The order assumption was the broken part, not the idea of reading the dialog. Parsing each row in place, rather than matching by position, works.

**7. Text watermark filter.** Some screenshots have colony names printed in color on the aerial photograph. I built a filter to detect and remove horizontally-aligned, regularly-spaced colored dots (text characters). Problem: birds in colonies also sit in rows at similar y-coordinates. The filter removed actual bird annotations. Disabled it entirely. The aspect ratio filter (skip anything wider than 3:1) catches actual text lines without touching bird colonies.

**8. Narrow color bins.** Started with tight HSV ranges (mean ± 5 hue units per color). Got 44% detection accuracy. Widened to cover entire color regions (e.g., green = H 36-82 instead of H 55-65). Jumped to 58%. Annotation dots have more color variation than I expected: fading, JPEG compression, background influence.

**9. Adaptive threshold tuning.** Built a system that starts narrow and widens if detected count is too low compared to CSV expected count. Same result as just using wide bins from the start. The iteration added complexity without benefit.

---

## Technical things that matter

**10. Red hue wraps around.** In OpenCV's HSV, hue goes 0-180. Red sits at both ends: 0-10 and 170-180. Taking the arithmetic mean of [5, 175] gives 90 (cyan, which is completely wrong). Had to implement circular mean using sin/cos components. Shows up as a 5-line function but took a while to debug.

**11. Count-based color-to-species matching.** The first detector grouped dots by colour and matched those groups to CSV species by count similarity — biggest group to most common species. It got colour names wrong sometimes but usually landed the species, because the matching went on counts rather than names. **This is retired and nothing uses it.** `detect.py` is imported by its own test and nothing else. It also breaks the rule the project settled on later: the pipeline reads what it needs from the dialog in the image, and never takes survey counts as an input. Kept here because the reasoning was sound given what was known at the time.

**12. Study image accuracy ≠ pipeline accuracy.** The four study images average 52.3% accuracy. The batch of 30 random images averages 70.8%. I almost reported 52.3% as the pipeline's performance. That would have been misleading. The study images were picked to be hard.

**13. Uniform scale_x is wrong for coordinate mapping.** When SIFT fails, I fall back to uniform scaling. Initially used scale_x = original_width / aerial_width. But the aerial region in the screenshot shows a subregion of the original and doesn't span the full width. Scale_x gave 23-39% horizontal stretch. The fix: use scale_y (original_height / aerial_height) for both axes. Heights match because the aerial view isn't cropped vertically.

**14. The pipeline works across all years and annotators.** I was worried that the annotation software might have changed between 2010 and 2021, or that different annotators might use different dot styles. Tested on 7 years and 10 annotators. No systematic failures. The format is consistent.

---

## Training was harder than expected

**15. bfloat16 from SAM 3 breaks DeepForest.** SAM 3 enables a global bfloat16 autocast. Even after deleting the SAM 3 model and freeing GPU memory, the autocast stays active. DeepForest training under bfloat16 produces all scores = 1.0 (garbage). Took a while to figure out, because the training "succeeds" with no errors and the model is useless. Fix: explicitly disable autocast before training.

**16. 80×80 boxes cause IoU problems.** My training annotations use fixed 80×80 pixel bounding boxes. The model learns to predict boxes closer to actual bird size (~106×105). During training, the IoU between an 80×80 ground truth box and a 106×105 predicted box is low enough that the model doesn't get credit for correct detections.

**17. Better boxes don't help if positions are wrong.** After discovering the box size issue, I switched to species-aware sizes (BRPE=110×100, LAGU=60×55). Results got worse, not better. The real problem: 43% of my training data uses uniform coordinate mapping with ~30px position error. A perfectly-sized box centered 30 pixels away from the actual bird still has low IoU.

**18. Position accuracy is everything.** Training only on SIFT-mapped images (0.5px accuracy, 920 annotations) improved the model by 29% over pretrained. Training on the full dataset (3,851 annotations including ~30px-error data) made the model worse than pretrained. Less data with accurate positions beats more data with inaccurate positions.

**19. The model isn't working yet, but the direction is clear.** SIFT-only training produced one high-confidence detection and a 29% improvement in max score. That's not a working detector. But it proves the recovered annotations can teach a model. The limitation is position accuracy, not the recovery approach itself. SAM 3 box refinement or more SIFT-mapped images should close the gap.

---

## Getting the measurement right

**20. I was scoring against the wrong column.** The CSV has `total_birds` and a set of per-class columns. I scored detection against `total_birds` for weeks. The dialog in the screenshot prints its own *Total Count*, so I compared both against it: the dot count is `category_sum`, the sum of the per-class columns. `total_birds` excludes chicks and undercounts by up to 57%, mean error 14.75 against 0.25. Every accuracy number measured before this is suspect, including the 70.8% in the first prototype.

**21. Four images is not a benchmark.** The four study images were picked to be hard, and I then tuned colour thresholds until they worked on those four. That is overfitting with extra steps. Rebuilt as 63 stratified pairs: 7 years × 3 density bands × 3 images, 40 colonies, 7 to 2,037 dots. Detection numbers dropped immediately, which is the point. Sampling by band matters because the median otherwise hides a failure that only happens at one end.

**22. The ground truth can be wrong too.** One dense frame looked like a bad regression, 450 down to 9. Opening it showed a "No photo coverage for this area" frame: no dots at all, just survey polygons and a text box. The 450 is an estimate typed into that box. The old detector had been scoring 288 by counting polygon lines. Check frames by eye, not by count.

**23. Marker to class is per image, not global.** I assumed a filled circle meant the same thing everywhere and started building a global shape dictionary. It doesn't. "Site" is a filled circle in one image and an asterisk in the next. The fix is to parse each dialog on its own and cut a 24×24 template from each row's own glyph, so nothing is assumed across images.

**24. OCR failed because the fixture was small, not because OCR was wrong.** I abandoned OCR at 4% precision (#5). That test ran on `sample_screenshot.png`, which is 668×317, roughly 2.3× smaller than the real screenshots. Markers are 5-7px there and 12-18px at full resolution. Retested at full resolution with Tesseract plus fuzzy matching against the 98 species codes: class names read 15/15 on one image. Count digits are around 10px and still read only 60-65%, which is now the main open limitation. Lesson: check what your test fixture actually is before concluding the method is at fault.

**25. The dialog is a floating box, not a right-hand panel.** Stage 1 split each screenshot at roughly 50% width and called the left half aerial. The dialog is a floating window that sits top-right, right, or bottom-right depending on the image, so the split was discarding much of the aerial and every bird in it. This was the "boundary detection cutting annotations" problem raised in review. Finding the dialog as a box and keeping everything else works on 14/14 test images.

**26. Ask what changed, not what colour it is.** Colour thresholding asks whether a pixel falls in a band, which leaves, water glints and red map labels also do. The clean original photograph still exists, so the better question is whether a pixel is present in the screenshot and absent from the photograph. Registering the original and subtracting cut detection error from 8.40× to 1.24× median.

I also wrote here that vegetation cancels out because it is in both images. **That is wrong**, and hand labels later showed how wrong: on a mangrove frame with nine real markers the pipeline returned 128 detections, nearly all of them foliage.

The mechanism took looking at the overlays to see. The screenshot is the original downscaled about 4x and re-encoded as JPEG. Where the scene is smooth, the two renders agree closely and the difference cancels. Where it carries fine high-contrast detail, downscaling and compression destroy that detail *differently* in each render, so a residual survives at exactly the scale of a marker. The shift-tolerant comparison cancels an edge that merely moved; it cannot cancel a texture whose fine structure came out different in the two images.

That is why the failures group the way they do. Bright dead branches on dark mangrove, white speckle in a canopy, dark debris on white sand — all of them are marker-sized high-contrast specks. A smooth marsh frame in the same set scores 0.82 precision. Subtraction removes the background it can reproduce, not the background it cannot.

**27. A registration step should refuse, not guess.** SIFT and RANSAC will happily return a homography from bad matches, and a mis-warped original makes the subtraction produce garbage everywhere. Adding a quality gate that returns a refusal instead means 96.7% success at 0.38px median reprojection error, and the 3.3% that fail fall back to the colour path rather than producing nonsense.

**28. Filter noise before measuring, not after.** On near-empty frames the leftover water glint and mudflat texture outnumbered the real dots, and that noise was distorting the marker-size estimate used to split merged blobs. Applying a saturation floor before estimating size, rather than after, cut sparse over-detection from 2.96× to 2.13× and medium from 1.59× to 1.24×.

**29. A method tested on its own templates will flatter itself.** Legend self-recovery degrades a legend glyph and checks that it recovers its own class. It reports 76-83%. On real aerial dots the same method scores 0.36. The template and the test glyph come from the same pixels, so the test was close to circular. Both numbers are now reported together, because quoting only the first one next to a figure of real behaviour is how you lose a reviewer's trust.

**30. Counts cannot tell you whether labels are right.** Where two classes share both a colour and a shape, their templates score almost level: WHIB site against WHIB bird scored 0.548 to 0.540 on the same dot. The pipeline split the dots roughly correctly, 89 and 254 against a truth of 232 and 86, and then swapped the two labels wholesale. Count-based scoring cannot see this, and the count-guided top-N filter caps group sizes without reassigning anything, so it doesn't fix it either. Improving classification needs hand-labelled dots, because counts are a proxy that a wrong answer can satisfy.

---

## Measuring what the pipeline actually does

**31. A count cannot tell you whether detection works.** For months detection was scored as detected-count over true-count, and it read a healthy 1.24×. Then 541 dots were hand-labelled and the same pipeline scored 0.138 precision. The count is not a wrong number. It answers a different question, and "found all 61 markers", "found 40 and invented 21" and "found 61 on empty water" all score 61 on it. Everything that had been unmeasurable until then, precision and recall and placement error and per-dot class accuracy, turned out to be where the problems were.

**32. The bug the count metric hid.** Chrome-masking judged whether a region was window furniture by the median saturation over a morphologically closed component. Closing bridges a scattered colony into one region covering a quarter of the frame, so the median measures the background *between* markers rather than the markers. It was deleting 92 of 345 real markers, 53 of 71 on one frame, and no count anywhere in the benchmark moved. Measuring the region's ink instead separates cleanly: marker regions run 10.9–94.9% saturated ink, dialogs 0.0–2.5%, nothing in between.

**33. Label so the labeller can disagree with the pipeline.** Seeding the labelling tool with the detector's output makes the work fast, but it will inflate recall if the labeller only confirms what is shown. Three things kept it honest: a sweep grid that credits a tile only at 2× zoom or closer (a 4px marker is invisible below that), two frames labelled with no seeds as a control, and the pipeline's own class guess stored in the file but never displayed. Seven of eight frames then matched the survey count exactly, on blind frames as well as seeded — which is the check that the labels are real.

**34. Knowing which images to trust beat making all images work.** Five ways of separating real markers from background were measured: saturation, area, elongation, template match score, and a learned model over all of them. None generalised; the discriminative direction reverses between frames. What worked instead was arithmetic. Correct detections can never outnumber the dots on the image, so precision is capped at present over reported. A frame reporting seven times what it holds cannot exceed 14% precision, and that needs no labels to compute. Verified on three frames: ratios of 1.00, 0.99 and 0.62 gave precision 1.00, 0.82 and 0.74. About 48% of the corpus passes, holding 72% of its dots.

**35. Check what an old finding actually compared.** Item 18 says position accuracy beats data quantity, and it was used to argue that a smaller clean dataset would be fine. Re-reading the experiment: it compared 920 accurate annotations against 3,851 that *included* those same 920 plus badly-positioned ones. It shows that mixing in bad data hurts. It never tested 3,851 accurate annotations, so it says nothing about how much data is enough. The finding is real and the use of it was not.

**36. A fix measured on broken data is not a fix.** Three replacements for the marker-size estimator were measured and compared carefully, on a benchmark where the chrome-masking bug was still deleting a quarter of the ink. The diagnosis behind them was correct and worth keeping: a distance transform reads the stroke half-width on outline glyphs, not the marker footprint. The fixes were fitted to a symptom and had to be dropped. Fix the upstream bug first, then re-derive.

---

## Building the classification stage

**37. A row that reads nothing must not therefore hold everything.** Each legend row is capped at the Count the dialog states, but the code applied a cap only where a count actually read. A row with no readable count stayed unlimited, which is exactly the state a row parsed *outside* the table is in. On one frame two such rows — one sitting on the horizontal scrollbar, one on the photograph below the dialog — took 49 detections between them, 31 of which were real markers of a class that then starved. The missing cap sat precisely where it was most needed. Capping only the rows parsed *below* the last counted row fixed it and moved per-dot accuracy 0.725 to 0.766. Capping every uncapped row instead destroys frames whose Count column barely reads: one fell 0.516 to 0.065.

**38. The colour drift from legend glyph to aerial marker is per class, not per frame.** A legend glyph is drawn crisply on a white table cell; the same marker in the aerial is a few pixels of thin stroke over vegetation at a quarter of the resolution, and its measured colour moves. Correcting that with one shift per frame is the wrong shape. On one frame the frame-wide median difference is `a = −12.5` while one row needs `−29.4` and another `−11.0`; on another, two rows of the *same species and colour* need `L = −84` and `L = −16`. The glyph explains it: a thin asterisk mixes with whatever it sits on, where a filled circle keeps its own colour. So the drift follows the shape, not the class. Measuring it per row moved 0.766 to 0.789.

**39. An estimate is wrong exactly where you need it most.** Those per-row offsets are estimated from the dots a first pass assigned, which works only where the first pass was right. Measured against hand labels, the estimate lands within 3.0 of the truth on a frame the pipeline handles well, and 76.7 away on a row that had absorbed a neighbour's dots. The containment matters more than the accuracy: the offsets are allowed to **add** candidate rows and never to remove one, so a contaminated estimate can do no more than offer an extra row for the template score to reject. Trying to detect and discard the contaminated ones saved one dot and cost six.

**40. Colour gated the choice and then said nothing.** A dot either reached a legend row's colour margin or it did not, and after that the template score ranked the candidates alone. So a dot sitting exactly on a row's colour and one that scraped in at the edge competed as equals. Folding the colour distance into the score moved 0.789 to 0.828, the largest single step of the stage. The weight was swept and 0.15 to 0.30 all land between 0.81 and 0.83, so the value is not balanced on a peak; 0.20 was taken from inside that band because it is the only point where no frame falls.

**41. A dot with nowhere to go was dropped instead of being given the next best place.** Assignment already lets a displaced dot fall to its next best *candidate* row, so this only concerned the dot whose every candidate was full. On one frame that was 25 detections whose single candidate row is one the dialog genuinely counts as zero — the count was right, the candidate set was too narrow to also offer the row next door. Retrying only those dots against any row still inside the colour reject radius moved 0.828 to 0.850. Two boundaries had to be measured: a dot rejected on colour outright must stay unassigned, and the retry must run only in the final pass, because letting it place dots in the first pass changed which dots each row held and moved the offsets that pass exists to measure.

**42. Ordering work on a stale error split wastes it.** The plan put same-colour species first at 60 dots. After the capacity fix that bucket held 13, and a single legend-parsing artefact held 31. This happened twice. Re-measure the buckets after every change, before choosing the next piece of work.

**43. A number measured on more frames is not a regression.** Classification read 0.850 on four hand-labelled frames and 0.781 on seven, because three harder frames joined the set. The pipeline did not change between those two readings. Say which frames a figure covers, every time.

**44. Counts agreeing proves nothing, and here is the sharpest case.** On one frame two legend rows each match the dialog's stated count exactly — two stated, two assigned — and both are wrong on every dot. Each row is capped at its count, so agreement is partly built in by construction. Only hand labels see it. The classification figure now rings each patch green or red by whether a hand label agrees, which is what made this visible at all.

**45. White is not red.** In that figure a patch with no hand label is ringed white, and it is tempting to read white as an error. Hand labelling is not exhaustive: 90 of one frame's 386 detections carry no label, and 252 of another's 463. An unlabelled detection is unverified. The figure states the split in its own title so a reader cannot mistake one for the other.

**46. A legend-parsing change can be a classification change in disguise.** The Name|Count divider is taken from the second table gridline, and that is wrong on 14 of the 25 frames the pipeline classifies: on one, the gridlines are `[8, 110, 195, 315]` while the marker sits at x=122, so the strip read as the Count column is really the Name column, and a dialog plainly showing 93, 70, 11, 23 comes back as 3, None, 0, 0. That frame scores 0.193 against 0.781 pooled. Taking the first gridline right of the marker is the correct geometry and improved species resolution 0.771 to 0.858 — and broke classification twice, 0.861 to 0.634 and then 0.861 to 0.600. Moving the name boundary changes the parsed `class_name`, and row identity is keyed on that. Both attempts were reverted. The lesson is about the gate, not the geometry: this has to be scored on per-dot accuracy, never on name coverage.

**47. A misread zero is worse than no reading at all.** Where OCR returns `None` for a count, the row stays uncapped and behaves as it always did. Where it returns `0` on a row the dialog clearly numbers, that row is closed to everything. The same frame that reads 93 as 3 also reads 11, 23 and 10 as 0, and 66 of its 83 labelled dots end up with no class. Any count reader should prefer refusing to guessing.

**48. A class holding two or three dots tells you nothing.** On one frame every row with 40 dots or more is right almost throughout, and both rows holding two dots are wrong on every dot. Per-class accuracy on a tiny class is noise, and a frame with few classes scores high for that reason alone — one reads 1.000 with two classes. Report the denominator beside every such number.

**49. A proxy can rank correctly and still read high.** With hand labels on only 4 of 25 frames, the standing check is a per-species tally against the survey manifest, which the pipeline never reads. Three frames were then labelled deliberately to test it, chosen to span its range. It predicted 1.000, 0.995 and 0.403; they measured 0.877, 0.730 and 0.193. The ordering holds, so the proxy can say which frames are weak. The values do not, so it cannot say how weak.

---

## Mapping the dots onto the photographs

**50. The transform does not land where you think it lands.** `align.H` maps screenshot pixels to *work-scale* original pixels, not to the original, because the original is downscaled before SIFT runs. The full-resolution coordinate needs `perspectiveTransform(p, H) / res.scale`. Drop the divide and the coordinates stay internally consistent, look plausibly sized, and sit in the wrong part of the photograph by a factor of `scale`. Nothing raises. The only way to catch it is to draw the dots on the photograph and look.

**51. Spacing between dots is not bird size.** The first attempt at a box size derived it from how far apart the dots sit. That measures crowding: nearest-neighbour distance runs 13.6px on a packed colony to 215px on a sparse frame, a sixteenfold spread, while the birds themselves differ about twofold. A box built on it is four times too large on exactly the sparse frames where a bird is easiest to see.

**52. An equivalent diameter is not a length.** The second attempt measured the bird as a circle of the same area as its blob. For a bird twice as long as it is wide, that diameter is 0.71 of the length, so every box was built from about 70% of the bird and cut it in half. On one frame the ibis measures 42px long where the equivalent diameter reads 20px, and 0.61m of White Ibis at about 1.5cm per pixel is 41px. The long side of a minimum-area rectangle is the measurement that agrees with the species.

**53. Judging a box size from three frames is how a wrong one survives.** A flat 100px box looked defensible on the sparse frames and was four to eight times too large on the dense ones. It took drawing all 25 frames on one page to see it. The same figure now draws every exported frame, because a sample that happens to contain the easy cases will confirm whatever it is shown.

**54. The measurement has to agree with something outside itself.** Dividing known species body lengths by the measured bird size gives 1.3 to 4.0 cm per pixel, which is what these surveys fly. The EXIF then explains the spread rather than merely agreeing with it: focal lengths from 28mm to 300mm and pixel pitches from 4.4 to 6.6um predict exactly the tenfold range in bird size the measurement finds. A number that only agrees with itself is not evidence.

---

## Training against the right ground truth

**55. Training on your own annotations and scoring against them measures agreement, not accuracy.** One run reported F1 0.225 to 0.267 and looked like an 18% gain. The same kind of run scored against 1,647 hand-placed dots instead put the fine-tuned model behind the pretrained one, 0.360 against 0.369. Same data, same architecture, opposite conclusions. Fine-tuning teaches the model the pipeline's habits, and scoring against the pipeline rewards exactly that.

**56. One threshold can flatter whichever model suits it.** That 18% compared both models at 0.1, which sits near the fine-tuned model's optimum and not the pretrained one's. Each at its own best threshold gives 0.267 against 0.250, so +7%. Both numbers are real and they answer different questions; lead with the conservative one and report the sweep.

**57. Fine-tuning on noisy labels lowers confidence rather than raising error.** Above threshold 0.3 the fine-tuned model's recall collapsed to 0.093 where the pretrained model held 0.161. It finds more birds and scores every one of them lower. That is what training on labels containing false positives does, and it shows up as a shifted operating range rather than as an obvious mistake.

**58. Hand labels seeded from the pipeline lean toward it.** Ten of the twelve label sets began as the pipeline's own output for a person to confirm, delete or add. They deleted 107 of 116 on one frame and added 81 on another, so the review was real, but the labels still favour the pipeline. That makes any test against them generous to us, which is worth saying out loud when the pipeline fails it anyway.

---

## DeepForest, three things that cost hours

**59. `config.score_thresh` never reaches the model.** Setting it leaves `model.model.score_thresh` untouched, so a run configured for 0.3 silently evaluates at 0.1. Both models used the same wrong value, so that comparison survived, but the recorded configuration was wrong. Set the attribute on the model.

**60. `evaluate()` is deprecated and reports no mAP.** DeepForest 2.0 asks for `trainer.validate()` instead. Its signature is also `evaluate(csv_file, iou_threshold, root_dir)`, so passing the tile directory positionally lands it in `iou_threshold`, which fails quietly rather than loudly.

**61. `trainer.validate()` returns only losses unless you ask for the metrics.** mAP and precision and recall are computed only when `(current_epoch + 1) % val_accuracy_interval == 0`. The default interval is 20 and a standalone validate runs at epoch 0, so the check never passes: the full pass runs, eight minutes go by, and only the losses come back. Set `config.validation.val_accuracy_interval = 1`.

**62. A public method whose defaults crash is easy to mistake for a feature.** `Model.create_anchor_generator` offers sizes down to 8px, which looked like the fix for small boxes. Its own default is a single size tuple against five FPN levels and raises an assertion, and `create_model` never calls it. Small boxes train regardless: torchvision's RetinaNet matches with `allow_low_quality_matches=True`, so every ground-truth box is assigned its best anchor whatever the IoU.

---

## What scale settled

**63. "The data is poor" and "there is not enough of it" are different problems and look identical from one run.** 18 frames moved mAP by nothing, and that reads as bad data. The same pipeline, the same recipe and the same architecture at 349 frames moved mAP@50 from 0.036 to 0.087. Nothing about the data changed between the two runs except how much of it there was. Any conclusion drawn from a small training set is about the size of that set until a larger one says otherwise.

**64. Precision and recall rising together is the claim; either one alone is not.** A model that only gained recall would be drawing more boxes and catching a few more birds by volume. On three of four thresholds both rose. The clearest single frame drew **fewer** boxes than the pretrained model, 828 against 1,103, and found nearly twice as many birds, 373 against 200.

**65. Beating the other model's best score at two thresholds beats winning at one.** Best against best was 0.288 to 0.333. The stronger statement is that the fine-tuned model passed 0.288 at both 0.10 and 0.20. One win can be a cutoff that happens to suit it; two is harder to explain that way.

**66. Batch size set for training carries into every validation pass after it.** The pretrained model was scored at batch size 1 and the fine-tuned one at 4, because `config.batch_size = 4` was set before `fit` and never reset. mAP accumulates over the epoch and is immune, but precision and recall were uncontrolled, and best F1 is built entirely from those two. Re-running both models one tile at a time put pretrained precision at 0.195 against 0.194, so the comparison held. The lesson is to notice which metric a headline actually rests on, not to trust that a caveat written two paragraphs down protects it.

**67. Count the frames the gain appears on, not just the average.** 40 of 60 held-out frames improved, 15 got worse, 5 stayed level. An identical average built from three spectacular frames and 57 flat ones would mean something quite different, and no pooled number distinguishes them.

**68. A figure drawn honestly shows your own noise as well as your result.** The before-and-after overlays put unmatched predictions in the vegetation around a colony rather than among the birds, which says grass texture costs precision and neighbour confusion does not. They also show recovered annotations in neat rows along one frame's edge, which are painted map labels sharing the markers' palette colours. That noise is in the training data, and hiding it would have made a weaker figure and a less trustworthy one.
