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

I also wrote here that vegetation cancels out because it is in both images. **That is wrong**, and hand labels later showed how wrong: on a mangrove frame with nine real markers the pipeline returned 128 detections, nearly all of them foliage. Vegetation cancels where the two renders agree closely, and on textured canopy they do not — the difference survives, and at this resolution a leaf and a marker are the same handful of pixels. Subtraction removes the *uniform* background, not the textured one.

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
