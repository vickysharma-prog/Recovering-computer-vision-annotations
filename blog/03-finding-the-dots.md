# Finding the Dots: Why Colour Detection Failed and Subtraction Worked

*Deep in the Wild, part 4. Third of six technical posts on recovering machine-readable bird annotations from eleven years of aerial survey screenshots.*

By the end of the last post the pipeline could read any screenshot's legend: find the dialog, name every class, and keep a small picture of the exact marker that class uses.

It still could not find a single dot on the photograph.

This post is about how it learned to, and about realising that the obvious approach was not merely imperfect but asking the wrong question.

## The obvious approach

The dots are bright, saturated colours painted onto a natural scene, so look for pixels of those colours. Convert to HSV, build a band around each marker colour, keep every pixel inside it, group the survivors into blobs, call each blob a dot.

I built that. It is what the first prototype ran on, and the numbers looked reasonable for a while.

Then I built a proper benchmark and it fell apart.

> **[IMAGE 1 HERE: `blog/images/post3-img1-before-after.jpg`]**

*The same photograph, two methods. Colour thresholds on the left, subtraction on the right.*

Look at the left panel. This frame contains **64** annotation dots. Colour thresholding returned **3,144**, forty-nine times too many. The image is not covered in birds. It is covered in vegetation, sand and water that happen to fall inside a red band.

This is not a tuning problem, and narrowing the bands does not fix it. A colour threshold asks: *is this pixel inside this range?* Sunlit leaves are. Water glints are. So are the red survey lines and painted area labels the surveyors drew on these photographs, because they used the same palette as the markers.

The question itself is wrong. What I actually wanted to ask was: **is this pixel part of something that was added to the photograph?**

## Asking what changed instead

That question has an exact answer available, and it had been sitting in the bucket the whole time. Every screenshot has a clean twin, the original unannotated photograph, filed at the same relative path.

A dot is whatever is present in the screenshot and absent from the photograph. Not "reddish". Added.

Two steps make it work.

**Line the two images up.** The screenshot shows the photograph shrunk and redrawn inside a window, so before anything can be compared the two have to be registered. `align.py` does this with SIFT feature matching and RANSAC.

The design decision that mattered here was not the algorithm. It was making it **refuse**. SIFT and RANSAC will happily hand back a transform derived from bad matches, and a badly aligned original makes the subtraction produce garbage across the entire frame. At the scale of 18,000 images, a silent bad warp is far worse than an admitted failure, because it marks a whole photograph as annotation and nothing raises an error. So registration returns a refusal rather than a transform it cannot vouch for.

With that gate in place it succeeds on **58 of 60** benchmark pairs, a 96.7% success rate, at a median reprojection error of **0.38 pixels**. The 3% that refuse fall back to the old colour path instead of poisoning the dataset.

**Subtract, and pay attention to colour rather than brightness.** The rest is the difference between the two images. Measured across the benchmark, grouped by how many dots each image holds:

| density band | colour thresholds | subtraction |
|---|---|---|
| dense | 3.56× over | **1.01×** |
| medium | 9.15× over | **1.42×** |
| sparse | 63.51× over | **6.07×** |
| **overall median** | **8.40× over** | **1.46×** |

A later fix, filtering noise *before* measuring rather than after, brought the overall figure to **1.24×**. From eight times too many detections to roughly a quarter too many.

## Confirmation from the people who own the data

Partway through this work, **The Water Institute** opened a discussion on the public repository.

They are not a bystander here. This entire archive is theirs. The bucket every screenshot in this project comes from, `twi-aviandata`, is their bucket, and the surveys behind it are their programme. So when they took the time to read the work and write up what they had learned recovering annotations from the same imagery, it was the data provider commenting on an attempt to recover their own data.

They had hit the same walls, with colour thresholds tuned for one survey year overfitting that year. And they had reached the same conclusion in almost the same words: stop asking which pixels have the annotation colours, and start asking which pixels differ between the screenshot and the clean original. They had also moved to LAB colour space rather than HSV, which I had arrived at separately, because light red and dark red markers do not separate on hue alone.

Two groups working independently and converging on the same two decisions is better evidence than either group's benchmark, and it means far more coming from the organisation that produced the archive in the first place. Their note also handed me something I had missed: the alignment that makes subtraction possible is the same alignment that reliably locates title bars and scroll bars, so they can be masked out before detection runs.

My thanks to them, both for the archive and for engaging with this so openly. The thread is [discussion #6](https://github.com/vickysharma-prog/Recovering-computer-vision-annotations/discussions/6).

## Where subtraction still struggles, and why

I first wrote that vegetation cancels out, since the leaves are in both images. That is not quite right, and one frame proved it: a mangrove colony carrying nine real markers returned **128** detections, nearly all foliage.

The reason is that the screenshot is not the original photograph. It is the original shrunk roughly fourfold and re-compressed as a JPEG. Where a scene is smooth, both versions agree closely and the difference cancels cleanly. Where a scene carries fine high-contrast detail, shrinking and compression destroy that detail *differently* in each version, leaving a residual at exactly the size of a marker.

That predicts precisely which frames struggle. Bright dead branches on dark mangrove, white speckle in a canopy, dark debris on pale sand: all marker-sized, high-contrast specks. A smooth marsh frame in the same benchmark scores 0.82 precision with no special handling at all.

Subtraction removes the background it can reproduce, and only that. Knowing the rule is what later made it possible to tell in advance which images the pipeline should be trusted on.

## Dense colonies needed one more idea

The opposite failure happens where birds pack together. Overlapping dots merge into a single blob, and a blob counted once is one dot instead of nine.

> **[IMAGE 2 HERE: `blog/images/post3-img2-dense-colony.jpg`]**

*A colony holding 1,050 dots. Subtraction with cluster splitting recovers 1,149, within 9%.*

The fix is a distance transform, which measures how far each ink pixel sits from the nearest edge of its blob. In a merged clump the local peaks of that measurement sit at the centre of each original marker, so the clump can be split back into its parts. On one dense study image this took detection from 391 dots to **637**, against a true count of **636**, moving recall from around 61% to essentially complete.

## The benchmark had to be rebuilt too

None of these numbers would mean anything on my original four study images, which I had picked because they were hard and then tuned thresholds against until they worked. That is overfitting with extra steps.

The benchmark is now **63 stratified pairs**: seven survey years by three density bands by three images, 40 colonies, dot counts from 7 to 2,037. Sampling by band keeps the dense tail in, because that is where detection was weakest and a median would have hidden it.

Detection scores dropped the moment I switched to it. That was the point.

## Where that leaves the pipeline

Put together, this stage is the one that turned the project from an idea into something that works. The pipeline can now take any screenshot in the archive, line up its clean original, and pull out the annotations that were painted onto it, including the packed colonies of a thousand birds that the first detector collapsed into a handful of blobs. It does this from the two images alone. No spreadsheet, no per-year tuning, nothing hand-set for a particular colony.

That is the core of the recovery, and it works.

## The problem with everything in this post

There is one catch, and it is a real one. Every number above is a ratio of counts. Detected dots over true dots. 8.40 became 1.46 became 1.24, and each of those looked like progress.

A count cannot tell you whether detection works.

"Found all 64 markers", "found 40 real ones and invented 24" and "found 64 things on empty water" all score exactly 64. The ratio is blind to whether a single detection sits on an actual dot.

I did not know that at the time. Finding out is the next post, and it was the most uncomfortable result of the project.

---

*Code and measurements: [github.com/weecology/recovering-computer-vision-annotations](https://github.com/weecology/recovering-computer-vision-annotations). Built during Google Summer of Code 2026 with the DeepForest project at Weecology Lab, University of Florida. Mentors: Dr. Ben Weinstein, Henry Senyondo, Dr. Ethan White and Dr. Josh Veitch-Michaelis.*
