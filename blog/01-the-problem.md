# The Problem: 2.8 Million Bird Annotations Trapped Inside Screenshots

*Deep in the Wild, part 2. First of six technical posts on recovering machine-readable bird annotations from eleven years of aerial survey screenshots.*

In April 2010 the Deepwater Horizon rig burned and sank, and oil came ashore along the Gulf of Mexico. What followed was the largest coastal bird survey the region has run. Between 2010 and 2021, aircraft flew the shorelines of Texas, Louisiana, Mississippi, Alabama and Florida and photographed the nesting colonies below. Back on the ground, people opened each photograph in a point-counting program and clicked once on every bird they could find.

Each click did two things. It added one to a running total, and it painted a small coloured dot onto the photograph at that spot. When an annotator finished an image, the program saved a screenshot of the screen: the photograph, the dots drawn over it, and a floating dialog box listing every class of bird with its marker and its count.

It never saved where the clicks were.

> **[IMAGE 1 HERE: `blog/images/post1-img1-study-images.png`]**

*Left: the screenshot, with the counting tool's dots and its dialog drawn over the photograph. Right: the same photograph, clean. Everything needed to read the annotations back is on the left, including the legend that says what each marker means.*

That sentence is the whole project. Eighteen people spent eleven years looking carefully at photographs and marking birds, and their work survives as pictures of dots rather than as data. You can see it. You cannot load it.

Over this summer I built a pipeline that reads it back. It finds each dot, works out which class of bird it belongs to using the legend in that same image, and places it on the original high-resolution photograph as a training annotation. It runs from the images alone. By the end of this series there is a dataset of **118,270 bird annotations across 413 photographs**, and a detector trained on it that measurably beats the model it started from.

This first post is about the two weeks before any of that existed, spent doing nothing but reading the data. It is the part I would defend hardest if I had to cut something.

## What actually survives

The archive belongs to **The Water Institute**, who ran the surveys and who make it publicly available at `twi-aviandata.s3.amazonaws.com`. Two facts about it shaped everything I built afterwards.

**There are 18,304 screenshots, and every one has a clean twin.** The unannotated, full-resolution photograph each screenshot was drawn on is still there, filed at the same relative path in a different folder. That pairing turned out to be the single most valuable thing in the dataset. If you hold the picture with the dots and the picture without, then a dot is simply whatever is present in one and absent from the other. The third post in this series is built entirely on that sentence.

**The archive is much larger than its description.** The project I applied for was advertised as recovering "340,000+ observations", and I repeated that figure in my last post and in my proposal. Reading the survey spreadsheet properly gives **2,810,895** individual bird observations, across 102 species, 442 colonies and 18 annotators, spread over 49,204 rows. About eight times the advertised number.

I took that straight to my mentors rather than quietly writing the bigger figure down, because a gap that size usually means two people are counting different things. It held up. Measured two independent ways, the archive really does hold close to 2.8 million human bird observations waiting to be recovered, which makes the prize a good deal larger than anyone had been saying.

## Why a count is not enough

If the spreadsheet already records how many birds are in each photograph, why bother recovering the dots?

Because a modern object detector does not learn from counts. It learns from positions. To train a model like [DeepForest](https://deepforest.readthedocs.io/) to find birds in aerial imagery you need a box around each bird, which means you need its coordinates. "This image contains 61 Laughing Gulls" teaches a detector nothing about what a Laughing Gull looks like from three hundred metres up. "There is one at pixel (2043, 1178)" teaches it a great deal.

So the survey produced two things. A set of counts, which are already useful and already used by ecologists. And a set of positions, which the software discarded the instant it saved the screenshot. Those positions are still on screen. They are just stored as coloured pixels instead of as numbers.

This shape of problem is everywhere. Ecology, medicine and astronomy are full of careful human annotation trapped inside a screenshot, a PDF or a printed figure, because the tool that produced it was built to show a person an answer rather than hand a computer one. My mentors were clear from the first meeting that the method matters more here than this one dataset. Nobody asked me to recover every last dot. They asked for an approach somebody else could copy.

## Two weeks of measuring before a line of pipeline code

I wanted to start writing a detector on day one. Not doing that is the best decision I made all summer.

Instead I mapped the bucket, read the spreadsheet properly, downloaded 25 images and measured dots by hand. Around 40% of my total time on this project went on measurement rather than building, and almost every constant in the finished code traces back to something measured here instead of guessed.

> **[IMAGE 2 HERE: `blog/images/post1-img2-dataset-analysis.png`]**

*What the survey spreadsheet holds, by species, by year, by annotator and by birds per image.*

Four findings changed what I built.

**Laughing Gull dominates, not Brown Pelican.** LAGU is 31% of every bird in the archive, at 859,840 observations. Brown Pelican gets far more attention after an oil spill and comes to 17%. I had been about to tune the whole detector around pelicans.

**Most images are easy.** The median photograph carries 66 dots and 63% carry fewer than 100. Only 391 images of 18,304 hold more than a thousand. My four original study images held 96 to 1,388 birds and I had chosen them precisely because they were hard, so for weeks I had been measuring the pipeline against the toughest end of the archive and reporting the result as its accuracy. Knowing the real distribution is what later let me stratify a proper benchmark instead.

**The dots are small and barely shaped.**

> **[IMAGE 3 HERE: `blog/images/post1-img3-dot-properties.png`]**

*Measured by hand across 1,199 dots.*

Median diameter **8 pixels**. Median circularity **0.57**, where a perfect circle scores 1.0. Six distinct colours. At eight pixels across, a dot drawn by a graphics routine and then squashed by JPEG compression is barely a shape at all, and that one measurement is why the classification stage five posts from now uses colour far more than shape.

**The colours spread wider than expected.** The hue and saturation scatter shows real clusters, but broad ones that overlap. My first detector used narrow bands, five hue units either side of each measured mean, and found 44% of the dots. Widening each band to cover its whole region took it to 58% straight away. Fading, compression and background showing through a thin stroke all shift a dot's colour further than the tidy version in my head allowed.

## Getting the ground truth right

The finding I am most glad I caught came last.

The spreadsheet has a column called `total_birds`, and about twenty per-class columns beside it: Site, WBN, ChickNest, Brood, RoostingAdults. I scored detection against `total_birds` for weeks, because it has the obvious name.

The dot count is actually `category_sum`, the sum of the per-class columns. `total_birds` leaves categories out, chicks among them, and comes in short on 5,135 of the 18,252 images that carry any dots at all, by a median of 11%.

What settled it was the tool's own dialog. That floating box prints its own **Total Count**, which is the number the annotator saw when they finished, and it owes nothing to the spreadsheet. Reading it off four dialogs by eye and comparing both columns gave a mean error of **0.25** for `category_sum` against **14.75** for `total_birds`. Four images is a small test, so the corpus-wide check carries the weight. They agree, which is why I stopped looking.

That correction cost me every accuracy figure I had produced up to that point, including the 70.8% I quoted in my last post. It was worth it. What replaced it is a far stricter test, and one rule I held for the rest of the summer: **the pipeline gets the screenshot and the clean photograph and nothing else.** The spreadsheet may check the answer. It never gets to be an input.

Almost everything good that followed came from that rule.

## What comes next

By the end of these two weeks I had the shape of the problem, a ground truth I trusted, and no pipeline whatsoever. What I did have was the observation the rest of the project rests on: everything needed to read a screenshot is already inside that screenshot. The dots sit on the photograph. The key explaining what each colour and shape means sits in the dialog box beside them.

Reading that key is where the real work starts, and it begins with an assumption I got wrong: that a red plus sign means the same bird in every image.

It does not. That is the next post.

---

*Code and measurements: [github.com/weecology/recovering-computer-vision-annotations](https://github.com/weecology/recovering-computer-vision-annotations). Built during Google Summer of Code 2026 with the DeepForest project at Weecology Lab, University of Florida. Mentors: Dr. Ben Weinstein, Henry Senyondo, Dr. Ethan White and Dr. Josh Veitch-Michaelis.*
