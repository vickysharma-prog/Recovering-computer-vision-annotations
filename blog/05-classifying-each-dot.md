# Classifying Each Dot: Giving Every Bird a Species

*Deep in the Wild, part 6. Fifth of six technical posts on recovering machine-readable bird annotations from eleven years of aerial survey screenshots.*

At this point the pipeline can find the dots and prove where they are. Every kept detection sits within a pixel of a real marker, and there is an arithmetic check that says which photographs to trust before anything is built from them.

Each of those dots is still anonymous. This post is about giving it a name, which turned out to be the hardest measurement problem in the project, and about the single most useful thing I learned all summer: that two numbers agreeing tells you almost nothing.

## Why this is harder than it sounds

The obvious approach is to match each dot's colour to a legend row. That works right up until you look at a real dialog.

> **[IMAGE 1 HERE: `blog/images/post5-img1-classification-mechanism.jpg`]**

*Eight classes from one screenshot. Left to right: the marker in the dialog, the 24×24 template cut from it, and six aerial patches assigned to that class.*

Look at the first and third rows. `LAGU sit` is a red circle. `ROSP site` is also a red circle. Same colour, same shape, two different species.

Now rows two and eight. `WHIB site` is a blue circle and `WHIB bird` is a blue circle. Same species, same colour, same shape, and they mean different things: one is a nest site, the other is an adult bird standing on the ground. Ecologically that distinction is the entire point of the survey.

Colour alone merges all of these. Shape alone merges them too. And these dots are eight pixels wide, so there is not much shape to work with in the first place.

## The class is the row, not the name

One decision made everything else tractable. **A dot's class identity is the legend row it belongs to, not the text read off that row.**

This sounds like a technicality. It is not. On one dialog, two separate rows both come out of OCR as `ad`. Key the classifier on the name and those two rows merge into one, silently, and every dot in both is now ambiguous. Key it on the row and they stay distinct even when the text is unreadable.

It also means a row whose name never reads is still perfectly usable. The dots still get sorted correctly. They just carry a class the pipeline cannot yet put a word to, which is a labelling problem for later rather than a recovery failure.

## How a dot gets a class

Four steps, and the third is the one people find surprising.

**Colour proposes candidate rows**, using that image's own palette in LAB space rather than any fixed set of colours.

**A template match ranks them.** Normalised cross-correlation against the 24×24 glyph cut from each candidate row, with colour agreement folded into the score, so a dot sitting squarely on a row's colour outranks one that barely scraped inside the margin.

**Each row is capped at the count the dialog states.** If a row is full, the dot goes to its next best row instead. Without this, a populous `site` row stays half empty while a nearly empty `bird` row fills up with its dots, which is a wholesale label swap that per-class counts cannot detect.

**A dot matching no row keeps no class.** The pipeline is allowed to say "I don't know", which was something my mentor Josh specifically asked for. A wrong label costs more than a missing one, because a wrong label gets accepted at review and a missing one gets looked at.

## The finding I liked most

A legend glyph is drawn crisply on a white table cell. The same marker out on the photograph is a few pixels of thin stroke over vegetation, at a quarter of the resolution, so its measured colour drifts. Correcting for that drift is obviously necessary.

What I assumed was that the correction would be one number per image. It is not. **The drift is per row.**

On one frame the image-wide median shift is `a = −12.5`, while one row needs `−29.4` and another needs `−11.0`. On a different frame, two rows of the *same species in the same colour* need lightness corrections of `−84` and `−16`.

The glyph explains it. A thin asterisk mixes with whatever it is sitting on, so its colour moves a long way toward the background. A filled circle holds its own colour. The drift follows the **shape** of the marker, not the class. Measuring it per row instead of per image moved accuracy from 0.766 to 0.789.

There is a containment rule that makes this safe. Those per-row corrections are estimated from the dots a first pass assigned, so they are least reliable exactly where the first pass was worst. So they are only ever allowed to *add* a candidate row, never to remove one. A bad estimate can do no more than offer an extra option for the template score to reject.

## The result

> **[IMAGE 2 HERE: `blog/images/post5-img2-accuracy-by-frame.png`]**

*Per-dot accuracy on each frame, with its own denominator.*

**0.781 per dot: 691 correct out of 885**, on the seven frames that both pass frame selection and carry a hand label on every dot. Dropping one badly broken frame gives 0.842.

Two things about that chart matter as much as the headline. The top bar reads 1.000, and it reads that way because the frame has only two classes, so it is not evidence of anything. And each bar carries its own denominator, because a class holding two dots tells you nothing at all.

In the mechanism figure earlier, the rings carry the verdict: green where a hand label agrees, red where it disagrees, **white where no label exists there**. White is not an error. Hand labelling is not exhaustive, and 90 of that frame's 386 detections have no label. An unringed patch is unverified, not wrong.

## The lesson worth taking away

Look at the bottom two rows of the mechanism figure. `BCNH site` holds two dots. `WHIB bird` holds two dots. Both rows **match the count the dialog states exactly**.

Both are wrong on every single dot.

That is the clearest thing in this repository. Each row is capped at its stated count, so the counts agreeing is partly built in by construction. A pipeline can produce per-class totals that match the ground truth perfectly and still have shuffled the labels wholesale. Only positions checked by a person can see it.

The same applies to a test I ran earlier and now always report twice. Taking a legend glyph, shrinking it to aerial scale, and checking the matcher recovers its own class scores **76 to 83%**. Against hand-labelled dots the same matcher scores **0.781**. The first number is flattering because the template and the test image come from the same pixels. Both get quoted, or neither.

## What this delivers

Every recovered dot now carries a species and a category, read entirely from the image it came from. No survey spreadsheet, no global assumptions, no per-year tuning.

That is a complete recovery: position, and identity, from a screenshot that was never meant to be read by anything but a person.

The last post is what happens when you hand that to a model.

---

*Code and measurements: [github.com/weecology/recovering-computer-vision-annotations](https://github.com/weecology/recovering-computer-vision-annotations). Built during Google Summer of Code 2026 with the DeepForest project at Weecology Lab, University of Florida. Mentors: Dr. Ben Weinstein, Henry Senyondo, Dr. Ethan White and Dr. Josh Veitch-Michaelis.*
