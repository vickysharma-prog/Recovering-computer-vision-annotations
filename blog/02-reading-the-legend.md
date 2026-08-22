# Reading the Legend: Teaching the Pipeline What Each Coloured Dot Means

*Deep in the Wild, part 3. Second of six technical posts on recovering machine-readable bird annotations from eleven years of aerial survey screenshots.*

The last post ended on one observation: everything needed to read a screenshot is already inside that screenshot. The dots sit on the photograph. The key explaining them sits in a dialog box in the corner.

This post is about reading that key, and about two assumptions I made early that were both wrong.

## The problem in one sentence

Knowing there is a bird at pixel (2043, 1178) is a start, but a training annotation has to say *what* is there. Brown Pelican or Laughing Gull. An adult bird, a nest, or a chick.

All of that is encoded in how the dot looks. The counting tool drew each class in its own colour and shape, and printed the whole scheme in its dialog: one row per class, each showing the marker, the class name and the count.

So the job is to read that table. Which sounds small, and was not.

## Mistake one: assuming the dialog is a panel

My first pipeline split every screenshot down the middle. The left half was the photograph, the right half was the dialog. It looked right on the first few images I tried.

The dialog is not a panel. It is a small floating window, and it lands wherever the annotator happened to leave it.

> **[IMAGE 1 HERE: `blog/images/post2-img1-dialog-located.png`]**

*Six screenshots with the dialog found automatically and outlined in red. It sits right-of-centre, hard right, bottom-right and lower-right corner. There is no fixed position to assume.*

Splitting at the halfway line therefore threw away a large part of the photograph, and every bird that was in it. My mentor Josh spotted this in review and put it plainly: the boundary detection was cutting annotations. He was right, and it was the most useful piece of feedback I got that month, because I had been measuring recall against a photograph I had already truncated. The detector was not missing those birds. I had deleted them before it ever looked.

The fix was to stop assuming and start searching. `locate_dialog` hunts for the thing the dialog actually is: a flat grey panel containing a tight, regularly spaced column of small coloured markers. Nothing else in an aerial photograph looks like that. It finds the window as a box wherever it sits, and everything outside the box stays available as photograph.

I then ran it over the four study dialogs plus ten screenshots pulled at random from the bucket, different colonies, cameras and years. It found the dialog in **14 out of 14**.

## Mistake two: assuming a shape means the same thing everywhere

With the dialog located, I started building what felt obvious: a dictionary. Filled circle means site. Plus means bird. Asterisk means chick. Build it once from a few images, apply it to all 18,304.

Then I read four dialogs side by side.

> **[IMAGE 2 HERE: `blog/images/post2-img2-marker-to-class.png`]**

*Four dialogs, each parsed by the live pipeline into marker, class and count. 80 of the 82 rows across them are named correctly.*

Look at what the parser reads out of images B and D.

In **image B**, a red plus is `BRPE wbn`, a white bird nest. In **image D**, a red plus is `BRPE bird`, an adult pelican. The identical marker, two different classes.

Now the same thing from the other direction. `LAGU site`, a Laughing Gull nest site, is a **yellow circle** in image A and a **blue star** in image B. Identical class, two completely different markers, and not even the same colour.

A global dictionary would have been wrong on both, silently, across thousands of images. It would have produced a dataset that looked complete and was mislabelled, which is worse than one that visibly fails.

So the rule became: **the marker-to-class map is read per image, never assumed.** This is also exactly what my mentors asked for at the start, phrased as "it should work for any image in isolation". It took building the wrong thing to understand why they had said it.

There is a neat consequence. Since the parser reads each row's marker from that image's own dialog, it can cut a **24 by 24 pixel picture of the glyph** straight out of the table and keep it. The legend glyph and the dot on the photograph are not merely similar. They are the same marker, drawn by the same code, in the same image. That template is what the classifier matches against later in this series, and it costs nothing, because the tool drew it for us.

## The failure I had to reverse

There was still a piece missing. The dialog shows a marker and a count, but the class name next to it is text, and text means optical character recognition.

I had already tried OCR early in the project and written it off. Tesseract on the dialog, upscaling, adaptive thresholding, fuzzy matching against species codes. It scored **4% precision**. I documented it as a dead end and moved on, which felt like good discipline at the time.

It was not a dead end. It was a bad test.

The fixture I had run it on, `sample_screenshot.png`, is 668 by 317 pixels. Real screenshots from the bucket are 1,160 to 1,580 pixels wide, so I had been testing on an image roughly **2.3 times too small**. At that size the legend markers are five to seven pixels across and the text is a smear. At full resolution the markers are twelve to eighteen pixels and the text is simply legible.

Retested at the real resolution, with Tesseract reading each row and the result fuzzy-matched against the survey's own list of species codes to repair small errors, it works. The fuzzy match is doing real work: OCR returns `BAPE` and the matcher corrects it to `BRPE`, returns `TAGU` and gets `LAGU`.

Measured properly, over 25 frames and 218 legend rows:

| | |
|---|---|
| rows whose class name reads | **0.904** |
| rows resolving to a species code | **0.771** |
| rows whose count reads | **0.693** |

Class names read on nine rows in ten, from the image alone, with no spreadsheet anywhere in the loop.

The reversal matters more than the number. I had killed a working method on one experiment run against the wrong data. The only reason I went back was that I had written down *why* it failed rather than just *that* it failed, and the reason did not survive a full-resolution image. Every dead end in this project now carries a root cause beside it, and two have since been reopened.

## What is still hard

The counts are the weak column, at 0.693. Those digits render around ten pixels tall in the original interface, and a ten-pixel `3` and a ten-pixel `8` are genuinely close. That number matters more than it looks, because the count is what later stops the classifier putting ninety dots into a class the dialog says holds two. It comes back in post five.

## Where this leaves us

At the end of this stage, given any screenshot, the pipeline can find the dialog, read every row, name the class, and hold a small picture of the exact marker that class uses in that image.

What it cannot do yet is find a single dot on the photograph.

That is next, and it starts with a method that seems obviously correct and fails badly: looking for pixels of the right colour.

---

*Code and measurements: [github.com/weecology/recovering-computer-vision-annotations](https://github.com/weecology/recovering-computer-vision-annotations). Built during Google Summer of Code 2026 with the DeepForest project at Weecology Lab, University of Florida. Mentors: Dr. Ben Weinstein, Henry Senyondo, Dr. Ethan White and Dr. Josh Veitch-Michaelis.*
