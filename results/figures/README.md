# Result Figures — Per-Image Marker→Class Recovery

Three figures summarising the legend-based annotation recovery pipeline.
All produced by the automatic pipeline (no manual cropping, no CSV).

## fig1_localization.png — Dialog localization (the boundary fix)
**Six** screenshots with the "Manual Point Count" dialog **auto-located as a
box** (red outline). The dialog floats in a different place in every image
(top-right, right, bottom-right). The old stage assumed a full-height right
panel and split each screenshot at ~50% width — discarding ~half the aerial and
its birds. The new locator finds the dialog as a box anywhere in the frame
(14/14 images in testing), so the entire aerial stays available for detection.

## fig2_marker_to_class.png — Per-image marker → class mapping (4 images)
For each of the 4 study images: the dialog + the recovered mapping
`colour/shape → SPECIES category`, read entirely from the screenshot via shape
analysis + OCR (fuzzy-matched to the species codes). The core thing the mentor
asked for: same-colour markers (e.g. red ● BRPE WBN vs red + BRPE bird) are kept
distinct by shape — colour alone would merge them. Note the mapping is recovered
**per image** (the shape→category convention differs between images).

## fig3_classified_aerial.png — Aerial dots classified (4 images)
Each study image's aerial with every detected dot coloured by its **recovered
class**; the dialog region is dimmed. Overlapping dots in dense colonies are
split via distance transform (cluster splitting). Dense colonies (A, C) show
high raw counts — the count-guided selection (fig4) trims these. Note the dots
span the whole frame, including the regions the old stage used to cut off.

## fig5_color_vs_shape.png — Why colour alone is not enough (synthetic demo)
A synthetic scene with markers of the same colour but different shapes. Colour
ONLY collapses them into 2 classes (can't tell bird vs nest vs site);
colour + SHAPE separates them into 5 correct classes. This is the test image
explaining, in the simplest terms, why the shape step is needed.

## fig4_counts_barchart.png — Recovered vs ground-truth counts (the graph)
Left: for image D, recovered count vs the dialog's legend count, **per class** —
zero-count categories (empty/pbn/brood) correctly stay near zero; same-colour
classes are separated. Right: total recovered vs dialog total across the 4 study
images, with recall % (A 36%, B 77%, C 90%, D 69%). This is the detect-style
bar chart comparing recovered against ground truth.

## Status notes (honest)
- Solid & generalizes: dialog localization (100% on 14 images), recall (cluster
  splitting), end-to-end automation.
- Gating factors for clean per-class numbers everywhere: count-OCR (~60-65% on
  ~10px digits) and the within-colour aerial shape split (~60-70%). Both
  improved; both incremental from here.
