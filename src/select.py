"""
Frame selection: decide which screenshots the pipeline can be trusted on.

The pipeline does not have to recover every annotation. The project's stated
contribution is the *method* — "an example for many projects that seek to harness
historical data" — so choosing the frames it handles well is part of the method
rather than an evasion of it.

What was missing was a single definition of "handles well". The figure had been
re-derived by hand in analysis scripts each time it was quoted, which is why the
same rule appears as 48% of images in one place and 41.7% in another. This module
is that definition; evaluation and export should call it rather than each pick a
threshold.

## Where this sits in the pipeline

**After detection, before classification.** The rule needs the number of dots the
detector returned, so it cannot run first and hand a shortlist to detection:

    CSV                 curation - which images are worth attempting at all
      |                 (drop rows with no usable count, stratify by colony/band)
      v
    detection           produces `detected`
      v
    THIS MODULE         accept / reject on the bound below
      v
    classification -> export

## The rule is arithmetic, not a heuristic

Let `detected` be the number of dots the detector returns and `count` the number the
image actually holds. Two facts bound accuracy before anything is measured:

    a true positive is a detection    ->  TP <= detected
    a true positive is a real marker  ->  TP <= count

With `ratio = detected / count`, those are ceilings on the two metrics:

    precision = TP / detected  <=  count / detected  =  1 / ratio
    recall    = TP / count     <=  detected / count  =  ratio

A frame returning 7x the dots it holds cannot exceed 14% precision however good the
detector is; a frame returning a tenth of them cannot exceed 10% recall. Neither
statement needs a hand label, and neither can be argued with.

One quality target `q` therefore fixes the band in both directions:

    q <= ratio <= 1/q     <=>    both ceilings are at least q

`q` is a stated requirement, not a number fitted to the benchmark. That distinction
matters here: tuning constants against the 63-frame benchmark they are scored on is
a mistake this project has made twice and retracted twice.

## Why the bound has to be two-sided

It had been quoted one-sided (`ratio <= 1.25`), which bounds precision only. That
admits frames the detector barely saw: `19May18Camera1-Card1-00622` returns **one**
detection against 19 dots and passes a one-sided cut with a ratio of 0.05. A dataset
built from it contributes one annotation where nineteen exist. The recall ceiling
rejects those, using the same arithmetic.

## Where `count` comes from

The survey manifest (`category_sum`). That does not break the rule that the pipeline
works from the image alone: the rule governs what may be read *per image at
inference*, while choosing which images to curate into a dataset is a separate
question the survey data is available for. A self-contained variant can pass the
dialog's own Count value instead, which is why `accept_frame` takes the count as an
argument rather than reading a file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Both the precision and the recall ceiling of an accepted frame are at least this.
# Raise it for a smaller, cleaner dataset; lower it for coverage.
DEFAULT_QUALITY = 0.6


@dataclass(frozen=True)
class FrameDecision:
    """Why a frame was accepted or rejected, so a run can be audited afterwards."""
    accepted: bool
    ratio: Optional[float]
    precision_ceiling: Optional[float]
    recall_ceiling: Optional[float]
    reason: str


def frame_ratio(detected: int, count: Optional[float]) -> Optional[float]:
    """
    `detected / count`, or None when the count is missing or not positive.

    A zero count is not a bird-free frame the detector agreed about — it is a frame
    with no usable number, so no bound can be formed. `16May15Camera2-Card1-00097`
    is the cautionary case: `category_sum` reads 450 while the image carries no
    drawn dots at all, the figure having been typed into a "no photo coverage" box.
    """
    if count is None or count <= 0:
        return None
    return detected / float(count)


def accept_frame(
    detected: int, count: Optional[float], quality: float = DEFAULT_QUALITY,
) -> FrameDecision:
    """
    Decide whether a frame is worth carrying into classification and export.

    Accepts when `quality <= detected/count <= 1/quality`, which is exactly the
    condition that neither ceiling implied by the counts falls below `quality`.
    """
    if not 0 < quality <= 1:
        raise ValueError(f"quality must be in (0, 1], got {quality}")
    ratio = frame_ratio(detected, count)
    if ratio is None:
        return FrameDecision(False, None, None, None, "no usable count")
    p_ceiling = min(1.0, 1.0 / ratio) if ratio > 0 else 1.0
    r_ceiling = min(1.0, ratio)
    if ratio == 0:
        return FrameDecision(False, 0.0, p_ceiling, 0.0, "nothing detected")
    if ratio > 1.0 / quality:
        return FrameDecision(
            False, ratio, p_ceiling, r_ceiling,
            f"over-detects {ratio:.2f}x, precision capped at {p_ceiling:.2f}")
    if ratio < quality:
        return FrameDecision(
            False, ratio, p_ceiling, r_ceiling,
            f"under-detects {ratio:.2f}x, recall capped at {r_ceiling:.2f}")
    return FrameDecision(True, ratio, p_ceiling, r_ceiling, "within band")
