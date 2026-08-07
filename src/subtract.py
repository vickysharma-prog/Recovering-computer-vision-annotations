"""
Isolate baked-in annotations by differencing a screenshot against its clean original.

The counting tool drew coloured markers directly onto a screenshot of an aerial
photograph. Because the un-annotated original still exists, the markers can be
recovered as "what is present here that was not in the photograph" instead of
"what falls inside a colour range". That matters: the symbology changes between
survey years, so any fixed colour threshold either overfits one year or floods
another with background vegetation, sand and sun-glint.

Three things make a naive `absdiff` useless here, and each is handled below:

1. **Sub-pixel misregistration.** The screenshot is the original downscaled ~4x
   and re-encoded, so even a well-fitted transform leaves fractional offsets.
   Every high-contrast edge — and an aerial full of birds on sand is nothing but
   high-contrast edges — then shows up as a difference. Handled by comparing each
   pixel against the best-matching nearby pixel rather than the exactly-aligned
   one: a genuine annotation differs from the original at EVERY nearby offset,
   while a shifted edge matches at SOME offset.
2. **Exposure and codec drift** between the render and the original. Handled by
   normalising per-channel statistics before differencing.
3. **The dialog and window chrome**, which have no counterpart at all and would
   otherwise register as one enormous annotation. Handled by masking regions the
   warp does not cover, plus large solid blocks of difference — the UI is a slab,
   annotations are small marks.

Usage:
    res = align(screenshot, original)
    if res.ok:
        sub = extract_annotations(screenshot, original, res)
        # sub.mask is the annotation ink
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.align import AlignResult, warp_to_screenshot

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        full = yaml.safe_load(f)
    return full.get("subtract", {})


_CONFIG = _load_config()

# Search radius for the shift-tolerant comparison, in screenshot pixels.
_SHIFT_RADIUS = _CONFIG.get("shift_radius", 2)
# Weight on Lab's L channel when differencing. Kept low so water/exposure
# differences (which are luminance) do not swamp annotation ink (which is
# chromatic) — but non-zero, so grey markers are not invisible.
_LUMINANCE_WEIGHT = _CONFIG.get("luminance_weight", 0.15)
# Threshold = median + k * MAD of the difference, floored by an absolute value so
# a near-empty frame cannot drive the threshold to zero.
_MAD_K = _CONFIG.get("mad_k", 8.0)
_MIN_DIFF = _CONFIG.get("min_diff", 14.0)          # Lab units
# UI slabs: a connected difference region larger than this fraction of the frame
# is chrome, not annotation.
_UI_AREA_FRAC = _CONFIG.get("ui_area_frac", 0.004)
# ...and its pixels must be duller than this to count as chrome. Measured median
# saturation: dialog 1, open water ~26, marker carpet 70-74.
_UI_MAX_SAT = _CONFIG.get("ui_max_sat", 40)
# Chrome test, applied to a region's INK pixels rather than to the whole region.
# `ui_ink_sat` is what counts as chromatic ink; a region is chrome when fewer than
# `ui_max_ink_sat_frac` of its ink pixels reach it. Measured over the 21 candidate
# regions on the six hand-labelled frames: regions containing real markers run
# 10.9%-94.9%, the dialogs run 0.0%-2.5%, and nothing falls between.
_UI_INK_SAT = _CONFIG.get("ui_ink_sat", 100)
_UI_MAX_INK_SAT_FRAC = _CONFIG.get("ui_max_ink_sat_frac", 0.05)
_UI_DILATE = _CONFIG.get("ui_dilate", 9)
_EDGE_MARGIN = _CONFIG.get("edge_margin", 2)       # shift wrap-around guard
# Minimum median saturation (HSV S, 0-255) of the screenshot under an ink blob for
# it to count as a marker. Annotation ink is chromatic by construction; the
# residual false ink on sparse frames — water glint, mudflat texture, the grey
# label panel — is not. Measured medians: marker carpet 70-74, a clean sparse
# frame's markers 169, versus background residual 7-40. A floor between them both
# removes that noise AND un-poisons the modal-size estimate, which otherwise
# collapses when noise blobs outvote the real markers.
_MARKER_SAT_MIN = _CONFIG.get("marker_sat_min", 50)


# ─────────────────────────────────────────────────────
# DATACLASS
# ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class SubtractResult:
    """Annotation ink recovered from a screenshot/original pair."""

    mask: np.ndarray        # uint8 0/255 — annotation pixels
    diff: np.ndarray        # float32 — shift-tolerant Lab distance
    valid: np.ndarray       # bool — pixels the original actually covers
    ui_mask: np.ndarray     # uint8 0/255 — chrome/dialog excluded from `mask`
    saturation: np.ndarray  # uint8 — screenshot HSV S, for the marker-chroma gate
    threshold: float

    @property
    def ink_frac(self) -> float:
        n = int(self.valid.sum())
        return float((self.mask > 0).sum()) / n if n else 0.0


# ─────────────────────────────────────────────────────
# DIFFERENCING
# ─────────────────────────────────────────────────────

def _match_exposure(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rescale b's per-channel mean/std onto a's, over covered pixels only."""
    out = b.copy()
    for c in range(a.shape[2]):
        av, bv = a[..., c][valid], b[..., c][valid]
        if bv.size == 0 or bv.std() < 1e-3:
            continue
        out[..., c] = (b[..., c] - bv.mean()) * (av.std() / bv.std()) + av.mean()
    return out


def _tolerant_diff(a: np.ndarray, b: np.ndarray,
                   radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each pixel of `a` to the closest nearby pixel of `b`.

    This is what makes the difference robust to the fractional misalignment left
    over after warping. Taking a minimum over a small search window means a real
    marker — which has no counterpart at any offset — keeps its large distance,
    while an edge that is merely shifted collapses to near zero.

    Returns `(chroma, full)`. Both are needed, and they serve opposite ends:

    * `chroma` down-weights L and drives ink detection, because water — ripples,
      glint — differs mainly in luminance and would otherwise swamp the markers.
    * `full` keeps L and drives UI detection, because the dialog is flat grey and
      over a sandy scene has almost no chromatic contrast; judged on chroma alone
      it goes unmasked, and everything inside it is then counted as markers.

    The shift that minimises the full distance is the one used for both, so this
    costs a single pass rather than two.
    """
    weights = np.array([_LUMINANCE_WEIGHT, 1.0, 1.0], np.float32)
    best_full = best_chroma = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            delta = a - np.roll(np.roll(b, dy, axis=0), dx, axis=1)
            full = np.linalg.norm(delta, axis=-1)
            if best_full is None:
                best_full = full
                best_chroma = np.linalg.norm(delta * weights, axis=-1)
                continue
            closer = full < best_full
            best_full = np.where(closer, full, best_full)
            best_chroma = np.where(
                closer, np.linalg.norm(delta * weights, axis=-1), best_chroma)
    return best_chroma.astype(np.float32), best_full.astype(np.float32)


def _ui_regions(binary: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    """Mask window chrome and the dialog — large *achromatic* blocks of difference.

    Deliberately derived from the difference itself rather than from the dialog
    locator: the dialog is precisely the region with no counterpart in the
    original, so the residual identifies it without depending on a separate
    detector that has its own failure modes.

    Size alone is NOT enough, and getting this wrong is catastrophic rather than
    merely inaccurate. In a dense colony the markers overlap into one connected
    carpet that can span an eighth of the frame — on a 2037-dot image that carpet
    was being classified as chrome and discarded, taking 92% of the ink with it.
    Bounding-box fill does not separate them either: it rescued that image but
    stopped masking dialogs elsewhere, and sparse frames (few real dots, so the
    dialog dominates) went from 0.98x to 9.85x.

    Saturation separates them, but it has to be read off the region's INK, not off
    the region. The earlier version took the median saturation over the whole
    closed component, and that measured the wrong thing: `MORPH_CLOSE` bridges a
    scattered colony into one region covering a quarter of the frame, whose median
    is then dominated by the **background between** the markers. On the six
    hand-labelled frames that deleted 92 of 345 real markers — 26.7%, and 53 of 71
    on `14June21…238` alone, where the offending region spanned 25% of the frame
    and its median saturation read 17 against a threshold of 40.

    Judged on ink pixels the two separate cleanly. Measured over the 21 candidate
    regions on those frames, as the fraction of ink at least `_UI_INK_SAT`
    saturated: regions holding real markers run 10.9% to 94.9%, the dialogs run
    0.0% to 2.5%, and nothing lands in between. A colony always has some chromatic
    ink however much background the closing drags in with it; a flat grey dialog
    has almost none.

    Size still gates first, and the size rule must not be loosened to compensate
    for anything: masking on size alone once discarded 92% of the ink on a
    2037-marker frame, and bounding-box fill rescued that image while sending
    sparse frames from 0.98x to 9.85x.
    """
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (_UI_DILATE, _UI_DILATE)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    frame_area = binary.shape[0] * binary.shape[1]
    ui = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < _UI_AREA_FRAC * frame_area:
            continue
        region = labels == i
        ink = region & (binary > 0)
        if not ink.any():
            continue
        if float((saturation[ink] >= _UI_INK_SAT).mean()) < _UI_MAX_INK_SAT_FRAC:
            ui[region] = 255
    if ui.any():
        ui = cv2.dilate(ui, np.ones((_UI_DILATE, _UI_DILATE), np.uint8))
    return ui


def extract_annotations(screenshot_rgb: np.ndarray, original_rgb: np.ndarray,
                        alignment: AlignResult) -> SubtractResult:
    """Recover annotation ink from an aligned screenshot/original pair."""
    if not alignment.ok:
        raise ValueError("cannot subtract with a rejected alignment")

    warped, covered = warp_to_screenshot(original_rgb, alignment,
                                         screenshot_rgb.shape)
    valid = covered > 0
    # np.roll wraps, so pixels within the search radius of a border can compare
    # against the opposite edge. Cheaper to drop that rim than to pad.
    if _EDGE_MARGIN:
        m = _EDGE_MARGIN + _SHIFT_RADIUS
        valid[:m, :] = valid[-m:, :] = valid[:, :m] = valid[:, -m:] = False

    saturation = cv2.cvtColor(screenshot_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]

    lab_shot = cv2.cvtColor(screenshot_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_orig = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_orig = _match_exposure(lab_shot, lab_orig, valid)

    # `diff` (chroma-weighted) finds ink; `diff_full` finds chrome. Water differs
    # mainly in luminance, so down-weighting L cut spurious ink from 8.58% to
    # 0.87% on a coastal frame while raising it on a dense colony (4.32% ->
    # 9.31%, against ~10.9% of pixels genuinely covered by its 2037 markers).
    diff, diff_full = _tolerant_diff(lab_shot, lab_orig, _SHIFT_RADIUS)
    diff[~valid] = 0.0
    diff_full[~valid] = 0.0

    # Robust threshold: the bulk of the frame is background whose residual is
    # small, so median + k*MAD sits above the noise without assuming any fixed
    # fraction of the image is ink (a percentile would force one through).
    vals = diff[valid]
    if vals.size == 0:
        empty = np.zeros(diff.shape, np.uint8)
        return SubtractResult(empty, diff, valid, empty, saturation, float("inf"))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) or 1.0
    threshold = max(med + _MAD_K * mad, _MIN_DIFF)

    binary = ((diff > threshold) & valid).astype(np.uint8) * 255

    # Chrome is located on the FULL difference, with its own threshold: a grey
    # dialog over a sandy scene barely registers chromatically, and if it goes
    # unmasked every glyph inside it is counted as a marker.
    full_vals = diff_full[valid]
    full_med = float(np.median(full_vals))
    full_mad = float(np.median(np.abs(full_vals - full_med))) or 1.0
    ui_binary = ((diff_full > max(full_med + _MAD_K * full_mad, _MIN_DIFF))
                 & valid).astype(np.uint8) * 255
    ui_mask = _ui_regions(ui_binary, saturation)
    mask = cv2.bitwise_and(binary, cv2.bitwise_not(ui_mask))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    return SubtractResult(mask, diff, valid, ui_mask, saturation, threshold)


# ─────────────────────────────────────────────────────
# INK -> DOT CANDIDATES
# ─────────────────────────────────────────────────────

def _marker_area(mask: np.ndarray, blobs: list) -> float:
    """Area of one marker, estimated so that merging cannot distort it.

    Blob areas are the obvious estimator and the wrong one: in a dense colony
    every marker is fused into a run, so the median blob is a whole run and the
    estimate scales with the crowding it is supposed to correct for. In the limit
    of a single merged blob the estimate equals that blob, no split is ever
    triggered, and the frame reports one dot.

    The distance transform does not have that problem. Its value at a marker's
    centre is that marker's radius whether or not the marker touches others, so
    the modal peak height gives a crowding-independent size.
    """
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    peaks = dist[dist >= np.maximum(cv2.dilate(dist, np.ones((5, 5), np.float32)) - 1e-6,
                                    1.0)]
    if peaks.size:
        radius = float(np.median(peaks))
        if radius >= 1.0:
            return max(np.pi * radius * radius, 4.0)
    return max(float(np.median([b[7] for b in blobs])), 4.0)


def dot_candidates(result: SubtractResult,
                   min_area: int = 3) -> list[tuple[float, float, int, int, int]]:
    """Blobs from the ink mask that plausibly are markers.

    Subtraction yields *all* annotation ink, which includes things that are not
    dots: the red site-label text and the transect lines drawn on many frames.
    Those are still genuine annotations — they simply are not what we count.

    Two cues separate them, applied in this order:

    1. **Elongation.** A transect line is hundreds of pixels long; no marker is.
    2. **Uniform size.** The counting tool draws every marker on an image at one
       size, so the surviving blobs cluster tightly around a modal area, while
       text glyphs vary. The mode is taken from the square-ish survivors, since
       that subset is dominated by markers.

    Overlapping markers in a dense colony merge into one blob, so anything much
    larger than the modal marker is split by distance transform rather than
    counted once — without this, dense colonies undercount badly.

    Colour is deliberately NOT used here. The stronger colour cue is the image's
    own legend palette, which lives in `classify` and is unavailable when the
    dialog fails to parse — so this stays usable on its own and callers can
    apply the palette on top when they have one.

    Returns `(cx, cy, w, h, area)` per candidate.
    """
    # Imported lazily: `classify` is a consumer of this module, so a top-level
    # import would close a cycle.
    from src.classify import _split_cluster

    n, labels, stats, cents = cv2.connectedComponentsWithStats(result.mask, 8)

    # Chroma gate, applied BEFORE size estimation — not merely to drop noise, but
    # to stop it corrupting the modal-marker size. On a sparse frame the false ink
    # (water glint, mudflat texture, the grey label panel) outnumbers the real
    # markers, so an ungated distance transform collapses the modal radius to its
    # floor, the size band widens, and both the direct-keep and the split branch
    # then flood with noise. Removing the low-saturation blobs first lets the size
    # estimate settle on the real markers. Colour *identity* still is not used —
    # only that annotation ink is chromatic at all, which needs no legend.
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT]); y0 = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        crop = labels[y0:y0 + h, x0:x0 + w] == i
        if float(np.median(result.saturation[y0:y0 + h, x0:x0 + w][crop])) < _MARKER_SAT_MIN:
            continue
        blobs.append((i, float(cents[i][0]), float(cents[i][1]), x0, y0, w, h, area))
    if not blobs:
        return []

    # Size estimate and splitting run on the gated ink only, so a discarded water
    # expanse can neither set the modal size nor be split into phantom markers.
    gated = np.isin(labels, [b[0] for b in blobs]).astype(np.uint8) * 255
    modal = _marker_area(gated, blobs)

    out: list[tuple[float, float, int, int, int]] = []
    for (i, cx, cy, x0, y0, w, h, area) in blobs:
        elongation = max(w, h) / max(min(w, h), 1)
        if area >= 1.5 * modal and elongation <= 4.0:
            n_est = max(1, int(round(area / modal)))
            sub = (labels[y0:y0 + h, x0:x0 + w] == i).astype(np.uint8)
            split = _split_cluster(sub, n_est)
            if len(split) > 1:
                # Each sub-dot carries its own footprint rather than the parent
                # cluster's bounding box. Inheriting `w, h` described the whole
                # run, so every later size or elongation test on a split piece was
                # reading the parent instead of the marker.
                #
                # Gating the split on the modal size band as well was tried and is
                # NOT here: it cost sparse recall 0.56 -> 0.10, because `modal` is
                # itself the unreliable quantity (it reads a stroke half-width on
                # outline glyphs), so a band built from it rejects valid splits on
                # exactly the frames where the estimate has least support.
                each = int(area / len(split))
                side = max(1, int(round(np.sqrt(max(each, 1)))))
                out.extend((x0 + lx, y0 + ly, side, side, each)
                           for (lx, ly) in split)
                continue
        if elongation <= 2.0 and 0.35 * modal <= area <= 3.0 * modal:
            out.append((cx, cy, w, h, area))
    return out
