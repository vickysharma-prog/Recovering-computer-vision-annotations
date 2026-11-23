"""
Read the photo number from a screenshot's title bar, to catch screenshots filed
against the wrong original.

On 2026-11-23, 22 of 1,119 frames were found to show a different photo from the one
their file name and paired original say: `17May11Camera1-Card3-0043` shows
`17May2011Camera1Card2 543.JPG`. Alignment still reports success on such pairs when
the neighbouring photo overlaps, so the title is the cheapest independent check.

OCR slips both ways on these title bars, so every plausible reading is kept and a
frame is a mismatch only when none of them equals the file number. Validated on 27
fresh frames: 26 match, 1 real mismatch, 0 false alarms.
"""
import re

import cv2
NUM = re.compile(r"(\d{2,6})\s*[\.\-,]?\s*(?:jp|jfp|ipg|pg|tif|tf|png)|(\d{3,6})\s*\(\s*1", re.I)
CARD = re.compile(r"(?:card|camera|cam)\W{0,2}\d\W{0,3}(\d{2,6})", re.I)
def nums(t):
    """Every reading of the photo number. OCR slips both ways, so all are kept:
    '1.108' needs the dot dropped, '2.076' needs it kept, 'card3157' needs the card
    digit split off. A frame is a mismatch only if NO reading equals its file number."""
    joined = re.sub(r"(?<=\d)[\.,](?=\d)", "", t)
    out = [int(a or b) for x in (t, joined) for a, b in NUM.findall(x)]
    out += [int(m) for x in (t, joined) for m in CARD.findall(x)]
    return out
def variants(img):
    h, w = img.shape[:2]
    s = img[:max(28, int(h * 0.035)), :int(w * .6)]
    yield cv2.resize(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    s = img[2:max(24, int(h * 0.026)), :int(w * .6)]
    for ch in (cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), s.max(axis=2), s.min(axis=2)):
        g = cv2.resize(ch, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield 255 - b if (b == 0).mean() > 0.5 else b
def title_number(img):
    """(numbers found, raw text) from the first variant that yields a number, else ([], '').

    Returns ([], '') when Tesseract is unavailable, so callers treat it as unreadable.
    """
    from src.legend import _ocr_engine
    pytesseract = _ocr_engine()
    if pytesseract is None:
        return [], ""
    for v in variants(img):
        txt = pytesseract.image_to_string(v, config="--psm 7").strip()
        n = nums(txt)
        if n: return n, txt
    return [], ""


def file_number(frame: str) -> int | None:
    """The trailing photo number of a frame name: `...Card3-0043` -> 43."""
    m = re.search(r"(\d+)$", frame)
    return int(m.group(1)) if m else None


def pairing_status(img, frame: str) -> str:
    """'match', 'mismatch' or 'unreadable'."""
    nums, _txt = title_number(img)
    if not nums:
        return "unreadable"
    return "match" if file_number(frame) in nums else "mismatch"


if __name__ == "__main__":
    assert 1108 in nums("28May201.2Cam1Card1. 1.108.JPG. (1/1)")
    assert 76 in nums("Camera. 2.076.JPG")
    assert 157 in nums("cameracard3157-PG (1)")
    assert 43 not in nums("Camera1Card2 543.JPG (1/1)")
    assert file_number("17May11Camera1-Card3-0043") == 43
    print("ok")
