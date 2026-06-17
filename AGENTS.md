# Project Overview
Recovery pipeline for 340,000+ bird annotations baked into corrupted aerial survey screenshots (Deepwater Horizon 2010). Extracts colored dot positions and maps them to original high-resolution images for DeepForest training data.

## Setup
```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Folder Structure

- src/ — One module per pipeline stage
- tests/ — One test file per src module, mirrors src/ structure
- data/fixtures/ — Sample images for integration tests (<1MB each)
- config.yaml — All thresholds and constants
- notebook/ — Prototype reference (Google Colab)

## Tech Stack
Python 3.10+, OpenCV, NumPy, PyYAML, pytest, scikit-image

Pipeline Stages
```text
decompose.py → detect.py → validate.py →
map_coords.py → species.py → export.py → pipeline.py
```

Configuration
All numeric thresholds live in config.yaml under their module key. Never hardcode values in src/.
```YAML

# Always do this:
_CONFIG["grey_low"]  # ✅

# Never do this:
GREY_LOW = 160       # ❌
```
## Coding Standards
- Descriptive names: has_dialog not FORMAT_A
- Type hints on all public functions and return values
- Docstrings on all public methods
- Vectorize with NumPy/OpenCV — avoid Python loops
- No comparative comments in code:

```Python
# ❌ Never:
# faster than the prototype loop approach
# measured from 25 study images, not guessed

# ✅ Always:
# Vectorized via sliding_window_view (NumPy >= 1.20)
```
## Testing Rules
Tests cover pipeline behaviour, not library internals.

```Python

# ✅ Test this - pipeline behaviour:
def test_no_pixels_lost(result_with_dialog):
    total = (
        result_with_dialog.aerial_width()
        + result_with_dialog.dialog.shape[1]
    )
    assert total == 900

# ❌ Not this - library internals:
def test_output_dtype_float32():
    assert _grey_profile(img).dtype == np.float32
```
## Commands
```Bash

# Run all tests
pytest tests/ -v

# Single module
pytest tests/test_decompose.py -v

# With short traceback
pytest tests/ --tb=short
```
## Known Gotchas
- Use float32 not bfloat16 — bfloat16 silently corrupts DeepForest training
- Use scale_y for both axes in coordinate mapping — scale_x is wrong due to aerial subregion cropping
- uint8 subtraction overflows — always cast to int16 first
- Text filter disabled in detect.py — colony-row birds match text-alignment heuristics and get wrongly removed
- Red HSV wraps around 180° — always use two ranges [0,20] and [160,180]
- Vegetation boost applies to green channel only — do not boost all colors
- detect.py expects RGB input — BGR input silently produces wrong detections

## Boundaries
✅ Always: Load config from config.yaml, one test file per module
⚠️ Ask first: New dependencies in requirements.txt
🚫 Never: Hardcode thresholds in src/, test NumPy/OpenCV internals, use bfloat16 anywhere in pipeline