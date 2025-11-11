# Scripts

Utilities for generating noisy/augmented data for VLM denoising experiments.

## Structure

### `text/`

Text augmentation and adversarial perturbations using nlpaug library.

**Key Functions:**

- Character-level: OCR errors, keyboard errors, insertions, deletions, swaps
- Word-level: synonym replacement, misspelling, abbreviation, word swapping
- Stylistic: leet speak, homoglyphs, case randomization, character repetition
- Semantic: back-translation, paraphrasing

**Usage:**

```python
# Applies 18 perturbation methods to each question in the dataset
python "text/augmentation/Adversarial Attack Using Libraries.py"
```

### `vision/`

Image corruption generation for VQA datasets.

#### `vision/augmentation/inference/`

Apply individual noise types to images with random severity levels.

**Main Script:** `noisy_main.py`

**Noise Types (18):**
Shot, Gaussian, Brightness, Speckle, Contrast, Snow, Defocus-blur, Pixelate, Spatter, Elastic, Impulse, Saturation, Zoom-blur, JPEG-compression, Fog, Frost, Rain, Motion-blur

#### `vision/augmentation/training/`

Batch generation of noisy VQA datasets with structured logging and reporting.

**Main Script:** `main.py`

**Components:**

- `generator.py` - Noise transformation pipeline
- `dataset.py` - VQA dataset loader
- `report.py` - Experiment reporting
- `utils.py` - Logging and file operations

**Requirements:** See `vision/augmentation/training/requirements.txt`

## Notes

- All scripts use `SEED=42` for reproducibility
- Vision scripts expect VQA2.0-style JSON annotations
- Text scripts output augmented question pairs with function labels
