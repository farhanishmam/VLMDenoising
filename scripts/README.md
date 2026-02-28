# Scripts

Utilities for generating noisy/augmented data for VLM denoising experiments.

## Structure

### `textual/`

Text augmentation and adversarial perturbations using the nlpaug library.

**Key Functions:**

- Character-level: OCR errors, keyboard errors, insertions, deletions, swaps
- Word-level: synonym replacement, misspelling, abbreviation, word swapping
- Stylistic: leet speak, homoglyphs, case randomization, character repetition
- Semantic: back-translation, paraphrasing

**Usage:**

```bash
cd textual
python download_dataset.py   # Step 1: download dataset
python noise_addition.py     # Step 2: apply 18 perturbation methods
python denoise_script.py     # Step 3: denoise with Gemini 2.0 Flash
```

### `visual/`

Image corruption generation for VQA datasets.

#### `visual/noise_addition/`

Apply individual noise types to images with random severity levels.

**Main Script:** `noisy_main.py`, `vqav2.py`

**Noise Types (18):**
Shot, Gaussian, Brightness, Speckle, Contrast, Snow, Defocus-blur, Pixelate, Spatter, Elastic, Impulse, Saturation, Zoom-Blur, JPEG-compression, Fog, Frost, Rain, Motion-blur

**Components:**

- `noise_addition/generator.py` - Noise transformation pipeline
- `noise_addition/dataset.py` - VQA dataset loader
- `training/report.py` - Experiment reporting
- `noise_addition/utils.py` - Logging and file operations

#### `visual/vdn/`

Visual Denoising Network inference pipeline.

**Components:**

- `pipeline.py` - End-to-end top-K weighted average denoising
- `vcrn.py` - Visual Corruption Routing Network (ResNet50 classifier)
- `csvd.py` - Corruption-Specific Visual Denoisers

## Notes

- All scripts use `SEED=42` for reproducibility
- Vision scripts expect VQA2.0-style JSON annotations
- Text scripts output augmented question pairs with function labels
