# Dataset Preparation & Augmented Dataset Generation

## Dataset Download Instructions

### 1. DARE Dataset (1_correct – Validation Split)

Download the **1_correct subset – validation split images** from:

- Hugging Face: https://huggingface.co/datasets/cambridgeltl/DARE
- Google Drive (direct images): https://drive.google.com/drive/folders/1n32Cu6d2hEFt-ZSJgorlm7X2t4juQwSe?usp=sharing

### 2. VQA-v2 Dataset

Download the VQA-v2 dataset from:

- https://visualqa.org/

From the full VQAv2 dataset, select a subset of **3000 images** for experimentation.

---

## Directory Organization

Organize the directories as follows:

```
project_root/
├── CLEAN_IMAGES_FOLDER/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
└── NOISY_IMAGES_FOLDER/
    └── (augmented images will be saved here)
```

### Folder Description

- **CLEAN_IMAGES_FOLDER** — Contains clean input images. Either:
  - VQAv2 subset (3000 images), or
  - DARE dataset (657 validation images)

- **NOISY_IMAGES_FOLDER** — Output directory where augmented (corrupted) images will be generated, organized by noise type.

---

## Augmented Dataset Generation Procedure

### 1. DARE Image Augmentation (Inference Dataset)

To generate augmented images for the DARE dataset:

1. Open `noisy_main.py`
2. Set `CLEAN_IMAGES_FOLDER` and `NOISY_IMAGES_FOLDER` to your actual paths
3. Run the script as-is (no other modifications needed)

```bash
cd scripts/visual/noise_addition
python noisy_main.py
```

### 2. VQAv2 Image Augmentation (Training Dataset)

To generate augmented images for the VQAv2 dataset:

1. Open `noisy_main.py`
2. Uncomment lines 51, 52, and 53
3. Comment out everything after those lines
4. Set `CLEAN_IMAGES_FOLDER` and `NOISY_IMAGES_FOLDER` to your actual paths

```bash
cd scripts/visual/noise_addition
python noisy_main.py
```

---

## Output Structure

After running, `NOISY_IMAGES_FOLDER` will be organized by noise type:

```
NOISY_IMAGES_FOLDER/
├── Gaussian-noise/
├── Shot-noise/
├── Brightness/
├── Contrast/
├── Snow/
├── Fog/
├── Frost/
├── Rain/
├── Spatter/
├── Defocus-blur/
├── Motion-blur/
├── Zoom-Blur/
├── Elastic/
├── Pixelate/
├── JPEG-compression/
├── Impulse-noise/
├── Speckle-noise/
└── Saturation/
```

Each subfolder contains corrupted versions of all input images at a randomly chosen severity level (1–5).
