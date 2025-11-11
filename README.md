<div align="center">

# Enhancing Vision Language Corruption Robustness using Cross-Distribution & Prompted Denoisers

</div>

<p align="center">
  <strong>Sameer Shafayet Latif*</strong>
    ·
    <strong>Sadab Shiper*</strong>
    ·
    <strong>K. M. Rahiduzzaman Kiran*</strong>
    ·
     <strong>Md Farhan Ishmam*</strong>
    ·
    <strong>Md Azam Hossain</strong>
    ·
    <strong>Abu Raihan Mostofa Kamal</strong>
    ·
    <strong>Md Hamjajul Ashmafee</strong>
</p>
<p align="center"><sup>*</sup>Equal Contribution</p>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat)](https://github.com/farhanishmam/VLMDenoising)
[![Code](https://img.shields.io/badge/Code-farhanishmam/VLMDenoising-blue?logo=GitHub)](https://github.com/farhanishmam/VLMDenoising)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

</div>

---

<p style="text-align: justify;">
While Vision Language Models (VLMs) excel in ideal conditions, their performance drops significantly when exposed to realistic multimodal corruptions like blurry images and grammatically incorrect text. This work introduces a novel framework to enhance VLM robustness by using plug-and-play denoisers. We propose: (i) **cross-distribution visual denoisers (VDN)** inspired by the Mixture of Experts (MoE) architecture, and (ii) a **prompted zero-shot textual denoiser (TDN)** using a frozen LLM. We also establish a new benchmark, **VLSRB**, featuring 18 visual and 18 textual corruption functions to evaluate system robustness. Our approach demonstrates an overall accuracy gain of up to 5.5%.
</p>

## Overview of the Denoising Framework

![Overview of the denoising framework showing how corruptions cause VLM failure and how VDN and TDN modules clean the input to produce a correct answer.](./assets/overview.png)

## Methodology

Our framework enhances robustness by using two separate plug-and-play modules for each modality.

### 1. Visual DeNoiser (VDN)

The VDN architecture is inspired by sparse expert networks (MoE). It is composed of two main modules:

- **Visual Corruption Routing Network (VCRN):** A router (e.g., ResNet-50) that classifies the corruption type of a given image.
- **Corruption-Specific Visual Denoiser (CSVD):** A set of "expert" denoisers (e.g., DRUNet), each trained to reconstruct images for a _specific_ corruption class (like 'Rain' or 'Elastic' noise).

The VCRN routes the corrupted image to the appropriate CSVD expert(s) to produce the final denoised image.

### 2. Textual DeNoiser (TDN)

The TDN is a generative language model that is prompted to denoise text in a zero-shot setting.

- **Model:** We use a frozen LLM (Gemini 2.0 Flash).
- [cite_start]**Prompting:** The model is given a specific prompt that instructs it to _only_ return the denoised version of the question and to _not_ answer it or alter its intent [cite: 166-173].

This simple, training-free approach is highly effective at correcting textual corruptions like typos, word swaps, and grammatical errors.

## Vision Language System Robustness Benchmark (VLSRB)

We establish VLSRB, a new benchmark to evaluate the _system robustness_ of VLMs (i.e., the end-to-end performance including pre-processing). The benchmark includes a rich suite of 36 multimodal corruption effects:

- **Visual Corruptions (18 types):** Expands on existing benchmarks by adding new corruptions. These are categorized into 6 classes: Additive Noise, Digital, Image Attribute Transformation, Weather, Blur, and Physical.
- **Textual Corruptions (18 types):** Replicates realistic character, word, and sentence-level perturbations, including typos, synonym replacement, OCR errors, and paraphrasing.

## Quick Start & Installation

The repository is designed to be run on GPUs (experiments conducted on NVIDIA RTX 3090 24GB).

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/farhanishmam/VLMDenoising.git](https://github.com/farhanishmam/VLMDenoising.git)
    cd VLMDenoising
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    # We recommend using Python 3.8+
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure API keys and paths:**

    ```bash
    # Copy the template and add your API keys
    cp config.template.json config.json
    # Edit config.json with your actual API keys and paths
    ```

    **Important:** Never commit `config.json` with your actual API keys. The file is already in `.gitignore`.

4.  **Set up models:**
    - **VDN:** The VDN modules (ResNet-50 router, DRUNet denoisers) are trained using the generalized training script (see below).
    - **TDN:** Requires API access for the frozen LLM (e.g., Gemini 2.0 Flash).

## Datasets

Our experiments use Visual Question Answering (VQA) as the core evaluation task.

- **Source Distribution (for VDN Training):** We use **VQAv2**. We create an augmented dataset of **270,000 images** (3,000 base images $\times$ 18 visual corruptions $\times$ 5 severity levels) to train the VDN modules.
- **Target Distribution (for Evaluation):** We use the **DARE** dataset for evaluation, as it provides a variety of VQA sub-tasks and a different visual distribution from VQAv2.

## Usage Guide

### Training the Visual Corruption Routing Network (VCRN)

The VCRN is a ResNet50-based classifier that routes corrupted images to the appropriate corruption-specific denoiser. Train it using:

```bash
python scripts/vcrn_training.py \
    --data_dir dataset/ \
    --batch_size 30 \
    --num_epochs 40 \
    --learning_rate 0.001
```

**Parameters:**

- `--data_dir`: Dataset directory with `train/`, `val/`, `test/` subdirectories (each containing 18 corruption type folders)
- `--output_model`: Output model filename (default: `corruption_classifier_resnet50.pt`)
- `--batch_size`: Training batch size (default: 30)
- `--num_epochs`: Number of epochs (default: 40)
- `--learning_rate`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)
- `--num_classes`: Number of corruption types (default: 18)
- `--eval_only`: Only evaluate existing model without training

**Expected Dataset Structure:**

```
dataset/
├── train/
│   ├── Brightness/
│   ├── Contrast/
│   ├── Gaussian-noise/
│   └── ... (18 corruption types)
├── val/
│   └── ... (same 18 corruption types)
└── test/
    └── ... (same 18 corruption types)
```

**Output:**

- Trained classifier: `corruption_classifier_resnet50.pt`
- Per-class and overall accuracy metrics

### Training Visual Denoisers (VDN)

The repository provides a **generalized training script** that replaces 54+ redundant notebooks. Train any denoiser (BRDNet, DnCNN, DRUNet) on any corruption type:

```bash
python scripts/csvd_training.py \
    --model BRDNet \
    --corruption Brightness \
    --clean_dir data/clean_images/ \
    --noisy_dir data/noisy_images/Brightness/ \
    --batch_size 30 \
    --num_epochs 50 \
    --learning_rate 0.0001
```

**Parameters:**

- `--model`: Choose from `BRDNet`, `DnCNN`, or `DRUNet`
- `--corruption`: Corruption type (e.g., `Brightness`, `Gaussian`, `Motion-blur`, etc.)
- `--clean_dir`: Directory with clean images
- `--noisy_dir`: Directory with noisy images (must contain L1-L5 subdirectories)
- `--batch_size`: Training batch size (default: 30)
- `--num_epochs`: Number of epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--early_stop_patience`: Early stopping patience (default: 10)

**Output:**

- Trained model: `{corruption}_{model}.pt`
- Loss plot: `{corruption}_{model}_loss.png`
- Test metrics: PSNR, SSIM, and loss printed to console

**Example - Train all BRDNet denoisers:**

```bash
for corruption in Brightness Contrast Defocus-blur Elastic Fog Frost Gaussian \
                  Impulse JPEG-compression Motion-blur Pixelate Rain \
                  Saturation Shot Snow Spatter Speckle Zoom-Blur; do
    python scripts/csvd_training.py \
        --model BRDNet \
        --corruption $corruption \
        --clean_dir data/clean/ \
        --noisy_dir data/noisy/$corruption/
done
```

### Running VLM Inference

The repository provides a **unified inference script** that handles all VLM models and image/text configurations:

```bash
python scripts/vlm_inference.py \
    --model gemini \
    --api_key YOUR_GEMINI_API_KEY \
    --data_path data/raw/Noisy-Denoised_QuestionPairs[new].csv \
    --image_dir data/images/ \
    --category count \
    --image_type clean \
    --text_type noisy
```

**Parameters:**

- `--model`: VLM model (`gemini`, `idefics2`, `instructblip`, `llava`, `janus`)
- `--api_key`: API key for cloud models (required for Gemini)
- `--data_path`: Path to CSV with questions and answers
- `--image_dir`: Base directory for images
- `--category`: Question category (`count`, `order`, `trick`, `vcr`, or `all`)
- `--image_type`: Image input type (`clean`, `noisy`, or `denoised`)
- `--text_type`: Text input type (`clean`, `noisy`, or `denoised`)
- `--checkpoint`: Path to checkpoint file (auto-created if not specified)
- `--checkpoint_freq`: Save checkpoint every N predictions (default: 50)

**Configurations:**

The script supports **9 different configurations** by combining image and text types:

| Image Type | Text Type | Description                     |
| ---------- | --------- | ------------------------------- |
| Clean      | Clean     | Baseline (no corruption)        |
| Clean      | Noisy     | Text-only corruption            |
| Clean      | Denoised  | Text denoising only             |
| Noisy      | Clean     | Image-only corruption           |
| Noisy      | Noisy     | Both modalities corrupted       |
| Noisy      | Denoised  | Corrupted image + denoised text |
| Denoised   | Clean     | Image denoising only            |
| Denoised   | Noisy     | Denoised image + corrupted text |
| Denoised   | Denoised  | Both modalities denoised        |

**Examples:**

```bash
# Baseline: clean image + clean text
python scripts/vlm_inference.py \
    --model gemini --api_key $API_KEY \
    --data_path data.csv --image_dir images/ \
    --category count --image_type clean --text_type clean

# Test visual denoiser: noisy image + clean text
python scripts/vlm_inference.py \
    --model gemini --api_key $API_KEY \
    --data_path data.csv --image_dir images/ \
    --category count --image_type noisy --text_type clean

# Test both denoisers: denoised image + denoised text
python scripts/vlm_inference.py \
    --model gemini --api_key $API_KEY \
    --data_path data.csv --image_dir images/ \
    --category all --image_type denoised --text_type denoised
```

**Checkpointing:**

The script automatically saves progress to checkpoint files and can resume from interruptions:

```bash
# Resume from previous run
python scripts/vlm_inference.py \
    --model gemini --api_key $API_KEY \
    --data_path data.csv \
    --checkpoint checkpoint_gemini_count_clean_noisy.json \
    --image_type clean --text_type noisy
```

## Repository Structure

```
VLMDenoising/
├── scripts/
│   ├── vcrn_training.py                      # VCRN training (corruption classifier)
│   ├── csvd_training.py                      # CSVD training (denoisers)
│   ├── vlm_inference.py                      # VLM inference script
│   ├── textual_corruptions.py                # Textual corruption functions
│   └── visual/                               # Visual corruption utilities
│       ├── common/                           # Shared utilities (consolidated)
│       │   ├── dataset.py                   # VQA dataset loader
│       │   ├── generator.py                 # 18 corruption types × 5 levels
│       │   └── utils.py                     # Image I/O and logging
│       ├── training/                        # Training dataset generation
│       └── inference/                       # Inference-time corruptions
├── data/
│   └── raw/                                  # Raw datasets
├── config.template.json                      # Configuration template
└── README.md
```

**Note:** All 193+ redundant notebooks have been removed and replaced with 3 generalized, production-ready Python scripts. Additionally, over 1,100 lines of duplicate code in the augmentation utilities have been consolidated into a shared `common/` directory. The codebase is now significantly cleaner and more maintainable.
