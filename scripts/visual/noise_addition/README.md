# Dataset Preparation & Augmented Dataset Generation

## 📥 Dataset Download Instructions

### 1️⃣ DARE Dataset (1_correct – Validation Split)

Download the **1_correct subset – validation split images** from:

- 🤗 Hugging Face: https://huggingface.co/datasets/cambridgeltl/DARE  
- 📂 Google Drive (direct images): https://drive.google.com/drive/folders/1n32Cu6d2hEFt-ZSJgorlm7X2t4juQwSe?usp=sharing  

---

### 2️⃣ VQA-v2 Dataset

Download the VQA-v2 dataset from:

- 🌐 https://visualqa.org/

From the full VQAv2 dataset:
- Select a subset of **3000 images** for experimentation.

---

## 📂 Directory Organization

Organize the directories as follows:
```
project_root/
│
├── CLEAN_IMAGES_FOLDER/
│ ├── image_1.jpg
│ ├── image_2.jpg
│ └── ...
│
├── NOISY_IMAGES_FOLDER/
│ ├── (augmented images will be saved here folder by folder (visual corruption wise))
│
├── noisy_main.py
```

### Folder Description

- **CLEAN_IMAGES_FOLDER**
  - Contains clean images.
  - Either:
    - VQAv2 subset (3000 images), or
    - DARE dataset (657 validation images)

- **NOISY_IMAGES_FOLDER**
  - Output directory.
  - Augmented (corrupted) images will be generated here folder by folder (Visual Corruption Wise).

---

# 🧪 Augmented Dataset Generation Procedure

## 🔹 1. DARE Image Augmentation (Inference Dataset)

To generate augmented images for the DARE dataset:

- Open `noisy_main.py`
- Adjust folder paths accordingly
- Run the script **without modifying any lines**

Then execute:

```bash
python noisy_main.py
```

## 🔹 1. VQAv2 Image Augmentation (Inference Dataset)

To generate augmented images for the DARE dataset:

- Open `vqav2.py`
- Adjust folder paths accordingly

Then execute:

```bash
python vqav2.py
```








