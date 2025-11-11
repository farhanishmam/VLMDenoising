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
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set up models:**
    - **VDN:** The VDN modules (ResNet-50 router, DRUNet denoisers) are trained as part of the framework.
    - **TDN:** Requires setting up API access for the frozen LLM (e.g., Gemini 2.0 Flash).

## Datasets

Our experiments use Visual Question Answering (VQA) as the core evaluation task.

- **Source Distribution (for VDN Training):** We use **VQAv2**. We create an augmented dataset of **270,000 images** (3,000 base images $\times$ 18 visual corruptions $\times$ 5 severity levels) to train the VDN modules.
- **Target Distribution (for Evaluation):** We use the **DARE** dataset for evaluation, as it provides a variety of VQA sub-tasks and a different visual distribution from VQAv2.
