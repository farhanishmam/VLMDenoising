# Notebooks

This directory contains all Jupyter notebooks for experiments and analysis.

## Structure

```
notebooks/
├── misc/                           # Miscellaneous notebooks
├── textual_denoising/              # Textual denoising experiments
├── visual_denoiser_training/       # Visual denoiser training
│   ├── BRDNet/                     # BRDNet denoiser notebooks
│   ├── DnCNN/                      # DnCNN denoiser notebooks
│   └── DRUNet/                     # DRUNet denoiser notebooks
└── vlm_inference/                  # VLM inference experiments
    ├── Gemini/                     # Gemini model experiments
    ├── Idefics2/                   # Idefics2 model experiments
    ├── InstructBLIP/               # InstructBLIP model experiments
    ├── Janus/                      # Janus model experiments
    └── llava-7B/                   # LLaVA-7B model experiments
```

## Running Notebooks

1. Install dependencies: `pip install -r requirements.txt`
2. Start Jupyter: `jupyter notebook` or `jupyter lab`
3. Navigate to the desired notebook

## Naming Convention

- Notebooks are organized by model/experiment type
- Each denoiser has separate notebooks for different noise types
- VLM inference notebooks are organized by model and experiment configuration
