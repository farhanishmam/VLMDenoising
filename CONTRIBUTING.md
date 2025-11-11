# Contributing to VLMDenoising

Thank you for your interest in contributing to this research project!

## Project Structure

Please maintain the following organization:

```
VLMDenoising/
├── scripts/                # Python scripts and utilities
│   ├── vision/             # Vision-related modules
│   │   └── augmentation/{training,inference}/
│   └── text/               # Text-related modules
│       └── augmentation/
├── notebooks/              # Jupyter notebooks for experiments
│   ├── visual_denoiser_training/{BRDNet,DnCNN,DRUNet}/
│   ├── textual_denoising/
│   ├── vlm_inference/{Gemini,Idefics2,InstructBLIP,Janus,llava-7B}/
│   └── misc/
└── data/                   # Datasets
    └── raw/                # Raw input data
```

## Guidelines

### Code Organization

1. **Scripts** (`scripts/`)

   - Add Python scripts and reusable modules here
   - Use proper Python package structure with `__init__.py`
   - Import from `scripts` in notebooks: `from scripts.vision.augmentation import module`
   - Includes vision and text augmentation utilities

2. **Notebooks** (`notebooks/`)

   - Keep notebooks focused on specific experiments
   - Place in appropriate subdirectories by category
   - Document findings and results within notebooks

3. **Data** (`data/`)
   - Never commit large datasets to git
   - Place raw data in `data/raw/`
   - Keep raw data immutable

### Development Workflow

1. **Environment Setup**

   ```bash
   pip install -r requirements.txt
   ```

2. **Adding New Code**

   - Create/modify files in `scripts/`
   - Add appropriate `__init__.py` if creating new packages
   - Update `requirements.txt` if adding dependencies

3. **Running Experiments**

   - Create new notebook in appropriate `notebooks/` subdirectory
   - Import code from `scripts/`
   - Document results inline

4. **Adding Dependencies**
   - Add to `requirements.txt` with version constraints
   - Use format: `package>=min_version,<max_version`