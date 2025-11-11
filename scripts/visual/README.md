# Image Augmentation Utilities

This directory contains utilities for applying visual corruptions to images for the VLMDenoising project.

## Structure

```
visual/
├── common/              # Shared utilities (consolidated from training & inference)
│   ├── __init__.py     # Package exports
│   ├── dataset.py      # VQADataset class for loading VQA data
│   ├── generator.py    # Generator class with 18 corruption types × 5 severity levels
│   └── utils.py        # Image I/O utilities and Logger class
├── training/           # Scripts for generating training datasets
│   ├── main.py        # Main script for batch corruption generation
│   └── report.py      # Reporting utilities
└── inference/          # Scripts for applying corruptions during inference
    ├── noisy_main.py  # Apply random corruptions to images
    └── noise_generator.py
```

## Available Corruptions

The `Generator` class supports 18 corruption types with 5 severity levels (L1-L5):

### Noise Corruptions
- **Gaussian-noise**: Additive Gaussian noise
- **Shot-noise**: Poisson (photon) noise
- **Impulse-noise**: Salt and pepper noise
- **Speckle-noise**: Multiplicative noise

### Blur Corruptions
- **Defocus-blur**: Out-of-focus blur
- **Motion-blur**: Camera motion blur
- **Zoom-Blur**: Radial blur effect

### Weather Corruptions
- **Snow**: Snow overlay effect
- **Fog**: Fog/mist effect
- **Frost**: Frost texture overlay
- **Rain**: Rain streaks
- **Spatter**: Water/mud splatter

### Attribute Transformations
- **Brightness**: Brightness adjustment
- **Contrast**: Contrast adjustment
- **Saturation**: Color saturation adjustment

### Digital Corruptions
- **Elastic**: Elastic deformation
- **Pixelate**: Pixelation/downsampling
- **JPEG-compression**: JPEG compression artifacts

## Usage

### Training: Generate Corrupted Dataset

```python
from scripts.visual.common import VQADataset, Generator, Logger

# Initialize
logger = Logger("logs/")
dataset = VQADataset(
    name="val",
    questionsJSON="path/to/questions.json",
    annotationsJSON="path/to/annotations.json",
    imageDirectory="path/to/images/",
    imagePrefix=None,
    logger=logger
)

# Create generator
generator = Generator(dataset, logger)

# Apply corruptions
transformations = ["Gaussian-noise_L3", "Brightness_L4", "Motion-blur_L2"]
generator.transform(transformations, outputPath="output/corrupted/")
```

### Inference: Apply Random Corruptions

```python
from scripts.visual.inference.noise_generator import apply_noise
from scripts.visual.common import VQADataset

# Apply random corruption
corrupted_image = apply_noise(
    dataset, 
    image_path="path/to/image.jpg",
    noise_type="Gaussian-noise",
    severity_level=3,
    image_filename="image.jpg"
)
```

## Consolidation

**Note:** Previously, `dataset.py`, `generator.py`, and `utils.py` were duplicated in both `training/` and `inference/` directories (1,121 lines × 2 = 2,242 lines of duplicate code). These have been consolidated into the `common/` directory, eliminating over 1,100 lines of redundancy.

## Dependencies

Required packages (see `training/requirements.txt`):
- numpy
- opencv-python
- scikit-image
- scipy
- imageio
- Pillow
- Wand (ImageMagick bindings)
- imgaug
- tqdm

