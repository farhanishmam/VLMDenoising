# Image Augmentation Utilities

This directory contains utilities for applying visual corruptions to images for the VLMDenoising project.

## Structure

```
visual/
├── noise_addition/      # Core utilities for corruption generation
│   ├── __init__.py
│   ├── dataset.py       # VQADataset class for loading VQA data
│   ├── generator.py     # Generator class with 18 corruption types × 5 severity levels
│   ├── utils.py         # Image I/O utilities and Logger class
│   ├── utils_new.py     # Random seed utilities
│   ├── noise_generator.py  # apply_noise() entry point
│   └── noisy_main.py    # Script for applying random corruptions to images
├── training/            # Scripts for generating training datasets
│   ├── main.py          # Main script for batch corruption generation
│   └── report.py        # Reporting utilities
└── vdn/                 # Visual Denoising Network pipeline
    ├── pipeline.py      # End-to-end denoising pipeline (top-K weighted avg)
    ├── csvd.py          # Corruption-Specific Visual Denoisers
    └── vcrn.py          # Visual Corruption Routing Network (ResNet50 classifier)
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
from scripts.visual.noise_addition.dataset import VQADataset
from scripts.visual.noise_addition.generator import Generator
from scripts.visual.noise_addition.utils import Logger

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

Run `noisy_main.py` directly from the `noise_addition/` directory:

```bash
cd scripts/visual/noise_addition
python noisy_main.py
```

Or use `apply_noise()` programmatically:

```python
from noise_generator import apply_noise
from dataset import VQADataset

# Apply a specific corruption
corrupted_image = apply_noise(
    dataset,
    image_path="path/to/image.jpg",
    noise_type="Gaussian-noise",
    severity=3,
    imageName="image.jpg"
)
```

## Dependencies

Required packages (see `requirements.txt`):
- numpy
- opencv-python
- scikit-image
- scipy
- imageio
- Pillow
- Wand (ImageMagick bindings)
- imgaug
- tqdm
