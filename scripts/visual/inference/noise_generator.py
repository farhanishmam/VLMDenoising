"""
Noise generator utility for applying various corruptions to images.
Uses the Generator class to apply specific noise types at specified severity levels.
"""

from ..common.generator import Generator
from ..common.dataset import VQADataset
from ..common.utils import loadImage


def apply_noise(generator, dataset, image_path, noise_type, severity, imageName):
    """
    Apply a specified noise type at a given severity level to an image.
    
    Args:
        generator: Generator instance with transformation methods
        dataset: Dataset instance containing image information
        image_path: Path to the image file
        noise_type: Type of noise/corruption to apply
        severity: Severity level (1-5)
        imageName: Name of the image file
        
    Returns:
        noisy_image: Corrupted image array
    """
    image_id = dataset.get_image_id_from_path(image_path)
    
    try:
        idx = dataset.imageIds.index(image_id)
    except ValueError:
        raise ValueError(f"Image ID {image_id} not found in dataset")
    
    # Map noise types to generator methods
    noise_map = {
        "Shot-noise": generator.transformToShotNoise,
        "Gaussian-noise": generator.transformToGaussianNoise,
        "Brightness": generator.transformToBrightness,
        "Speckle-noise": generator.transformToSpeckleNoise,
        "Contrast": generator.transformToContrast,
        "Snow": generator.transformToSnow,
        "Defocus-blur": generator.transformToDefocusBlur,
        "Pixelate": generator.transformToPixelate,
        "Spatter": generator.transformToSpatter,
        "Elastic": generator.transformToElastic,
        "Impulse-noise": generator.transformToImpulseNoise,
        "Saturation": generator.transformToSaturate,
        "Zoom-Blur": generator.transformToZoomBlur,
        "JPEG-compression": generator.transformToJpegCompression,
        "Frost": generator.transformToFrost,
        "Rain": generator.transformToRain,
        "Fog": generator.transformToFog,
        "Motion-blur": generator.transformToMotionBlur,
    }
    
    if noise_type not in noise_map:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    
    return noise_map[noise_type](idx, severity)
