import os
import random
import torch
import cv2
from noise_generator import apply_noise
from utils_new import set_random_seed

from logging import exception
import errno
from tqdm import tqdm
import numpy as np
from imageio import imread
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import os
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
from utils import saveImage
from dataset import VQADataset
from generator import Generator

# Set the random seed for reproducibility
SEED = 42
set_random_seed(SEED)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


CLEAN_IMAGES_FOLDER = "1_correct_validation_images"
NOISY_IMAGES_FOLDER = "Noisy DARE TEST"


name = "val"
annotationsJSON = "annotations/filtered_answers.json"
questionsJSON = "questions/filtered_questions.json"
imagePrefix = None


# Assuming the dataset and logger are already initialized
logger = None  # Replace this with actual logger initialization
dataset = VQADataset(name, questionsJSON, annotationsJSON, CLEAN_IMAGES_FOLDER, imagePrefix, logger)

generator = Generator(dataset, logger)

NOISE_TYPES = [
    "Shot-noise",
    "Gaussian-noise",
    "Brightness",
    "Speckle-noise",
    "Contrast",
    "Snow",
    "Defocus-blur",
    "Pixelate",
    "Spatter",
    "Elastic",
    "Impulse-noise",
    "Saturation",
    "Zoom-Blur",
    "JPEG-compression",
    "Fog",
    "Frost",
    "Rain",
    "Motion-blur"
]


# Create folders for noisy images
for noise in NOISE_TYPES:
    os.makedirs(os.path.join(NOISY_IMAGES_FOLDER, noise), exist_ok=True)


# Process clean images
clean_images = os.listdir(CLEAN_IMAGES_FOLDER)

# Apply random noise with random severity to each image
for image_file in tqdm(clean_images, desc="Processing images"):
    image_path = os.path.join(CLEAN_IMAGES_FOLDER, image_file)
    
    # For each noise type, choose a random severity and apply noise
    for noise_type in NOISE_TYPES:
        severity_level = random.randint(1, 5)  # Random severity between 1 and 5
        print(f'Noise type: {noise_type} Severity level: {severity_level}')
        
        noisy_image = apply_noise(dataset,image_path, noise_type, severity_level, image_file)
        
        # Use the saveImage function to save the noisy image
        saveImage(noisy_image, os.path.join(NOISY_IMAGES_FOLDER, noise_type), image_file)
        
