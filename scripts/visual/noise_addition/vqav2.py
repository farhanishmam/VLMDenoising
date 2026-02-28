import os
import random
import torch
import cv2
from noise_generator import apply_noise
from utils_new import set_random_seed

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

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CLEAN_IMAGES_FOLDER = "<VQAv2 3000 subset images folder>"
NOISY_IMAGES_FOLDER = "<Output Folder>"

name = "val"
annotationsJSON = "annotations/filtered_answers.json"
questionsJSON = "questions/filtered_questions.json"
imagePrefix = None

# Assuming the dataset and logger are already initialized
logger = None  # Replace this with actual logger initialization
dataset = VQADataset(name, questionsJSON, annotationsJSON, CLEAN_IMAGES_FOLDER, imagePrefix, logger)

generator = Generator(dataset, logger)
# transformationsList = ["Defocus-blur_L1"] ## test
transformationsList = list(generator.validTransformations.keys())
generator.transform(transformationsList, outputPath=NOISY_IMAGES_FOLDER)
