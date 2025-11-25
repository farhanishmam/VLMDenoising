import cv2
from generator import Generator
from dataset import VQADataset
from utils import loadImage


name = "val"
annotationsJSON = "annotations/filtered_answers.json"
questionsJSON = "questions/filtered_questions.json"

CLEAN_IMAGES_FOLDER = "DARE Dataset/DARE Main Dataset/1_correct_validation_images"
NOISY_IMAGES_FOLDER = "Noisy DARE TEST"


logger = None  # Replace this with actual logger initialization
dataset = VQADataset(name, questionsJSON, annotationsJSON, CLEAN_IMAGES_FOLDER, None, logger)  # Replace this with actual dataset initialization
generator = Generator(dataset, logger)


# -----------------------------
# APPLY NOISE FUNCTION
# -----------------------------
def apply_noise(dataset,image_path, noise_type, severity, imageName):

    # Retrieve image ID → index in dataset
    image_id = dataset.get_image_id_from_path(image_path)
    try:
        idx = dataset.imageIds.index(image_id)
    except ValueError:
        raise ValueError(f"Image ID {image_id} not found in dataset.")
    
    # Load clean image
    image = loadImage(CLEAN_IMAGES_FOLDER, imageName)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    # Convert to RGB normalized
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0 
    
    # Apply the correct noise transformation based on noise type and severity
    if noise_type == "Shot-noise":
        noisy_image = generator.transformToShotNoise(idx, severity)
    elif noise_type == "Gaussian-noise":
        noisy_image = generator.transformToGaussianNoise(idx, severity)
    elif noise_type == "Brightness":
        noisy_image = generator.transformToBrightness(idx, severity)
    elif noise_type == "Speckle-noise":
        noisy_image = generator.transformToSpeckleNoise(idx, severity)
    elif noise_type == "Contrast":
        noisy_image = generator.transformToContrast(idx, severity)
    elif noise_type == "Snow":
        noisy_image = generator.transformToSnow(idx, severity)
    elif noise_type == "Defocus-blur":
        noisy_image = generator.transformToDefocusBlur(idx, severity)
    elif noise_type == "Pixelate":
        noisy_image = generator.transformToPixelate(idx, severity)
    elif noise_type == "Spatter":
        noisy_image = generator.transformToSpatter(idx, severity)
    elif noise_type == "Elastic":
        noisy_image = generator.transformToElastic(idx, severity)
    elif noise_type == "Impulse-noise":
        noisy_image = generator.transformToImpulseNoise(idx, severity)
    elif noise_type == "Saturation":
        noisy_image = generator.transformToSaturate(idx, severity)
    elif noise_type == "Zoom-Blur":
        noisy_image = generator.transformToZoomBlur(idx, severity)
    elif noise_type == "JPEG-compression":
        noisy_image = generator.transformToJpegCompression(idx, severity)
    elif noise_type == "Frost":
        noisy_image = generator.transformToFrost(idx, severity)
    elif noise_type == "Rain":
        noisy_image = generator.transformToRain(idx, severity)
    elif noise_type == "Fog":
        noisy_image = generator.transformToFog(idx, severity)
    elif noise_type == "Motion-blur":
        noisy_image = generator.transformToMotionBlur(idx, severity)

    return noisy_image
