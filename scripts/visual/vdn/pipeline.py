import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from vcrn import classifier_transform, resnet50
from ..common.utils import Logger
from csvd import denoisers, denoiser_transform, device, class_names
from PIL import Image
from pathlib import Path

logger = Logger(logPath='Logger')
logger.info("Logger initialized.")

input_folder = 'Noisy DARE TEST/'
output_folder = 'Weighted AVG Denoised [TopK]/'


noise_classes = [
    "Brightness", "Contrast", "Defocus-blur", "Elastic", "Fog",
    "Frost", "Gaussian-noise", "Impulse-noise", "JPEG-compression", "Motion-blur",
    "Pixelate", "Rain", "Saturation", "Shot-noise", "Snow", "Spatter",
    "Speckle-noise", "Zoom-Blur"
]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all subfolders in the input directory (representing different noise types)
for noise_type in os.listdir(input_folder):
    noise_folder_path = os.path.join(input_folder, noise_type)
    
    # Skip if the folder doesn't exist or if it's not a directory
    if not os.path.isdir(noise_folder_path):
        continue
    
    # Create corresponding output folder for each noise type (same name as in input)
    output_noise_folder = os.path.join(output_folder, noise_type)
    Path(output_noise_folder).mkdir(parents=True, exist_ok=True)
    
    # Loop through all images in the current noise folder
    for image_name in os.listdir(noise_folder_path):
        # Full image path
        image_path = os.path.join(noise_folder_path, image_name)
        
        if not os.path.isfile(image_path):
            continue
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_classifier = classifier_transform(image).unsqueeze(0).to(device)
        input_denoiser = denoiser_transform(image).unsqueeze(0).to(device)
        
        K = 3  # top-K denoisers to use

        # === Get confidence scores ===
        with torch.no_grad():
            logits = resnet50(input_classifier)
            confidence_scores = F.softmax(logits, dim=1)[0]  # shape: [18]

        # === Get top-K indices and normalize their scores ===
        topk_scores, topk_indices = torch.topk(confidence_scores, K)
        topk_scores_normalized = topk_scores / topk_scores.sum()  # shape: [K]

        # Print top-K confidence scores
        print("\nTop-K Confidence Scores:")
        for i in range(K):
            print(f"Class {topk_indices[i].item()}: {topk_scores_normalized[i].item():.10f}")

        # === Apply top-K denoisers and compute weighted output ===
        final_output = torch.zeros_like(input_denoiser.squeeze(0))  # shape: [3, H, W]
        with torch.no_grad():
            for i in range(K):
                class_idx = topk_indices[i].item()
                class_name = class_names[class_idx]
                denoised_img = denoisers[class_name](input_denoiser, 0.0)[0]  # shape: [1, 3, H, W]
                final_output += topk_scores_normalized[i] * denoised_img 

        # Clamp to valid image range
        final_output = final_output.clamp(0, 1)

        # Convert tensor back to PIL image
        final_image = transforms.ToPILImage()(final_output.cpu())

        # Save the denoised image to the output directory
        final_image.save(os.path.join(output_noise_folder, image_name))

print("Denoising and saving complete.")

logger.info("Processing complete.")