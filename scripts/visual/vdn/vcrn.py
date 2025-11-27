import torch
from torchvision import models, transforms
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from ..common.utils import Logger 

# Initialize logger
logger = Logger(logPath='Logger')


# Load ResNet50 model
resnet50 = models.resnet50(pretrained=True)
num_classes = 18
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"noise_classifier.py device: {device}")

state_dict = torch.load('Resnet50_New.pt', weights_only=True)
resnet50.load_state_dict(state_dict)
resnet50.to(device)
logger.info('Resnet50 loaded')
resnet50.eval()

noise_classes = [
    "Brightness", "Contrast", "Defocus-blur", "Elastic", "Fog",
    "Frost", "Gaussian-noise", "Impulse-noise", "JPEG-compression", "Motion-blur",
    "Pixelate", "Rain", "Saturation", "Shot-noise", "Snow", "Spatter",
    "Speckle-noise", "Zoom-Blur"
]

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])