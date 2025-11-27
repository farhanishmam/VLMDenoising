import torch
import os
from deepinv.models import DRUNet
from torchvision import transforms
from PIL import Image
import torchvision.transforms as T
from ..common.utils import Logger

logger = Logger(logPath='Logger')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

def load_model(Noise_type):
    model = DRUNet(in_channels=3, out_channels=3, pretrained='download', device='cuda') 
    model.load_state_dict(torch.load(f'<ENTER THE PATH OF THE .PT FILES OF THE CSVD>', weights_only=True))
    model.eval()
    return model

denoisers = {
    "Brightness": load_model("Brightness"),
    "Contrast": load_model("Contrast"),
    "Defocus-blur": load_model("DefocusBlur"),
    "Elastic": load_model("Elastic"),
    "Fog": load_model("Fog"),
    "Frost": load_model("Frost"),
    "Gaussian-noise": load_model("Gaussian"),
    "Impulse-noise": load_model("Impulse"),
    "JPEG-compression": load_model("JPEG"),
    "Motion-blur": load_model("MotionBlur"),
    "Pixelate": load_model("Pixelate"),
    "Rain": load_model("Rain"),
    "Saturation": load_model("Saturation"),
    "Shot-noise": load_model("Shot"),
    "Snow": load_model("Snow"),
    "Spatter": load_model("Spatter"),
    "Speckle-noise": load_model("Speckle"),
    "Zoom-Blur": load_model("ZoomBlur"),
}

denoiser_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

for denoiser in denoisers.values():
    denoiser.eval()

class_names = list(denoisers.keys())

logger.info("All denoisers loaded")