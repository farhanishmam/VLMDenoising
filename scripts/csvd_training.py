#!/usr/bin/env python3
"""
Corruption-Specific Visual Denoiser (CSVD) Training Script
Trains denoiser models (BRDNet, DnCNN, DRUNet) for specific corruption types.

Usage:
    python csvd_training.py --model BRDNet --corruption Gaussian-noise --clean_dir data/clean --noisy_dir data/noisy
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time
import matplotlib.pyplot as plt


# ==================== MODEL DEFINITIONS ====================

class BatchRenormalization(nn.Module):
    """Batch Renormalization Layer"""
    def __init__(self, num_features, eps=1e-3):
        super(BatchRenormalization, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps)

    def forward(self, x):
        return self.bn(x)


class BRDNet(nn.Module):
    """Batch Renormalization Denoising Network"""
    def __init__(self):
        super(BRDNet, self).__init__()
        
        # First branch layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.br1 = BatchRenormalization(64)
        self.relu = nn.ReLU()

        self.conv_blocks_x = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1), 
                BatchRenormalization(64),
                nn.ReLU()
            ) for _ in range(7)]
        )
        
        self.conv_blocks_y = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
                nn.ReLU()
            ) for _ in range(7)]
        )

        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        # Second branch layers
        self.conv_y_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.br_y_1 = BatchRenormalization(64)

        self.conv_blocks_y_extra = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
                nn.ReLU()
            ) for _ in range(6)]
        )

        self.conv_y_final = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.conv_concat = nn.Conv2d(6, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x_in = x
        x = self.relu(self.br1(self.conv1(x)))
        x = self.conv_blocks_x(x)
        x = self.conv2(x)
        x = x_in - x

        y_in = x_in
        y = self.relu(self.br_y_1(self.conv_y_1(y_in)))
        y = self.conv_blocks_y(y)
        y = self.conv_blocks_y_extra(y)
        y = self.conv_y_final(y)
        y = y_in - y

        out = torch.cat([x, y], dim=1)
        out = self.conv_concat(out)
        out = x_in - out
        return out


def get_dncnn_model():
    """Get DnCNN model from deepinv"""
    try:
        from deepinv.models import DnCNN
        return DnCNN(in_channels=3, out_channels=3, depth=20, pretrained=None, device='cuda')
    except ImportError:
        print("deepinv not installed. Install with: pip install deepinv")
        return None


def get_drunet_model():
    """Get DRUNet model from deepinv"""
    try:
        from deepinv.models import DRUNet
        return DRUNet(in_channels=3, out_channels=3, pretrained=None, device='cuda')
    except ImportError:
        print("deepinv not installed. Install with: pip install deepinv")
        return None


# ==================== DATASET ====================

class DenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_base_dir, transform=None):
        self.clean_dir = clean_dir
        self.transform = transform
        self.noisy_images = []
        
        for level in range(1, 6):
            folder = os.path.join(noisy_base_dir, f'L{level}')
            if os.path.exists(folder):
                self.noisy_images.extend([
                    (os.path.join(folder, img), os.path.join(clean_dir, img))
                    for img in sorted(os.listdir(folder))
                ])

    def __len__(self):
        return len(self.noisy_images)
    
    def __getitem__(self, idx):
        noisy_img = Image.open(self.noisy_images[idx][0]).convert('RGB')
        clean_img = Image.open(self.noisy_images[idx][1]).convert('RGB')
        if self.transform:
            noisy_img, clean_img = self.transform(noisy_img), self.transform(clean_img)
        return noisy_img, clean_img


# ==================== TRAINING ====================

def calculate_metrics(outputs, clean_images):
    psnr_vals, ssim_vals = [], []
    for i in range(outputs.size(0)):
        out = outputs[i].cpu().detach().numpy().transpose(1, 2, 0)
        clean = clean_images[i].cpu().detach().numpy().transpose(1, 2, 0)
        dr = clean.max() - clean.min()
        psnr_vals.append(psnr(clean, out, data_range=dr))
        ssim_vals.append(ssim(clean, out, data_range=dr, channel_axis=-1, win_size=3))
    return np.mean(psnr_vals), np.mean(ssim_vals)


def train_model(model, train_loader, val_loader, args, device):
    model.to(device)
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_loss, best_epoch, patience = float('inf'), 0, 0
    train_hist, val_hist = [], []
    
    for epoch in range(args.num_epochs):
        start = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            loss = criterion(model(noisy), clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * noisy.size(0)
        train_loss /= len(train_loader.dataset)
        train_hist.append(train_loss)
        
        # Validate
        model.eval()
        val_loss, psnr_sum, ssim_sum = 0.0, 0.0, 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                outputs = model(noisy)
                val_loss += criterion(outputs, clean).item() * noisy.size(0)
                p, s = calculate_metrics(outputs, clean)
                psnr_sum, ssim_sum = psnr_sum + p, ssim_sum + s
        
        val_loss /= len(val_loader.dataset)
        val_hist.append(val_loss)
        avg_psnr, avg_ssim = psnr_sum / len(val_loader), ssim_sum / len(val_loader)
        
        # Save & early stop
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience = val_loss, epoch + 1, 0
            torch.save(model.state_dict(), f'{args.corruption}_{args.model}.pt')
            print(f'[Epoch {best_epoch}] Saved best model (val loss: {best_val_loss:.4f})')
        else:
            patience += 1
        
        if patience >= args.early_stop_patience:
            print(f'Early stop at epoch {epoch+1}')
            break

        print(f'Epoch {epoch+1}/{args.num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | {time.time()-start:.2f}s')
    
    return train_hist, val_hist, best_epoch


def evaluate_model(model, test_loader, args, device):
    model.load_state_dict(torch.load(f'{args.corruption}_{args.model}.pt'))
    model.eval()
    test_loss, psnr_sum, ssim_sum, criterion = 0.0, 0.0, 0.0, nn.MSELoss()

    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)
            test_loss += criterion(outputs, clean).item() * noisy.size(0)
            p, s = calculate_metrics(outputs, clean)
            psnr_sum, ssim_sum = psnr_sum + p, ssim_sum + s

    test_loss /= len(test_loader.dataset)
    print(f'\nTest: Loss {test_loss:.4f}, PSNR {psnr_sum/len(test_loader):.4f}, SSIM {ssim_sum/len(test_loader):.4f}')


def main():
    parser = argparse.ArgumentParser(description='Train visual denoiser models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['BRDNet', 'DnCNN', 'DRUNet'],
                       help='Model architecture to use')
    parser.add_argument('--corruption', type=str, required=True,
                       help='Corruption type (e.g., Brightness, Contrast, Gaussian)')
    parser.add_argument('--clean_dir', type=str, required=True,
                       help='Directory containing clean images')
    parser.add_argument('--noisy_dir', type=str, required=True,
                       help='Directory containing noisy images (with L1-L5 subdirs)')
    parser.add_argument('--batch_size', type=int, default=30,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--train_split', type=float, default=0.70,
                       help='Training set split ratio')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation set split ratio')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    if args.model == 'BRDNet':
        model = BRDNet()
    elif args.model == 'DnCNN':
        model = get_dncnn_model()
    elif args.model == 'DRUNet':
        model = get_drunet_model()
    
    if model is None:
        print("Failed to initialize model")
        return
    
    # Setup dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = DenoisingDataset(
        clean_dir=args.clean_dir,
        noisy_base_dir=args.noisy_dir,
        transform=transform
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    
    # Train model
    print(f"\nTraining {args.model} on {args.corruption} corruption...")
    train_loss_history, val_loss_history, best_epoch = train_model(
        model, train_loader, val_loader, args, device
    )
    
    # Plot training curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{args.model} - {args.corruption}: Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{args.corruption}_{args.model}_loss.png')
    print(f"Saved loss plot to {args.corruption}_{args.model}_loss.png")
    
    # Evaluate on test set
    evaluate_model(model, test_loader, args, device)
    
    print(f"\nTraining complete! Best model: {args.corruption}_{args.model}.pt (epoch {best_epoch})")


if __name__ == "__main__":
    main()

