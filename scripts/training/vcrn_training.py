#!/usr/bin/env python3
"""
Visual Corruption Routing Network (VCRN) Training Script
Trains a ResNet50 classifier to identify corruption types and route to appropriate denoisers.

Usage:
    python vcrn_training.py --data_dir dataset/ --batch_size 30 --num_epochs 40
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image


# ==================== DATASET ====================

class NoiseClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels = [], []
        self.noise_types = sorted(os.listdir(root_dir))
        
        for idx, noise_type in enumerate(self.noise_types):
            folder = os.path.join(root_dir, noise_type)
            if not os.path.isdir(folder):
                continue
            for img in os.listdir(folder):
                path = os.path.join(folder, img)
                if path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(path)
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        return (self.transform(image) if self.transform else image, self.labels[idx])
    
    def get_noise_types(self):
        return self.noise_types


# ==================== TRAINING ====================

def train_model(model, train_loader, val_loader, criterion, optimizer, args, device):
    best_val_loss, best_val_acc, early_stop_counter = float('inf'), 0.0, 0
    print(f"\nTraining for {args.num_epochs} epochs (patience: {args.patience}) on {device}\n")
    
    for epoch in range(args.num_epochs):
        start = time.time()
        
        # Training
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
        
        train_loss, train_acc = train_loss / total, 100.0 * correct / total
        
        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * images.size(0)
                correct += outputs.max(1)[1].eq(labels).sum().item()
                total += labels.size(0)
        
        val_loss, val_acc = val_loss / total, 100.0 * correct / total
        
        # Save best & early stopping
        if val_loss < best_val_loss:
            best_val_loss, best_val_acc, early_stop_counter = val_loss, val_acc, 0
            torch.save(model.state_dict(), args.output_model)
            print(f'[Epoch {epoch+1}] Val improved! Saved model')
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= args.patience:
            print(f'\nEarly stop at epoch {epoch+1}: Best val loss {best_val_loss:.4f}, acc {best_val_acc:.2f}%')
            break
        
        print(f'Epoch {epoch+1}/{args.num_epochs} | Train: {train_loss:.4f}/{train_acc:.2f}% | Val: {val_loss:.4f}/{val_acc:.2f}% | {time.time()-start:.2f}s')
    
    print(f"\nTraining complete! Best: {args.output_model} ({best_val_acc:.2f}%)")
    return best_val_loss, best_val_acc


def evaluate_model(model, test_loader, criterion, model_path, device):
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()
    
    test_loss, correct, total = 0.0, 0, 0
    class_correct, class_total = {}, {}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item() * images.size(0)
            predicted = outputs.max(1)[1]
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            for label, pred in zip(labels, predicted):
                l = label.item()
                class_total[l] = class_total.get(l, 0) + 1
                if label == pred:
                    class_correct[l] = class_correct.get(l, 0) + 1
    
    test_loss, test_acc = test_loss / total, 100.0 * correct / total
    print(f'\n{"="*60}\nTest: Loss {test_loss:.4f}, Acc {test_acc:.2f}%\n{"="*60}\n')
    return test_loss, test_acc, class_correct, class_total


def count_dataset_statistics(dataset_dir):
    print(f"\n{'='*60}\nDataset Statistics:\n{'='*60}")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
        print(f"\n{split.upper()}:")
        total = 0
        for noise_type in sorted(os.listdir(split_dir)):
            folder = os.path.join(split_dir, noise_type)
            if os.path.isdir(folder):
                count = len([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {noise_type:20s}: {count:5d}")
                total += count
        print(f"  {'Total':20s}: {total:5d}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train Visual Corruption Routing Network (VCRN)'
    )
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Base directory containing train/val/test subdirs')
    parser.add_argument('--output_model', type=str, default='corruption_classifier_resnet50.pt',
                       help='Output model filename')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=30,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=40,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model configuration
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ResNet50 weights')
    parser.add_argument('--num_classes', type=int, default=18,
                       help='Number of corruption types (classes)')
    
    # Evaluation
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate existing model (no training)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Display dataset statistics
    count_dataset_statistics(args.data_dir)
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')
    
    train_dataset = NoiseClassificationDataset(train_dir, transform=train_transforms)
    val_dataset = NoiseClassificationDataset(val_dir, transform=test_val_transforms)
    test_dataset = NoiseClassificationDataset(test_dir, transform=test_val_transforms)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    
    # Get corruption types
    noise_types = train_dataset.get_noise_types()
    print(f"\nCorruption types ({len(noise_types)}):")
    for i, noise_type in enumerate(noise_types):
        print(f"  {i}: {noise_type}")
    print()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    
    # Initialize model
    if args.pretrained:
        print("Loading pre-trained ResNet50 model...")
        resnet_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        print("Initializing ResNet50 model from scratch...")
        resnet_model = models.resnet50(weights=None)
    
    # Modify final layer for corruption classification
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, args.num_classes)
    resnet_model = resnet_model.to(device)
    
    print(f"Model initialized with {args.num_classes} output classes")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=args.learning_rate)
    
    if not args.eval_only:
        # Train model
        best_val_loss, best_val_acc = train_model(
            resnet_model, train_loader, val_loader, 
            criterion, optimizer, args, device
        )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, class_correct, class_total = evaluate_model(
        resnet_model, test_loader, criterion, args.output_model, device
    )
    
    # Display per-class accuracy
    print("Per-class accuracy:")
    for class_idx in sorted(class_correct.keys()):
        if class_total[class_idx] > 0:
            acc = 100.0 * class_correct[class_idx] / class_total[class_idx]
            noise_type = noise_types[class_idx] if class_idx < len(noise_types) else "Unknown"
            print(f"  {noise_type:20s}: {acc:5.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})")
    
    print(f"\nTraining complete! Model saved to: {args.output_model}")


if __name__ == "__main__":
    main()

