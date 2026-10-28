"""
Reproducible Data Pipeline for Sign Language Recognition
University Hackathon Project 13: Sign Language Recognition and Translation Assistant
Author: Gehna
Date: 2026-10-28
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Standard dataset constants
DEFAULT_IMG_SIZE = (64, 64)
NORMALIZATION_MEAN = [0.5, 0.5, 0.5]
NORMALIZATION_STD = [0.5, 0.5, 0.5]

def set_seed(seed: int = 42):
    """
    Ensures complete end-to-end reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_train_transforms(img_size=DEFAULT_IMG_SIZE):
    """
    Semantics-Preserving Training Augmentations:
    - Minor rotation: +/- 10 degrees
    - Minor translation: +/- 5%
    - Minor scaling: 95% to 105%
    - Minor brightness and contrast jitter: +/- 10%
    
    CRITICAL POLICY:
    RandomHorizontalFlip is STRICTLY EXCLUDED.
    Horizontal flipping reverses handedness and spatial orientation,
    corrupting the linguistic semantics of sign language.
    """
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomRotation(degrees=(-10, 10)),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    ])

def get_eval_transforms(img_size=DEFAULT_IMG_SIZE):
    """
    Strictly deterministic preprocessing for Validation and Testing.
    Zero random transformations.
    """
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    ])

class ASLDataset(Dataset):
    """
    PyTorch Dataset backed by reproducible CSV manifests.
    Keeps the underlying image repository read-only and immutable.
    """
    def __init__(self, manifest_csv: str, transform=None):
        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest file not found: {manifest_csv}")
            
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.classes = sorted(self.df['class_name'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        label = int(row['label'])
        
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if self.transform is not None:
                    tensor_img = self.transform(img)
                else:
                    tensor_img = T.ToTensor()(img)
            return tensor_img, label
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

def get_dataloaders(
    data_dir: str = 'data/splits',
    batch_size: int = 64,
    img_size=DEFAULT_IMG_SIZE,
    num_workers: int = 0,
    seed: int = 42
):
    """
    Constructs train, validation, and test DataLoaders from manifests.
    """
    set_seed(seed)
    
    train_csv = os.path.join(data_dir, 'train_manifest.csv')
    val_csv = os.path.join(data_dir, 'val_manifest.csv')
    test_csv = os.path.join(data_dir, 'test_manifest.csv')
    
    train_dataset = ASLDataset(train_csv, transform=get_train_transforms(img_size))
    val_dataset   = ASLDataset(val_csv, transform=get_eval_transforms(img_size))
    test_dataset  = ASLDataset(test_csv, transform=get_eval_transforms(img_size))
    
    generator = torch.Generator().manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes
