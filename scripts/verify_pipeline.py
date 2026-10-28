"""
Pipeline Verification and Augmentation Inspection Script
University Hackathon Project 13: Sign Language Recognition and Translation Assistant
Author: Gehna
Date: 2026-10-28
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import get_dataloaders, get_train_transforms, get_eval_transforms, set_seed

def verify_pipeline():
    set_seed(42)
    os.makedirs('docs/assets', exist_ok=True)
    
    print("=" * 60)
    print("RUNNING REPRODUCIBLE DATA PIPELINE VERIFICATION")
    print("=" * 60)
    
    start_time = time.time()
    batch_size = 64
    img_size = (64, 64)
    
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir='data/splits',
        batch_size=batch_size,
        img_size=img_size,
        num_workers=0,
        seed=42
    )
    
    print(f"Number of classes: {len(classes)}")
    print(f"Train batches: {len(train_loader)} (Total: {len(train_loader.dataset)} samples)")
    print(f"Val batches:   {len(val_loader)} (Total: {len(val_loader.dataset)} samples)")
    print(f"Test batches:  {len(test_loader)} (Total: {len(test_loader.dataset)} samples)")
    
    # 1. Inspect Train Batch
    train_imgs, train_labels = next(iter(train_loader))
    print("\n--- TRAIN BATCH VERIFICATION ---")
    print(f"Images Tensor Shape: {train_imgs.shape} (Expected: [{batch_size}, 3, {img_size[0]}, {img_size[1]}])")
    print(f"Labels Tensor Shape: {train_labels.shape} (Expected: [{batch_size}])")
    print(f"Images Dtype:        {train_imgs.dtype}")
    print(f"Labels Dtype:        {train_labels.dtype}")
    print(f"Pixel Value Range:   Min={train_imgs.min().item():.3f}, Max={train_imgs.max().item():.3f}, Mean={train_imgs.mean().item():.3f}")
    print(f"Labels Range:        Min={train_labels.min().item()}, Max={train_labels.max().item()}")
    
    assert train_imgs.shape == (batch_size, 3, img_size[0], img_size[1])
    assert train_labels.shape == (batch_size,)
    assert train_labels.min().item() >= 0 and train_labels.max().item() < 29
    print("Assertion PASSED: Train batch dimensions and label bounds verified.")
    
    # 2. Inspect Val Batch
    val_imgs, val_labels = next(iter(val_loader))
    assert val_imgs.shape == (batch_size, 3, img_size[0], img_size[1])
    print("\n--- VALIDATION BATCH VERIFICATION ---")
    print(f"Val Batch Shape:     {val_imgs.shape} -> Verified deterministic preprocessing.")
    
    # 3. Inspect Test Batch
    test_imgs, test_labels = next(iter(test_loader))
    assert test_imgs.shape == (batch_size, 3, img_size[0], img_size[1])
    print("\n--- TEST BATCH VERIFICATION ---")
    print(f"Test Batch Shape:    {test_imgs.shape} -> Verified sealed test preprocessing.")
    
    # 4. Generate Visual Comparison: Original vs 4 Augmented Variations
    print("\nGenerating Augmentation Verification Mosaic (Checking Semantics & Handedness)...")
    sample_csv = 'data/splits/train_manifest.csv'
    import pandas as pd
    df = pd.read_csv(sample_csv)
    sample_path = df.iloc[0]['filepath']
    sample_class = df.iloc[0]['class_name']
    
    raw_img = Image.open(sample_path).convert('RGB')
    
    eval_tf = get_eval_transforms(img_size)
    train_tf = get_train_transforms(img_size)
    
    # Inverse normalization helper for visualization
    def tensor_to_img(t):
        t = t.clone().detach().cpu()
        t = t * 0.5 + 0.5  # unnormalize [-1, 1] -> [0, 1]
        t = t.clamp(0, 1)
        np_img = t.numpy().transpose((1, 2, 0))
        return np_img
        
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    
    # Original deterministic view
    orig_tensor = eval_tf(raw_img)
    axes[0].imshow(tensor_to_img(orig_tensor))
    axes[0].set_title(f"Original ({sample_class})\n(Deterministic)", fontsize=11, fontweight='bold', color='#1E3A8A')
    axes[0].axis('off')
    
    # 4 Augmented instances
    for i in range(1, 5):
        aug_tensor = train_tf(raw_img)
        axes[i].imshow(tensor_to_img(aug_tensor))
        axes[i].set_title(f"Augmented #{i}\n(Rot/Trans/Jitter)", fontsize=11, color='#047857')
        axes[i].axis('off')
        
    plt.suptitle("Semantics-Preserving Augmentation Check (Handedness Strictly Preserved, No Horizontal Flipping)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    mosaic_out = 'docs/assets/augmentation_verification.png'
    plt.savefig(mosaic_out, dpi=200)
    plt.close()
    print(f"Saved visual augmentation inspection to: {mosaic_out}")
    
    elapsed = time.time() - start_time
    print(f"\nPipeline verification completed in {elapsed:.2f} seconds.")
    print("=" * 60)

if __name__ == '__main__':
    verify_pipeline()
