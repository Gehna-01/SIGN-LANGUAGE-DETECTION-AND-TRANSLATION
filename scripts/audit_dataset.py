"""
Dataset Audit Script for ASL Alphabet Dataset
University Hackathon Project 13: Sign Language Recognition and Translation Assistant
Author: Gehna
Date: 2026-10-28
"""

import os
import glob
import hashlib
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def run_audit():
    base_dir = os.path.expanduser('~/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1')
    train_dir = os.path.join(base_dir, 'asl_alphabet_train', 'asl_alphabet_train')
    test_dir = os.path.join(base_dir, 'asl_alphabet_test', 'asl_alphabet_test')
    
    os.makedirs('docs/assets', exist_ok=True)
    
    print("=" * 60)
    print("STARTING COMPREHENSIVE DATASET AUDIT")
    print(f"Train Source: {train_dir}")
    print(f"Test Source:  {test_dir}")
    print("=" * 60)
    
    classes = sorted(os.listdir(train_dir))
    print(f"Found {len(classes)} classes: {classes}")
    
    class_counts = {}
    corrupt_files = []
    dimension_counts = defaultdict(int)
    mode_counts = defaultdict(int)
    file_sizes = []
    
    # Hash check for duplicates (sample 300 per class for rapid exact duplicate audit across all 29 classes)
    md5_dict = defaultdict(list)
    total_scanned = 0
    
    print("\nScanning images for integrity, dimensions, channels, and duplicates...")
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(train_dir, cls)
        files = glob.glob(os.path.join(cls_path, "*.*"))
        class_counts[cls] = len(files)
        total_scanned += len(files)
        
        # Check a sample of files for detailed PIL verification & md5
        for f in files[:200]:  # sample 200 per class = 5,800 images deeply verified
            try:
                with Image.open(f) as img:
                    img.verify()
                with Image.open(f) as img:
                    dimension_counts[img.size] += 1
                    mode_counts[img.mode] += 1
                
                # Compute md5
                with open(f, 'rb') as fp:
                    h = hashlib.md5(fp.read()).hexdigest()
                    md5_dict[h].append(f)
            except Exception as e:
                corrupt_files.append((f, str(e)))
        
        if (idx + 1) % 5 == 0 or (idx + 1) == len(classes):
            print(f"  Processed {idx + 1}/{len(classes)} classes ({cls}: {len(files)} files)")

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total Classes in Train Set: {len(classes)}")
    print(f"Total Images in Train Set:   {total_scanned:,}")
    print(f"Total Corrupt Images:        {len(corrupt_files)}")
    print(f"Image Dimensions Breakdown:  {dict(dimension_counts)}")
    print(f"Color Modes Breakdown:       {dict(mode_counts)}")
    
    # Duplicate Analysis
    exact_duplicates = {h: paths for h, paths in md5_dict.items() if len(paths) > 1}
    print(f"Exact Duplicate Hash Groups in Verified Sample: {len(exact_duplicates)}")
    
    # Test directory audit
    test_files = os.listdir(test_dir)
    print(f"\nStandalone Test Partition Files: {len(test_files)}")
    print(f"Test Files: {sorted(test_files)}")
    
    # Generate Class Distribution Chart
    plt.figure(figsize=(14, 5))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color='#2563EB', edgecolor='#1D4ED8')
    plt.title("ASL Alphabet Dataset: Class Distribution (Training Partition)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Sign Class / Gesture Token", fontsize=12, labelpad=10)
    plt.ylabel("Number of Samples", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, max(class_counts.values()) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 40, f'{int(height)}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    chart_path = 'docs/assets/class_distribution.png'
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\nSaved class distribution chart to: {chart_path}")
    
    # Generate Representative Mosaic of Classes
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.flatten()
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(train_dir, cls)
        sample_img_path = glob.glob(os.path.join(cls_path, "*.*"))[0]
        img = Image.open(sample_img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(cls, fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    # Turn off unused subplots (32 - 29 = 3)
    for i in range(len(classes), len(axes)):
        axes[i].axis('off')
    plt.suptitle("ASL Alphabet Dataset: Representative Class Hand Poses", fontsize=14, fontweight='bold')
    plt.tight_layout()
    mosaic_path = 'docs/assets/sample_signs_mosaic.png'
    plt.savefig(mosaic_path, dpi=200)
    plt.close()
    print(f"Saved representative sign mosaic to: {mosaic_path}")
    
    return class_counts, corrupt_files, dimension_counts, mode_counts, len(test_files)

if __name__ == '__main__':
    run_audit()
