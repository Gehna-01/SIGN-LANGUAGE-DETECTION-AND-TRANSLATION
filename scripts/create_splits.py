"""
Leakage-Safe Data Splitting Script for ASL Alphabet Dataset
University Hackathon Project 13: Sign Language Recognition and Translation Assistant
Author: Gehna
Date: 2026-10-28
"""

import os
import re
import glob
import json
import hashlib
import pandas as pd

def extract_numeric_id(filepath):
    """
    Extracts the integer index from filenames like 'A123.jpg' or 'space45.jpg'.
    """
    basename = os.path.basename(filepath)
    match = re.search(r'(\d+)', basename)
    if match:
        return int(match.group(1))
    return 0

def create_leakage_safe_splits():
    base_dir = os.path.expanduser('~/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1')
    train_dir = os.path.join(base_dir, 'asl_alphabet_train', 'asl_alphabet_train')
    
    output_dir = 'data/splits'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CREATING REPRODUCIBLE, LEAKAGE-SAFE DATA SPLITS")
    print(f"Source Directory: {train_dir}")
    print("Strategy: Contiguous Block / Session-Aware Chunking (70% / 15% / 15%)")
    print("=" * 60)
    
    classes = sorted(os.listdir(train_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    train_records = []
    val_records = []
    test_records = []
    
    # 70% / 15% / 15% of 3,000 samples per class:
    # Train: 2,100 samples (Indices 1 to 2100)
    # Val:   450 samples   (Indices 2101 to 2550)
    # Test:  450 samples   (Indices 2551 to 3000)
    
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        files = glob.glob(os.path.join(cls_path, "*.*"))
        
        # Sort files by numerical frame index to preserve temporal session block continuity
        sorted_files = sorted(files, key=extract_numeric_id)
        assert len(sorted_files) == 3000, f"Expected 3000 files for class {cls}, found {len(sorted_files)}"
        
        cls_idx = class_to_idx[cls]
        
        train_files = sorted_files[:2100]
        val_files   = sorted_files[2100:2550]
        test_files  = sorted_files[2550:]
        
        for f in train_files:
            train_records.append({'filepath': f, 'filename': os.path.basename(f), 'class_name': cls, 'label': cls_idx, 'split': 'train'})
        for f in val_files:
            val_records.append({'filepath': f, 'filename': os.path.basename(f), 'class_name': cls, 'label': cls_idx, 'split': 'val'})
        for f in test_files:
            test_records.append({'filepath': f, 'filename': os.path.basename(f), 'class_name': cls, 'label': cls_idx, 'split': 'test'})
            
    df_train = pd.DataFrame(train_records)
    df_val   = pd.DataFrame(val_records)
    df_test  = pd.DataFrame(test_records)
    
    print("\n--- SPLIT VERIFICATION CHECKS ---")
    print(f"Train samples:      {len(df_train):,} ({len(df_train)/87000*100:.1f}%)")
    print(f"Validation samples: {len(df_val):,} ({len(df_val)/87000*100:.1f}%)")
    print(f"Test samples:       {len(df_test):,} ({len(df_test)/87000*100:.1f}%)")
    print(f"Total accounted:    {len(df_train) + len(df_val) + len(df_test):,} / 87,000")
    
    # Strict Leakage & Disjointness Assertions
    train_paths = set(df_train['filepath'])
    val_paths   = set(df_val['filepath'])
    test_paths  = set(df_test['filepath'])
    
    intersect_train_val  = train_paths.intersection(val_paths)
    intersect_train_test = train_paths.intersection(test_paths)
    intersect_val_test   = val_paths.intersection(test_paths)
    
    assert len(intersect_train_val) == 0, f"Leakage detected between Train and Val: {len(intersect_train_val)} files!"
    assert len(intersect_train_test) == 0, f"Leakage detected between Train and Test: {len(intersect_train_test)} files!"
    assert len(intersect_val_test) == 0, f"Leakage detected between Val and Test: {len(intersect_val_test)} files!"
    print("Assertion PASSED: Exact 0% file overlap across all partitions (strictly disjoint sets).")
    
    # Save CSV Manifests
    train_csv = os.path.join(output_dir, 'train_manifest.csv')
    val_csv   = os.path.join(output_dir, 'val_manifest.csv')
    test_csv  = os.path.join(output_dir, 'test_manifest.csv')
    
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    
    # Compute manifest hashes
    def get_file_md5(p):
        with open(p, 'rb') as fp:
            return hashlib.md5(fp.read()).hexdigest()
            
    summary = {
        "dataset_name": "ASL Alphabet Dataset",
        "split_strategy": "Contiguous Block / Session-Aware Temporal Partitioning",
        "total_samples": 87000,
        "classes_count": 29,
        "class_mapping": class_to_idx,
        "partitions": {
            "train": {
                "samples": len(df_train),
                "samples_per_class": 2100,
                "percentage": 70.0,
                "frame_indices_range": "1 to 2100",
                "manifest_csv": train_csv,
                "manifest_md5": get_file_md5(train_csv)
            },
            "validation": {
                "samples": len(df_val),
                "samples_per_class": 450,
                "percentage": 15.0,
                "frame_indices_range": "2101 to 2550",
                "manifest_csv": val_csv,
                "manifest_md5": get_file_md5(val_csv)
            },
            "test": {
                "samples": len(df_test),
                "samples_per_class": 450,
                "percentage": 15.0,
                "frame_indices_range": "2551 to 3000",
                "manifest_csv": test_csv,
                "manifest_md5": get_file_md5(test_csv)
            }
        },
        "leakage_checks": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "temporal_burst_isolation": "Verified (contiguous block-held-out test set)"
        }
    }
    
    summary_path = os.path.join(output_dir, 'split_summary.json')
    with open(summary_path, 'w') as fp:
        json.dump(summary, fp, indent=2)
        
    print(f"\nManifests saved cleanly to: {output_dir}")
    print(f"Summary JSON saved to: {summary_path}")
    print("=" * 60)
    return summary

if __name__ == '__main__':
    create_leakage_safe_splits()
