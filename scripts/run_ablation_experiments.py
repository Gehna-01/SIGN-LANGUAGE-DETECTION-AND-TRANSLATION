"""
Controlled Generalization & Regularization Experiments
======================================================
Scientific ablation benchmark isolating the quantitative contributions of:
1. Full Regularization (Baseline: Dropout 0.4, Weight Decay 1e-4, Data Augmentation)
2. Ablation 1: No Dropout (dropout_rate = 0.0)
3. Ablation 2: No Weight Decay (weight_decay = 0.0)
4. Ablation 3: No Data Augmentation (train on raw unaugmented images)

Maintains identical random seeds, architectures, and data subsets across all runs
to adhere to the ceteris paribus scientific standard.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import (
    ASLDataset,
    get_train_transforms,
    get_eval_transforms,
    set_seed
)
from src.models.baseline_cnn import SignBaselineCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Run Controlled Regularization Ablation Experiments")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs per condition (default: 2)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--subset-fraction", type=float, default=0.04,
                        help="Data fraction per condition for CPU speed (default: 0.04 ~2,436 train samples)")
    parser.add_argument("--output-json", type=str, default="docs/assets/ablation_experiments_summary.json",
                        help="Destination JSON summary file")
    parser.add_argument("--output-plot", type=str, default="docs/assets/ablation_comparison.png",
                        help="Destination comparison plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def get_ablation_loaders(subset_fraction: float, use_augmentation: bool, batch_size: int, seed: int):
    """
    Constructs deterministic train and validation loaders with or without data augmentation.
    """
    set_seed(seed)
    train_csv = 'data/splits/train_manifest.csv'
    val_csv = 'data/splits/val_manifest.csv'
    
    train_transform = get_train_transforms() if use_augmentation else get_eval_transforms()
    val_transform = get_eval_transforms()
    
    train_ds = ASLDataset(train_csv, transform=train_transform)
    val_ds = ASLDataset(val_csv, transform=val_transform)
    
    if subset_fraction < 1.0:
        np.random.seed(seed)
        n_train = int(len(train_ds) * subset_fraction)
        n_val = int(len(val_ds) * subset_fraction)
        
        train_idx = np.random.choice(len(train_ds), size=n_train, replace=False)
        val_idx = np.random.choice(len(val_ds), size=n_val, replace=False)
        
        train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
    return train_loader, val_loader, train_ds.classes


def run_condition(name: str, config: dict, args, device):
    """
    Trains a model under an isolated ablation condition for args.epochs.
    """
    set_seed(args.seed)
    print(f"\n>>> Running Condition: {name}")
    print(f"    Config: Dropout={config['dropout']}, WeightDecay={config['weight_decay']}, Augmentation={config['augmentation']}")
    
    train_loader, val_loader, classes = get_ablation_loaders(
        subset_fraction=args.subset_fraction,
        use_augmentation=config['augmentation'],
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    model = SignBaselineCNN(num_classes=len(classes), in_channels=3, dropout_rate=config['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=config['weight_decay'])
    
    history = []
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Training loop
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss += loss.item() * images.size(0)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(val_targets, val_preds)
        epoch_val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        
        history.append({
            "epoch": epoch,
            "train_loss": round(float(epoch_train_loss), 4),
            "train_acc": round(float(epoch_train_acc * 100), 2),
            "val_loss": round(float(epoch_val_loss), 4),
            "val_acc": round(float(epoch_val_acc * 100), 2),
            "val_macro_f1": round(float(epoch_val_f1 * 100), 2),
            "gen_gap": round(float((epoch_train_acc - epoch_val_acc) * 100), 2)
        })
        print(f"    Epoch {epoch}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc*100:.2f}%, Val Acc={epoch_val_acc*100:.2f}%, Val F1={epoch_val_f1*100:.2f}%, Gap={history[-1]['gen_gap']:.2f}%")
        
    total_time = round(time.time() - start_time, 2)
    final_metrics = history[-1]
    
    return {
        "condition": name,
        "config": config,
        "execution_time_seconds": total_time,
        "final_train_loss": final_metrics["train_loss"],
        "final_train_acc": final_metrics["train_acc"],
        "final_val_loss": final_metrics["val_loss"],
        "final_val_acc": final_metrics["val_acc"],
        "final_val_macro_f1": final_metrics["val_macro_f1"],
        "final_generalization_gap": final_metrics["gen_gap"],
        "history": history
    }


def generate_ablation_plot(results: list, save_path: str):
    """
    Renders publication-quality comparative visualization of ablation experiments.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    conditions = [r["condition"] for r in results]
    clean_labels = [c.replace("_", " ") for c in conditions]
    val_accs = [r["final_val_acc"] for r in results]
    val_f1s = [r["final_val_macro_f1"] for r in results]
    gen_gaps = [r["final_generalization_gap"] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.25)
    
    colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4']
    
    # Panel 1: Validation Accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(clean_labels)), val_accs, color=colors, alpha=0.85, edgecolor='black', width=0.55)
    ax1.set_title('(A) Final Validation Accuracy (%)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax1.set_ylim([0, max(val_accs) * 1.35])
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2.0, h + 0.3, f"{h:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_xticks(range(len(clean_labels)))
    ax1.set_xticklabels(clean_labels, rotation=20, ha='right', fontsize=10)
    
    # Panel 2: Validation Macro-F1
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(clean_labels)), val_f1s, color=colors, alpha=0.85, edgecolor='black', width=0.55)
    ax2.set_title('(B) Final Validation Macro-F1 (%)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylabel('Validation Macro-F1 (%)', fontsize=11)
    ax2.set_ylim([0, max(val_f1s) * 1.35])
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, h + 0.2, f"{h:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(len(clean_labels)))
    ax2.set_xticklabels(clean_labels, rotation=20, ha='right', fontsize=10)
    
    # Panel 3: Generalization Gap
    ax3 = axes[2]
    bars3 = ax3.bar(range(len(clean_labels)), gen_gaps, color=colors, alpha=0.85, edgecolor='black', width=0.55)
    ax3.set_title('(C) Generalization Gap (Train - Val Acc %)', fontsize=13, fontweight='bold', pad=10)
    ax3.set_ylabel('Generalization Gap (%)', fontsize=11)
    ax3.set_ylim([0, max(gen_gaps) * 1.35])
    ax3.grid(True, linestyle='--', alpha=0.5, axis='y')
    for b in bars3:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2.0, h + 0.3, f"{h:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_xticks(range(len(clean_labels)))
    ax3.set_xticklabels(clean_labels, rotation=20, ha='right', fontsize=10)
    
    fig.suptitle("Controlled Generalization & Regularization Experiments: Ablation Comparison\n(ASL Sign Language Classifier Baseline)",
                 fontsize=15, fontweight='bold', y=1.02)
                 
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved ablation comparison chart to: {save_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 95)
    print("CONTROLLED REGULARIZATION ABLATION EXPERIMENTS INITIALIZATION")
    print("=" * 95)
    print(f"Device:           {device}")
    print(f"Epochs/Condition: {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Subset Fraction:  {args.subset_fraction:.1%}")
    print(f"Seed:             {args.seed}")
    print("=" * 95)
    
    conditions = {
        "Full_Regularization": {
            "dropout": 0.4,
            "weight_decay": 1e-4,
            "augmentation": True,
            "description": "Baseline: Dropout 0.4, Weight Decay 1e-4, Spatial/Color Augmentation"
        },
        "Ablation_No_Dropout": {
            "dropout": 0.0,
            "weight_decay": 1e-4,
            "augmentation": True,
            "description": "Ablation: Dropout completely disabled (p=0.0)"
        },
        "Ablation_No_Weight_Decay": {
            "dropout": 0.4,
            "weight_decay": 0.0,
            "augmentation": True,
            "description": "Ablation: Decoupled weight decay disabled (lambda=0.0)"
        },
        "Ablation_No_Augmentation": {
            "dropout": 0.4,
            "weight_decay": 1e-4,
            "augmentation": False,
            "description": "Ablation: Train on raw unaugmented images (zero spatial/color jitter)"
        }
    }
    
    results = []
    for cond_name, cond_config in conditions.items():
        res = run_condition(cond_name, cond_config, args, device)
        results.append(res)
        
    print("\n" + "=" * 95)
    print("ABLATION EXPERIMENT MASTER RESULTS TABLE:")
    print("=" * 95)
    print(f"{'Condition':<26} | {'Val Acc':<10} | {'Val Macro-F1':<14} | {'Gen Gap':<10} | {'Train Loss':<12} | {'Val Loss':<10}")
    print("-" * 95)
    for r in results:
        print(f"{r['condition']:<26} | {r['final_val_acc']:<9.2f}% | {r['final_val_macro_f1']:<13.2f}% | {r['final_generalization_gap']:<9.2f}% | {r['final_train_loss']:<12.4f} | {r['final_val_loss']:<10.4f}")
    print("=" * 95)
    
    # Save structured summary
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump({
            "metadata": {
                "experiment": "Controlled Regularization Ablation Study",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "epochs_per_condition": args.epochs,
                "subset_fraction": args.subset_fraction,
                "seed": args.seed,
                "device": str(device)
            },
            "results": results
        }, f, indent=2)
    print(f"[SUCCESS] Saved ablation experiment JSON log to: {args.output_json}")
    
    # Render Plot
    generate_ablation_plot(results, args.output_plot)


if __name__ == "__main__":
    main()
