"""
Baseline CNN Training Pipeline & Experiment Setup
=================================================
End-to-end reproducible PyTorch training pipeline for the SignBaselineCNN model
on the 29-class ASL Alphabet dataset.

Features:
- AdamW optimizer with decoupled weight decay.
- Cross-Entropy Loss with logits.
- Learning rate scheduling via ReduceLROnPlateau.
- Macro-F1 score and accuracy tracking on validation set.
- Checkpoint management (best_baseline_cnn.pt and latest_baseline_cnn.pt).
- Early stopping mechanism.
- Full metric history export to JSON and CSV.
- Configurable subset fraction for CPU-efficient benchmarking or full training.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import get_dataloaders, set_seed
from src.models.baseline_cnn import SignBaselineCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline CNN Sign Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    parser.add_argument("--subset-fraction", type=float, default=0.1,
                        help="Fraction of dataset to use (default: 0.1 for CPU speed, set 1.0 for full)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--metrics-dir", type=str, default="docs/assets", help="Directory to save metrics logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executes a single training epoch across all mini-batches.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    start_time = time.time()
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    elapsed = time.time() - start_time
    
    return epoch_loss, epoch_acc, elapsed


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on validation or test sets.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    elapsed = time.time() - start_time
    
    return val_loss, val_acc, val_macro_f1, elapsed


def run_training():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("BASELINE CNN TRAINING PIPELINE INITIALIZATION")
    print("=" * 85)
    print(f"Device:           {device}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Weight Decay:     {args.weight_decay}")
    print(f"Subset Fraction:  {args.subset_fraction:.1%} ({'Subsampled for fast CPU convergence' if args.subset_fraction < 1.0 else 'Full Dataset'})")
    print(f"Seed:             {args.seed}")
    print("-" * 85)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    # 1. Load Data
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir='data/splits',
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed
    )
    
    # Apply subsetting if subset_fraction < 1.0
    if args.subset_fraction < 1.0:
        np.random.seed(args.seed)
        train_ds = train_loader.dataset
        val_ds = val_loader.dataset
        
        n_train = int(len(train_ds) * args.subset_fraction)
        n_val = int(len(val_ds) * args.subset_fraction)
        
        train_indices = np.random.choice(len(train_ds), size=n_train, replace=False)
        val_indices = np.random.choice(len(val_ds), size=n_val, replace=False)
        
        train_loader = DataLoader(
            Subset(train_ds, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            Subset(val_ds, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        print(f"Subsampled Training Samples:   {len(train_loader.dataset):,} / 60,900")
        print(f"Subsampled Validation Samples: {len(val_loader.dataset):,} / 13,050")
    else:
        print(f"Training Samples:   {len(train_loader.dataset):,} (100% full split)")
        print(f"Validation Samples: {len(val_loader.dataset):,} (100% full split)")
        
    print(f"Target Classes:     {len(classes)} ({', '.join(classes[:5])}... {classes[-1]})")
    print("-" * 85)
    
    # 2. Instantiate Model, Loss & Optimizer
    model = SignBaselineCNN(num_classes=len(classes), in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    best_macro_f1 = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    history = {
        "metadata": {
            "model": "SignBaselineCNN",
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "subset_fraction": args.subset_fraction,
            "seed": args.seed,
            "num_classes": len(classes)
        },
        "epochs": []
    }
    
    print(f"{'Epoch':<7} | {'Train Loss':<11} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val Macro-F1':<13} | {'LR':<9} | {'Time':<7}")
    print("-" * 85)
    
    total_start = time.time()
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, t_train = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_f1, t_val = evaluate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        epoch_time = t_train + t_val
        is_best = val_f1 > best_macro_f1
        
        if is_best:
            best_macro_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best checkpoint
            best_ckpt_path = os.path.join(args.checkpoint_dir, "best_baseline_cnn.pt")
            torch.save({
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_macro_f1": float(val_f1),
                "classes": classes,
                "hyperparameters": vars(args)
            }, best_ckpt_path)
            ckpt_indicator = " [BEST SAVED]"
        else:
            patience_counter += 1
            ckpt_indicator = ""
            
        # Save latest checkpoint
        latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest_baseline_cnn.pt")
        torch.save({
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "classes": classes
        }, latest_ckpt_path)
        
        # Record metrics
        epoch_record = {
            "epoch": int(epoch),
            "train_loss": round(float(train_loss), 5),
            "train_acc": round(float(train_acc), 5),
            "val_loss": round(float(val_loss), 5),
            "val_acc": round(float(val_acc), 5),
            "val_macro_f1": round(float(val_f1), 5),
            "lr": float(current_lr),
            "time_seconds": round(float(epoch_time), 2),
            "is_best": bool(is_best)
        }
        history["epochs"].append(epoch_record)
        
        print(f"{epoch:<7d} | {train_loss:<11.4f} | {train_acc*100:<9.2f}% | {val_loss:<10.4f} | {val_acc*100:<8.2f}% | {val_f1*100:<12.2f}% | {current_lr:<9.1e} | {epoch_time:<6.1f}s{ckpt_indicator}")
        
        # Early Stopping
        if patience_counter >= args.patience:
            print("-" * 85)
            print(f"Early stopping triggered at epoch {epoch} (no validation improvement for {args.patience} epochs).")
            break
            
    total_elapsed = time.time() - total_start
    print("-" * 85)
    print(f"Training Complete! Total Time: {total_elapsed:.1f}s")
    print(f"Best Validation Epoch: {best_epoch} | Val Loss: {best_val_loss:.4f} | Val Macro-F1: {best_macro_f1*100:.2f}%")
    print(f"Checkpoints Saved: {os.path.join(args.checkpoint_dir, 'best_baseline_cnn.pt')}")
    print("=" * 85)
    
    # Custom encoder for numpy types
    def numpy_serializer(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        return str(o)

    # Save Metrics Logs
    json_path = os.path.join(args.metrics_dir, "baseline_training_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2, default=numpy_serializer)
    print(f"Saved JSON metrics to: {json_path}")
    
    df_metrics = pd.DataFrame(history["epochs"])
    csv_path = os.path.join(args.metrics_dir, "baseline_training_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"Saved CSV metrics to:  {csv_path}")


if __name__ == "__main__":
    run_training()
