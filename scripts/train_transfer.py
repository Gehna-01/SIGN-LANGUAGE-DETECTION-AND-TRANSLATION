"""
Pretrained ResNet-18 Transfer Learning Training Pipeline
========================================================
Fine-tunes the Pretrained ResNet-18 model on the ASL Alphabet dataset under
the exact same split, subset, and experimental conditions as the Baseline CNN.

Features:
- Loads ImageNet-1K pretrained weights via Torchvision.
- Custom 29-class regularized classification head (Linear -> BN -> ReLU -> Dropout -> Linear).
- Layer freezing strategy: 'freeze_shallow' (freezes conv1..layer2, trains layer3..layer4 + head).
- AdamW optimizer with differential learning rates (1e-4 for backbone, 1e-3 for classification head).
- Cross-Entropy Loss with logits.
- Learning rate scheduling via ReduceLROnPlateau.
- Macro-F1 and Categorical Accuracy tracking.
- Dual checkpointing (best_resnet18_transfer.pt and latest_resnet18_transfer.pt).
- Full metrics export to JSON and CSV.
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
from src.models.transfer_resnet import SignResNet18Transfer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 Transfer Learning Sign Classifier")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="Learning rate for unfrozen backbone (default: 1e-4)")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Learning rate for classification head (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW (default: 1e-4)")
    parser.add_argument("--freeze-strategy", type=str, default="freeze_shallow",
                        choices=["full", "freeze_all", "freeze_shallow"],
                        help="Freezing strategy (default: freeze_shallow)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    parser.add_argument("--subset-fraction", type=float, default=0.05,
                        help="Fraction of dataset to use (default: 0.05 to match baseline benchmark exactly)")
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
        
        # Gradient clipping for numerical stability
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
    print("RESNET-18 TRANSFER LEARNING TRAINING PIPELINE")
    print("=" * 85)
    print(f"Device:           {device}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Backbone LR:      {args.backbone_lr}")
    print(f"Head LR:          {args.head_lr}")
    print(f"Weight Decay:     {args.weight_decay}")
    print(f"Freeze Strategy:  {args.freeze_strategy}")
    print(f"Subset Fraction:  {args.subset_fraction:.1%} (Exact parity with Step 17 Baseline)")
    print(f"Seed:             {args.seed}")
    print("-" * 85)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    # 1. Load Data with exact same splitting and subsetting
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir='data/splits',
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed
    )
    
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
    
    # 2. Instantiate Pretrained ResNet-18 Model
    model = SignResNet18Transfer(
        num_classes=len(classes),
        pretrained=True,
        fine_tune_mode=args.freeze_strategy
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Separate parameters for differential learning rate
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr}
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-6
    )
    
    # 3. Training Loop
    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_resnet18_transfer.pt")
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_resnet18_transfer.pt")
    
    print("\nStarting Transfer Learning Optimization...")
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val F1 (Macro)':<14} | {'Elapsed':<8}")
    print("-" * 85)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_sec = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1, val_sec = evaluate(model, val_loader, criterion, device)
        total_sec = train_sec + val_sec
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[-1]['lr']
        
        is_best = val_macro_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            no_improve_epochs = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'val_macro_f1': float(val_macro_f1),
                'classes': classes,
                'args': vars(args)
            }, best_checkpoint_path)
            best_marker = " [BEST]"
        else:
            no_improve_epochs += 1
            best_marker = ""
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'val_macro_f1': float(val_macro_f1),
            'classes': classes,
            'args': vars(args)
        }, latest_checkpoint_path)
        
        epoch_record = {
            'epoch': epoch,
            'train_loss': round(float(train_loss), 5),
            'train_acc': round(float(train_acc), 5),
            'val_loss': round(float(val_loss), 5),
            'val_acc': round(float(val_acc), 5),
            'val_macro_f1': round(float(val_macro_f1), 5),
            'lr': current_lr,
            'time_seconds': round(float(total_sec), 2),
            'is_best': bool(is_best)
        }
        history.append(epoch_record)
        
        print(f"{epoch:<6} | {train_loss:<11.4f} | {train_acc:<10.2%} | {val_loss:<10.4f} | {val_acc:<9.2%} | {val_macro_f1:<14.2%}{best_marker} | {total_sec:<6.1f}s")
        
        if no_improve_epochs >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs).")
            break
            
    print("-" * 85)
    print(f"Training Complete! Best Validation Macro-F1: {best_val_f1:.2%} at Epoch {best_epoch}")
    print(f"Best checkpoint saved to:   {best_checkpoint_path}")
    print(f"Latest checkpoint saved to: {latest_checkpoint_path}")
    
    # 4. Save Metric History
    metrics_json_path = os.path.join(args.metrics_dir, "resnet18_training_metrics.json")
    metrics_csv_path = os.path.join(args.metrics_dir, "resnet18_training_metrics.csv")
    
    summary_data = {
        'metadata': {
            'model': 'SignResNet18Transfer',
            'freeze_strategy': args.freeze_strategy,
            'device': str(device),
            'timestamp': datetime.now().isoformat(),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'backbone_lr': args.backbone_lr,
            'head_lr': args.head_lr,
            'weight_decay': args.weight_decay,
            'subset_fraction': args.subset_fraction,
            'seed': args.seed,
            'num_classes': len(classes)
        },
        'epochs': history
    }
    
    with open(metrics_json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Metrics JSON saved to:     {metrics_json_path}")
    
    df = pd.DataFrame(history)
    df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics CSV saved to:      {metrics_csv_path}")
    print("=" * 85)


if __name__ == "__main__":
    run_training()
