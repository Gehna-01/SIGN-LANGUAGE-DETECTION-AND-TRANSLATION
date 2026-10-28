"""
Baseline CNN vs Pretrained ResNet-18 Benchmark
==============================================
Runs an end-to-head empirical evaluation comparing the custom 4-stage
Baseline CNN against the fine-tuned Pretrained ResNet-18 Transfer Learning model.

Evaluates:
- Validation Accuracy (Top-1 and Top-3).
- Macro Precision, Macro Recall, Macro F1.
- Per-class F1 scores and Delta F1 (ResNet - Baseline).
- Parameter count and Checkpoint disk footprint.
- Inference Latency on CPU (Mean, Std, P50, P95 per single 64x64 image).
- Generates publication-ready 4-panel comparison visualization.
- Exports comprehensive JSON benchmark summary.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, top_k_accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import get_dataloaders, set_seed
from src.models.baseline_cnn import SignBaselineCNN
from src.models.transfer_resnet import SignResNet18Transfer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Baseline CNN vs Pretrained ResNet-18")
    parser.add_argument("--baseline-checkpoint", type=str, default="checkpoints/best_baseline_cnn.pt")
    parser.add_argument("--resnet-checkpoint", type=str, default="checkpoints/best_resnet18_transfer.pt")
    parser.add_argument("--val-fraction", type=float, default=0.20,
                        help="Fraction of validation set to evaluate (0.20 = 2,610 samples)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latency-runs", type=int, default=200, help="Number of latency profiling runs")
    parser.add_argument("--output-json", type=str, default="docs/assets/baseline_vs_resnet_benchmark.json")
    parser.add_argument("--output-plot", type=str, default="docs/assets/baseline_vs_resnet_comparison.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def measure_inference_latency(model, device, num_runs=200, warm_up=30):
    """
    Profiles single-sample CPU inference latency in milliseconds.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    
    # Warm-up passes
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(dummy_input)
            
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
            
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "fps": float(1000.0 / np.mean(latencies))
    }


def evaluate_model(model, dataloader, device, num_classes=29):
    """
    Evaluates classification metrics on a dataset.
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_targets.extend(targets.numpy())
            all_preds.extend(preds)
            all_probs.append(probs)
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    
    acc = float(accuracy_score(all_targets, all_preds))
    top3_acc = float(top_k_accuracy_score(all_targets, all_probs, k=3, labels=list(range(num_classes))))
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(num_classes)), average='macro', zero_division=0
    )
    
    class_p, class_r, class_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(num_classes)), average=None, zero_division=0
    )
    
    return {
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class_f1": [float(x) for x in class_f1]
    }


def run_benchmark():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")  # Strictly profile CPU edge kiosk performance
    
    print("=" * 85)
    print("BASELINE CNN VS PRETRAINED RESNET-18 BENCHMARK SUITE")
    print("=" * 85)
    print(f"Device:               {device}")
    print(f"Baseline Checkpoint:  {args.baseline_checkpoint}")
    print(f"ResNet Checkpoint:    {args.resnet_checkpoint}")
    print(f"Validation Fraction:  {args.val_fraction:.1%} (Contiguous evaluation)")
    print(f"Latency Profiling:    {args.latency_runs} single-image runs")
    print("-" * 85)
    
    # 1. Load Data
    _, val_loader, _, classes = get_dataloaders(
        data_dir='data/splits',
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed
    )
    
    val_ds = val_loader.dataset
    n_val = int(len(val_ds) * args.val_fraction)
    np.random.seed(args.seed)
    val_indices = np.random.choice(len(val_ds), size=n_val, replace=False)
    val_eval_loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"Evaluated Validation Samples: {len(val_eval_loader.dataset):,} / {len(val_ds):,} across {len(classes)} classes")
    print("-" * 85)
    
    # 2. Load Models
    print("Loading Baseline CNN...")
    baseline_model = SignBaselineCNN(num_classes=len(classes), in_channels=3).to(device)
    baseline_ckpt = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    baseline_model.load_state_dict(baseline_ckpt['model_state_dict'])
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    baseline_size_mb = os.path.getsize(args.baseline_checkpoint) / (1024 * 1024)
    
    print("Loading Pretrained ResNet-18...")
    resnet_model = SignResNet18Transfer(num_classes=len(classes), pretrained=False).to(device)
    resnet_ckpt = torch.load(args.resnet_checkpoint, map_location=device, weights_only=False)
    resnet_model.load_state_dict(resnet_ckpt['model_state_dict'])
    resnet_params = sum(p.numel() for p in resnet_model.parameters())
    resnet_size_mb = os.path.getsize(args.resnet_checkpoint) / (1024 * 1024)
    
    print(f"Baseline CNN Parameters:      {baseline_params:,} ({baseline_size_mb:.2f} MB)")
    print(f"ResNet-18 Parameters:         {resnet_params:,} ({resnet_size_mb:.2f} MB)")
    print("-" * 85)
    
    # 3. Profiling Inference Latency
    print("Profiling single-image CPU inference latency...")
    baseline_latency = measure_inference_latency(baseline_model, device, num_runs=args.latency_runs)
    resnet_latency = measure_inference_latency(resnet_model, device, num_runs=args.latency_runs)
    print(f"Baseline CNN Latency:  {baseline_latency['mean_ms']:.2f} ms (+/- {baseline_latency['std_ms']:.2f}) -> {baseline_latency['fps']:.1f} FPS")
    print(f"ResNet-18 Latency:     {resnet_latency['mean_ms']:.2f} ms (+/- {resnet_latency['std_ms']:.2f}) -> {resnet_latency['fps']:.1f} FPS")
    print("-" * 85)
    
    # 4. Evaluating Classification Performance
    print("Evaluating Baseline CNN on validation set...")
    baseline_metrics = evaluate_model(baseline_model, val_eval_loader, device, num_classes=len(classes))
    print(f"Baseline CNN -> Acc: {baseline_metrics['accuracy']:.2%} | Top-3: {baseline_metrics['top3_accuracy']:.2%} | Macro-F1: {baseline_metrics['macro_f1']:.2%}")
    
    print("Evaluating Pretrained ResNet-18 on validation set...")
    resnet_metrics = evaluate_model(resnet_model, val_eval_loader, device, num_classes=len(classes))
    print(f"ResNet-18    -> Acc: {resnet_metrics['accuracy']:.2%} | Top-3: {resnet_metrics['top3_accuracy']:.2%} | Macro-F1: {resnet_metrics['macro_f1']:.2%}")
    print("-" * 85)
    
    # 5. Load Training History for Loss Curves
    baseline_history_path = "docs/assets/baseline_training_metrics.json"
    resnet_history_path = "docs/assets/resnet18_training_metrics.json"
    
    baseline_history = []
    if os.path.exists(baseline_history_path):
        with open(baseline_history_path, 'r') as f:
            baseline_history = json.load(f)['epochs']
            
    resnet_history = []
    if os.path.exists(resnet_history_path):
        with open(resnet_history_path, 'r') as f:
            resnet_history = json.load(f)['epochs']
            
    # 6. Generate 4-Panel Publication Comparison Figure
    print("Generating 4-panel comparison visualization...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    palette_baseline = "#3b82f6"  # Blue
    palette_resnet = "#10b981"    # Green
    
    # Panel 1: Key Metric Comparison Bar Chart
    ax1 = axes[0, 0]
    metric_labels = ['Top-1 Acc', 'Top-3 Acc', 'Macro Prec', 'Macro Recall', 'Macro F1']
    b_vals = [
        baseline_metrics['accuracy'] * 100,
        baseline_metrics['top3_accuracy'] * 100,
        baseline_metrics['macro_precision'] * 100,
        baseline_metrics['macro_recall'] * 100,
        baseline_metrics['macro_f1'] * 100
    ]
    r_vals = [
        resnet_metrics['accuracy'] * 100,
        resnet_metrics['top3_accuracy'] * 100,
        resnet_metrics['macro_precision'] * 100,
        resnet_metrics['macro_recall'] * 100,
        resnet_metrics['macro_f1'] * 100
    ]
    
    x = np.arange(len(metric_labels))
    width = 0.35
    rects1 = ax1.bar(x - width/2, b_vals, width, label='Baseline CNN (Scratch)', color=palette_baseline, alpha=0.9)
    rects2 = ax1.bar(x + width/2, r_vals, width, label='Pretrained ResNet-18 (Transfer)', color=palette_resnet, alpha=0.9)
    
    ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Global Classification Performance Benchmark', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=11, fontweight='semibold')
    ax1.legend(loc='upper left', frameon=True)
    ax1.set_ylim(0, max(max(b_vals), max(r_vals)) * 1.25)
    
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 2: Validation Loss & Convergence
    ax2 = axes[0, 1]
    if baseline_history and resnet_history:
        epochs_b = [e['epoch'] for e in baseline_history]
        val_loss_b = [e['val_loss'] for e in baseline_history]
        epochs_r = [e['epoch'] for e in resnet_history]
        val_loss_r = [e['val_loss'] for e in resnet_history]
        
        ax2.plot(epochs_b, val_loss_b, marker='o', linewidth=2.2, color=palette_baseline, label='Baseline CNN Val Loss')
        ax2.plot(epochs_r, val_loss_r, marker='s', linewidth=2.5, color=palette_resnet, label='ResNet-18 Val Loss')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Validation Cross-Entropy Loss', fontsize=12, fontweight='bold')
        ax2.set_title('(B) Optimization Convergence Comparison', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xticks(epochs_b)
        ax2.legend(loc='upper right', frameon=True)
    else:
        ax2.text(0.5, 0.5, "Convergence data loaded", ha='center', va='center')
        
    # Panel 3: Latency vs Model Size Trade-off
    ax3 = axes[1, 0]
    models = ['Baseline CNN', 'Pretrained ResNet-18']
    latencies = [baseline_latency['mean_ms'], resnet_latency['mean_ms']]
    sizes = [baseline_size_mb, resnet_size_mb]
    
    scatter = ax3.scatter(latencies[0], sizes[0], s=350, color=palette_baseline, label='Baseline CNN (0.43M params)', zorder=5)
    scatter2 = ax3.scatter(latencies[1], sizes[1], s=550, color=palette_resnet, label='ResNet-18 (11.25M params)', zorder=5)
    
    ax3.annotate(f"Baseline CNN\n{baseline_latency['mean_ms']:.1f} ms / {baseline_latency['fps']:.0f} FPS\n({baseline_size_mb:.2f} MB)",
                 (latencies[0], sizes[0]), xytext=(15, 10), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color=palette_baseline, lw=1.5), fontsize=10)
    ax3.annotate(f"Pretrained ResNet-18\n{resnet_latency['mean_ms']:.1f} ms / {resnet_latency['fps']:.0f} FPS\n({resnet_size_mb:.2f} MB)",
                 (latencies[1], sizes[1]), xytext=(-120, -35), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color=palette_resnet, lw=1.5), fontsize=10, fontweight='bold')
                 
    ax3.set_xlabel('CPU Single-Image Inference Latency (ms)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Model Disk Size (MB)', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Computational Footprint vs Edge Latency Trade-off', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlim(0, max(latencies) * 1.4)
    ax3.set_ylim(-5, max(sizes) * 1.3)
    ax3.axvline(33.3, color='gray', linestyle='--', alpha=0.7, label='30 FPS Threshold (33.3 ms)')
    ax3.legend(loc='upper left', frameon=True)

    # Panel 4: Per-Class F1 Score Delta (ResNet - Baseline)
    ax4 = axes[1, 1]
    f1_delta = np.array(resnet_metrics['per_class_f1']) - np.array(baseline_metrics['per_class_f1'])
    bar_colors = [palette_resnet if d >= 0 else '#ef4444' for d in f1_delta]
    
    bars = ax4.bar(classes, f1_delta * 100, color=bar_colors, alpha=0.85, width=0.6)
    ax4.axhline(0, color='black', linewidth=1.0)
    ax4.set_xlabel('ASL Alphabet Class (29 Classes)', fontsize=12, fontweight='bold')
    ax4.set_ylabel(r'$\Delta$ F1 Score (%) [ResNet - Baseline]', fontsize=12, fontweight='bold')
    ax4.set_title(r'(D) Per-Class F1 Gain from Transfer Learning ($\Delta$ F1)', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xticks(range(len(classes)))
    ax4.set_xticklabels(classes, rotation=45, fontsize=8, ha='right')
    
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {args.output_plot}")
    
    # 7. Export Benchmark JSON
    benchmark_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "evaluated_samples": len(val_eval_loader.dataset),
            "num_classes": len(classes),
            "classes": classes
        },
        "models": {
            "baseline_cnn": {
                "name": "SignBaselineCNN",
                "checkpoint": args.baseline_checkpoint,
                "parameters": baseline_params,
                "size_mb": round(baseline_size_mb, 2),
                "metrics": {
                    "accuracy": baseline_metrics['accuracy'],
                    "top3_accuracy": baseline_metrics['top3_accuracy'],
                    "macro_precision": baseline_metrics['macro_precision'],
                    "macro_recall": baseline_metrics['macro_recall'],
                    "macro_f1": baseline_metrics['macro_f1']
                },
                "latency_cpu": baseline_latency
            },
            "resnet18_transfer": {
                "name": "SignResNet18Transfer",
                "checkpoint": args.resnet_checkpoint,
                "parameters": resnet_params,
                "size_mb": round(resnet_size_mb, 2),
                "metrics": {
                    "accuracy": resnet_metrics['accuracy'],
                    "top3_accuracy": resnet_metrics['top3_accuracy'],
                    "macro_precision": resnet_metrics['macro_precision'],
                    "macro_recall": resnet_metrics['macro_recall'],
                    "macro_f1": resnet_metrics['macro_f1']
                },
                "latency_cpu": resnet_latency
            }
        },
        "relative_improvements": {
            "accuracy_multiplier": round(resnet_metrics['accuracy'] / max(1e-6, baseline_metrics['accuracy']), 2),
            "macro_f1_multiplier": round(resnet_metrics['macro_f1'] / max(1e-6, baseline_metrics['macro_f1']), 2),
            "accuracy_delta_percent": round((resnet_metrics['accuracy'] - baseline_metrics['accuracy']) * 100, 2),
            "macro_f1_delta_percent": round((resnet_metrics['macro_f1'] - baseline_metrics['macro_f1']) * 100, 2)
        },
        "per_class_f1_deltas": {
            cls_name: round(float(f1_delta[i] * 100), 2) for i, cls_name in enumerate(classes)
        }
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Benchmark summary JSON saved to: {args.output_json}")
    print("=" * 85)


if __name__ == "__main__":
    run_benchmark()
