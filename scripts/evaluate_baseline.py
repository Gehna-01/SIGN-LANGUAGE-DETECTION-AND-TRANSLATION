"""
Baseline CNN Evaluation & Confusion Matrix Analysis
===================================================
Independent evaluation module for the Baseline CNN Sign Classifier.
Computes:
- Overall Accuracy, Macro-Precision, Macro-Recall, Macro-F1
- Class-by-class metrics table (26 letters + 3 operational tokens)
- 29x29 Confusion Matrix with normalized heatmap visualization
- Error analysis ranking the top confused sign pairs (e.g. M vs N, E vs S)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import get_dataloaders, set_seed
from src.models.baseline_cnn import SignBaselineCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Baseline CNN Sign Classifier")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_baseline_cnn.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Split to evaluate: 'val' or 'test'")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--subset-fraction", type=float, default=0.2,
                        help="Fraction of split to evaluate (default: 0.2 for fast CPU, 1.0 for full)")
    parser.add_argument("--output-plot", type=str, default="docs/assets/baseline_confusion_matrix.png",
                        help="Filepath for confusion matrix heatmap")
    parser.add_argument("--output-json", type=str, default="docs/assets/baseline_evaluation_results.json",
                        help="Filepath for evaluation metrics JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def plot_confusion_matrix(cm_normalized, classes, save_path, top_confused_pairs=None):
    """
    Renders and saves a publication-quality 29-class confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(16, 14))
    
    # Custom color palette with clear distinction
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    ax = sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Normalized Prediction Probability (True Class Recall)'},
        linewidths=0.5,
        linecolor="#e0e0e0"
    )
    
    plt.title("Baseline CNN Sign Classifier: 29-Class Normalized Confusion Matrix\n(Evaluation on ASL Alphabet Validation Split)",
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Predicted Class Label", fontsize=13, fontweight='bold', labelpad=10)
    plt.ylabel("True Ground-Truth Class Label", fontsize=13, fontweight='bold', labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved high-resolution confusion matrix plot to: {save_path}")


def run_evaluation():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("BASELINE CNN SIGN CLASSIFIER EVALUATION & ERROR ANALYSIS")
    print("=" * 95)
    print(f"Device:           {device}")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Target Split:     {args.split.upper()}")
    print(f"Subset Fraction:  {args.subset_fraction:.1%} ({'Subsampled for fast CPU inference' if args.subset_fraction < 1.0 else 'Full Split'})")
    print("-" * 95)
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}. Please run training first.")
        
    # 1. Load Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint.get("classes", [])
    num_classes = len(classes)
    
    print(f"Checkpoint Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Classes Count:    {num_classes}")
    print(f"Classes:          {', '.join(classes[:5])}... {classes[-1]}")
    print("-" * 95)
    
    # 2. Instantiate Model and Load Weights
    model = SignBaselineCNN(num_classes=num_classes, in_channels=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # 3. Load Dataset
    train_loader, val_loader, test_loader, canonical_classes = get_dataloaders(
        data_dir='data/splits',
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed
    )
    
    target_loader = val_loader if args.split == "val" else test_loader
    
    if args.subset_fraction < 1.0:
        np.random.seed(args.seed)
        ds = target_loader.dataset
        n_eval = int(len(ds) * args.subset_fraction)
        indices = np.random.choice(len(ds), size=n_eval, replace=False)
        target_loader = DataLoader(
            Subset(ds, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        print(f"Evaluating on Subsampled Split: {len(target_loader.dataset):,} / {len(ds):,} images")
    else:
        print(f"Evaluating on Full Split:       {len(target_loader.dataset):,} images")
        
    # 4. Inference Accumulation
    all_targets = []
    all_preds = []
    all_probs = []
    
    print("\nRunning deterministic inference forward pass...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(target_loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_targets.extend(targets.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 5. Global Metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    print("\n" + "=" * 95)
    print("GLOBAL PERFORMANCE METRICS SUMMARY:")
    print("=" * 95)
    print(f"Total Evaluated Samples:   {len(all_targets):,}")
    print(f"Overall Accuracy:          {acc*100:6.2f}%")
    print(f"Macro-Averaged Precision:  {macro_prec*100:6.2f}%")
    print(f"Macro-Averaged Recall:     {macro_rec*100:6.2f}%")
    print(f"Macro-Averaged F1-Score:   {macro_f1*100:6.2f}%")
    print(f"Weighted F1-Score:         {weighted_f1*100:6.2f}%")
    print("=" * 95)
    
    # 6. Class-by-Class Breakdown
    clf_dict = classification_report(all_targets, all_preds, target_names=classes, output_dict=True, zero_division=0)
    df_clf = pd.DataFrame(clf_dict).transpose()
    
    print("\nCLASS-BY-CLASS DETAILED PERFORMANCE BREAKDOWN:")
    print("-" * 95)
    print(f"{'Class Name':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<10}")
    print("-" * 95)
    
    for cls in classes:
        p = clf_dict[cls]['precision']
        r = clf_dict[cls]['recall']
        f = clf_dict[cls]['f1-score']
        s = int(clf_dict[cls]['support'])
        print(f"{cls:<12} | {p*100:<11.2f}% | {r*100:<11.2f}% | {f*100:<11.2f}% | {s:<10d}")
    print("-" * 95)
    
    # 7. Confusion Matrix & Top Error Pairs
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(classes)))
    # Row-normalized confusion matrix (recall per true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    
    # Identify top confused off-diagonal pairs
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": classes[i],
                    "pred_class": classes[j],
                    "count": int(cm[i, j]),
                    "percentage_of_true": float(cm_norm[i, j] * 100)
                })
                
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    top_k = confused_pairs[:10]
    
    print("\nTOP-10 MOST FREQUENT CONFUSED SIGN PAIRS (ERROR DIAGNOSTICS):")
    print("-" * 95)
    print(f"{'Rank':<6} | {'True Sign':<14} | {'Predicted Sign':<16} | {'Error Count':<14} | {'Error % of True':<16}")
    print("-" * 95)
    for rank, pair in enumerate(top_k, 1):
        print(f"#{rank:<5} | {pair['true_class']:<14} | {pair['pred_class']:<16} | {pair['count']:<14d} | {pair['percentage_of_true']:<15.1f}%")
    print("=" * 95)
    
    # 8. Render Visual Heatmap
    plot_confusion_matrix(cm_norm, classes, args.output_plot, top_k)
    
    # 9. Serialize JSON Output
    results_payload = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "evaluated_samples": len(all_targets),
            "classes_count": len(classes),
            "classes": classes
        },
        "global_metrics": {
            "accuracy": float(acc),
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1)
        },
        "top_confused_pairs": top_k,
        "classification_report": {k: v for k, v in clf_dict.items() if k in classes}
    }
    
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results_payload, f, indent=2)
    print(f"[SUCCESS] Saved detailed evaluation results to: {args.output_json}")


if __name__ == "__main__":
    run_evaluation()
