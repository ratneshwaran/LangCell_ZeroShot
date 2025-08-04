#!/usr/bin/env python3
"""
Simple ROC and Precision-Recall Curves for LangCell Results

A simplified script to generate ROC and PR curves from LangCell classification results.
This can be easily run from a Jupyter notebook.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
import json

def load_results(results_path, type2text_path=None):
    """Load results and cell type information."""
    # Load results
    results = torch.load(results_path, map_location='cpu')
    labels = results['labels'].numpy()
    preds = results['preds'].numpy()
    
    # Load cell type information if available
    cell_types = None
    if type2text_path:
        try:
            with open(type2text_path, 'r') as f:
                type2text = json.load(f)
            cell_types = list(type2text.keys())
        except:
            print("Warning: Could not load type2text file")
    
    # If no cell types provided, use numeric labels
    if cell_types is None:
        n_classes = max(labels.max(), preds.max()) + 1
        cell_types = [f"Class_{i}" for i in range(n_classes)]
    
    # Get prediction scores
    if 'logits' in results:
        scores = results['logits'].numpy()
    elif 'sim_logits' in results and 'ctm_logits' in results:
        # Combine similarity and CTM logits
        alpha = 0.1
        sim_logits = results['sim_logits'].numpy()
        ctm_logits = results['ctm_logits'].numpy()
        scores = alpha * sim_logits + (1 - alpha) * ctm_logits
    else:
        raise ValueError("No prediction scores found in results")
    
    return labels, preds, scores, cell_types

def plot_roc_curves(labels, scores, cell_types, figsize=(15, 5)):
    """Plot ROC curves for all classes."""
    n_classes = len(cell_types)
    
    # Binarize labels
    labels_bin = label_binarize(labels, classes=range(n_classes))
    if n_classes == 2:
        labels_bin = np.hstack([1 - labels_bin, labels_bin])
    
    # Compute ROC curves
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Individual classes
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax1.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{cell_types[i]} (AUC = {roc_auc[i]:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - Per Class')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Micro average
    ax2.plot(fpr["micro"], tpr["micro"],
            label=f'micro-average (AUC = {roc_auc["micro"]:.3f})',
            color='deeppink', linestyle=':', linewidth=4)
    ax2.plot([0, 1], [0, 1], 'k--', lw=2)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve - Micro Average')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return roc_auc

def plot_pr_curves(labels, scores, cell_types, figsize=(15, 5)):
    """Plot Precision-Recall curves for all classes."""
    n_classes = len(cell_types)
    
    # Binarize labels
    labels_bin = label_binarize(labels, classes=range(n_classes))
    if n_classes == 2:
        labels_bin = np.hstack([1 - labels_bin, labels_bin])
    
    # Compute PR curves
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels_bin[:, i], scores[:, i])
        average_precision[i] = average_precision_score(labels_bin[:, i], scores[:, i])
    
    # Compute micro-average PR
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        labels_bin.ravel(), scores.ravel()
    )
    average_precision["micro"] = average_precision_score(labels_bin.ravel(), scores.ravel())
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Individual classes
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax1.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{cell_types[i]} (AP = {average_precision[i]:.3f})')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves - Per Class')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Micro average
    ax2.plot(recall["micro"], precision["micro"],
            label=f'micro-average (AP = {average_precision["micro"]:.3f})',
            color='deeppink', linestyle=':', linewidth=4)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve - Micro Average')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return average_precision

def plot_performance_summary(labels, scores, cell_types, figsize=(12, 8)):
    """Plot a summary of performance metrics."""
    n_classes = len(cell_types)
    
    # Binarize labels
    labels_bin = label_binarize(labels, classes=range(n_classes))
    if n_classes == 2:
        labels_bin = np.hstack([1 - labels_bin, labels_bin])
    
    # Calculate metrics
    roc_auc_scores = []
    ap_scores = []
    support = []
    
    for i in range(n_classes):
        roc_auc_scores.append(roc_auc_score(labels_bin[:, i], scores[:, i]))
        ap_scores.append(average_precision_score(labels_bin[:, i], scores[:, i]))
        support.append(np.sum(labels_bin[:, i]))
    
    # Overall metrics
    micro_roc_auc = roc_auc_score(labels_bin.ravel(), scores.ravel())
    micro_ap = average_precision_score(labels_bin.ravel(), scores.ravel())
    
    # Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # ROC AUC by class
    bars1 = ax1.bar(range(n_classes), roc_auc_scores)
    ax1.set_title('ROC AUC by Class')
    ax1.set_ylabel('ROC AUC')
    ax1.set_xticks(range(n_classes))
    ax1.set_xticklabels(cell_types, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    for bar, value in zip(bars1, roc_auc_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # AP by class
    bars2 = ax2.bar(range(n_classes), ap_scores)
    ax2.set_title('Average Precision by Class')
    ax2.set_ylabel('Average Precision')
    ax2.set_xticks(range(n_classes))
    ax2.set_xticklabels(cell_types, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    for bar, value in zip(bars2, ap_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Support by class
    bars3 = ax3.bar(range(n_classes), support)
    ax3.set_title('Number of Samples by Class')
    ax3.set_ylabel('Support')
    ax3.set_xticks(range(n_classes))
    ax3.set_xticklabels(cell_types, rotation=45, ha='right')
    for bar, value in zip(bars3, support):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                f'{value}', ha='center', va='bottom', fontsize=8)
    
    # Overall metrics
    overall_metrics = {
        'Micro ROC AUC': micro_roc_auc,
        'Micro AP': micro_ap
    }
    
    ax4.bar(range(len(overall_metrics)), list(overall_metrics.values()))
    ax4.set_title('Overall Performance Metrics')
    ax4.set_ylabel('Score')
    ax4.set_xticks(range(len(overall_metrics)))
    ax4.set_xticklabels(overall_metrics.keys(), rotation=45, ha='right')
    ax4.set_ylim(0, 1)
    for i, (key, value) in enumerate(overall_metrics.items()):
        ax4.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'roc_auc': roc_auc_scores,
        'ap': ap_scores,
        'support': support,
        'micro_roc_auc': micro_roc_auc,
        'micro_ap': micro_ap
    }

def run_analysis(results_path, type2text_path=None):
    """Run the complete analysis."""
    print("Loading results...")
    labels, preds, scores, cell_types = load_results(results_path, type2text_path)
    
    print(f"Dataset: {len(labels)} samples, {len(cell_types)} classes")
    print(f"Classes: {cell_types}")
    print()
    
    print("Generating ROC curves...")
    roc_auc = plot_roc_curves(labels, scores, cell_types)
    
    print("Generating Precision-Recall curves...")
    ap_scores = plot_pr_curves(labels, scores, cell_types)
    
    print("Generating performance summary...")
    metrics = plot_performance_summary(labels, scores, cell_types)
    
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Micro ROC AUC: {metrics['micro_roc_auc']:.4f}")
    print(f"Micro Average Precision: {metrics['micro_ap']:.4f}")
    print()
    print("Per-class metrics:")
    for i, cell_type in enumerate(cell_types):
        print(f"  {cell_type}: ROC AUC = {roc_auc[i]:.4f}, AP = {ap_scores[i]:.4f}")
    print("="*50)
    
    return metrics

# Example usage:
if __name__ == "__main__":
    # Configuration
    results_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/results_pancreas.pt"
    type2text_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/type2text_pancreas.json"
    
    # Run analysis
    metrics = run_analysis(results_path, type2text_path) 