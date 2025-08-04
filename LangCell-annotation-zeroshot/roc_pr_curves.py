#!/usr/bin/env python3
"""
ROC and Precision-Recall Curves for LangCell Classification Results

This script generates comprehensive ROC and Precision-Recall curves for 
LangCell cell type classification results, including:
- Per-class ROC curves
- Macro and micro averaged ROC curves
- Per-class Precision-Recall curves
- Macro and micro averaged Precision-Recall curves
- Performance metrics summary
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path
import json

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LangCellEvaluator:
    def __init__(self, results_path, type2text_path=None, output_dir="plots"):
        """
        Initialize the evaluator with results and cell type information.
        
        Args:
            results_path: Path to the saved results .pt file
            type2text_path: Path to type2text JSON file (optional)
            output_dir: Directory to save plots
        """
        self.results_path = results_path
        self.type2text_path = type2text_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        self.results = torch.load(results_path, map_location='cpu')
        self.labels = self.results['labels'].numpy()
        self.preds = self.results['preds'].numpy()
        
        # Load cell type information if available
        self.cell_types = None
        if type2text_path and os.path.exists(type2text_path):
            with open(type2text_path, 'r') as f:
                type2text = json.load(f)
            self.cell_types = list(type2text.keys())
        
        # If no cell types provided, use numeric labels
        if self.cell_types is None:
            n_classes = max(self.labels.max(), self.preds.max()) + 1
            self.cell_types = [f"Class_{i}" for i in range(n_classes)]
        
        self.n_classes = len(self.cell_types)
        
        # Get prediction scores
        if 'logits' in self.results:
            self.scores = self.results['logits'].numpy()
        elif 'sim_logits' in self.results and 'ctm_logits' in self.results:
            # Combine similarity and CTM logits
            alpha = 0.1
            sim_logits = self.results['sim_logits'].numpy()
            ctm_logits = self.results['ctm_logits'].numpy()
            self.scores = alpha * sim_logits + (1 - alpha) * ctm_logits
        else:
            raise ValueError("No prediction scores found in results")
    
    def prepare_data(self):
        """Prepare data for ROC and PR curve analysis."""
        # Binarize labels for multi-class ROC
        self.labels_bin = label_binarize(
            self.labels, 
            classes=range(self.n_classes)
        )
        
        # Handle case where we have only 2 classes
        if self.n_classes == 2:
            self.labels_bin = np.hstack([1 - self.labels_bin, self.labels_bin])
    
    def compute_roc_curves(self):
        """Compute ROC curves for each class and overall."""
        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()
        
        # Compute ROC curve and ROC area for each class
        for i in range(self.n_classes):
            self.fpr[i], self.tpr[i], _ = roc_curve(
                self.labels_bin[:, i], 
                self.scores[:, i]
            )
            self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        self.fpr["micro"], self.tpr["micro"], _ = roc_curve(
            self.labels_bin.ravel(), 
            self.scores.ravel()
        )
        self.roc_auc["micro"] = auc(self.fpr["micro"], self.tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.n_classes)]))
        
        # Then interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, self.fpr[i], self.tpr[i])
        
        # Finally average it and compute AUC
        mean_tpr /= self.n_classes
        self.fpr["macro"] = all_fpr
        self.tpr["macro"] = mean_tpr
        self.roc_auc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])
    
    def compute_pr_curves(self):
        """Compute Precision-Recall curves for each class and overall."""
        self.precision = dict()
        self.recall = dict()
        self.average_precision = dict()
        
        # Compute Precision-Recall curve and AP for each class
        for i in range(self.n_classes):
            self.precision[i], self.recall[i], _ = precision_recall_curve(
                self.labels_bin[:, i], 
                self.scores[:, i]
            )
            self.average_precision[i] = average_precision_score(
                self.labels_bin[:, i], 
                self.scores[:, i]
            )
        
        # Compute micro-average Precision-Recall curve and AP
        self.precision["micro"], self.recall["micro"], _ = precision_recall_curve(
            self.labels_bin.ravel(), 
            self.scores.ravel()
        )
        self.average_precision["micro"] = average_precision_score(
            self.labels_bin.ravel(), 
            self.scores.ravel()
        )
        
        # Compute macro-average Precision-Recall curve and AP
        # First aggregate all recall values
        all_recall = np.unique(np.concatenate([self.recall[i] for i in range(self.n_classes)]))
        
        # Then interpolate all PR curves at these points
        mean_precision = np.zeros_like(all_recall)
        for i in range(self.n_classes):
            mean_precision += np.interp(all_recall, self.recall[i], self.precision[i])
        
        # Finally average it and compute AP
        mean_precision /= self.n_classes
        self.recall["macro"] = all_recall
        self.precision["macro"] = mean_precision
        self.average_precision["macro"] = np.trapz(mean_precision, all_recall)
    
    def plot_roc_curves(self, figsize=(15, 10)):
        """Plot ROC curves for all classes and averages."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot individual class ROC curves
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_classes))
        for i, color in zip(range(self.n_classes), colors):
            ax1.plot(
                self.fpr[i], self.tpr[i], color=color, lw=2,
                label=f'{self.cell_types[i]} (AUC = {self.roc_auc[i]:.3f})'
            )
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=2)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Per Class')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot macro and micro averages
        ax2.plot(
            self.fpr["micro"], self.tpr["micro"],
            label=f'micro-average ROC curve (AUC = {self.roc_auc["micro"]:.3f})',
            color='deeppink', linestyle=':', linewidth=4
        )
        ax2.plot(
            self.fpr["macro"], self.tpr["macro"],
            label=f'macro-average ROC curve (AUC = {self.roc_auc["macro"]:.3f})',
            color='navy', linestyle=':', linewidth=4
        )
        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves - Macro/Micro Average')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pr_curves(self, figsize=(15, 10)):
        """Plot Precision-Recall curves for all classes and averages."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot individual class PR curves
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_classes))
        for i, color in zip(range(self.n_classes), colors):
            ax1.plot(
                self.recall[i], self.precision[i], color=color, lw=2,
                label=f'{self.cell_types[i]} (AP = {self.average_precision[i]:.3f})'
            )
        
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curves - Per Class')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot macro and micro averages
        ax2.plot(
            self.recall["micro"], self.precision["micro"],
            label=f'micro-average PR curve (AP = {self.average_precision["micro"]:.3f})',
            color='deeppink', linestyle=':', linewidth=4
        )
        ax2.plot(
            self.recall["macro"], self.precision["macro"],
            label=f'macro-average PR curve (AP = {self.average_precision["macro"]:.3f})',
            color='navy', linestyle=':', linewidth=4
        )
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves - Macro/Micro Average')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pr_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_summary(self, figsize=(12, 8)):
        """Plot a summary of performance metrics for each class."""
        # Calculate metrics for each class
        metrics = []
        for i in range(self.n_classes):
            metrics.append({
                'Class': self.cell_types[i],
                'ROC AUC': self.roc_auc[i],
                'Average Precision': self.average_precision[i],
                'Support': np.sum(self.labels_bin[:, i])
            })
        
        df_metrics = pd.DataFrame(metrics)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # ROC AUC by class
        bars1 = ax1.bar(range(len(df_metrics)), df_metrics['ROC AUC'])
        ax1.set_title('ROC AUC by Class')
        ax1.set_ylabel('ROC AUC')
        ax1.set_xticks(range(len(df_metrics)))
        ax1.set_xticklabels(df_metrics['Class'], rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        # Add value labels on bars
        for bar, value in zip(bars1, df_metrics['ROC AUC']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Average Precision by class
        bars2 = ax2.bar(range(len(df_metrics)), df_metrics['Average Precision'])
        ax2.set_title('Average Precision by Class')
        ax2.set_ylabel('Average Precision')
        ax2.set_xticks(range(len(df_metrics)))
        ax2.set_xticklabels(df_metrics['Class'], rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        # Add value labels on bars
        for bar, value in zip(bars2, df_metrics['Average Precision']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Support by class
        bars3 = ax3.bar(range(len(df_metrics)), df_metrics['Support'])
        ax3.set_title('Number of Samples by Class')
        ax3.set_ylabel('Support')
        ax3.set_xticks(range(len(df_metrics)))
        ax3.set_xticklabels(df_metrics['Class'], rotation=45, ha='right')
        # Add value labels on bars
        for bar, value in zip(bars3, df_metrics['Support']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                    f'{value}', ha='center', va='bottom', fontsize=8)
        
        # Overall metrics
        overall_metrics = {
            'Micro ROC AUC': self.roc_auc["micro"],
            'Macro ROC AUC': self.roc_auc["macro"],
            'Micro AP': self.average_precision["micro"],
            'Macro AP': self.average_precision["macro"]
        }
        
        ax4.bar(range(len(overall_metrics)), list(overall_metrics.values()))
        ax4.set_title('Overall Performance Metrics')
        ax4.set_ylabel('Score')
        ax4.set_xticks(range(len(overall_metrics)))
        ax4.set_xticklabels(overall_metrics.keys(), rotation=45, ha='right')
        ax4.set_ylim(0, 1)
        # Add value labels on bars
        for i, (key, value) in enumerate(overall_metrics.items()):
            ax4.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive performance report."""
        # Calculate classification report
        report = classification_report(
            self.labels, self.preds, 
            target_names=self.cell_types,
            output_dict=True
        )
        
        # Create summary
        summary = {
            'Overall Metrics': {
                'Micro ROC AUC': self.roc_auc["micro"],
                'Macro ROC AUC': self.roc_auc["macro"],
                'Micro Average Precision': self.average_precision["micro"],
                'Macro Average Precision': self.average_precision["macro"]
            },
            'Per-Class Metrics': report,
            'Confusion Matrix': confusion_matrix(self.labels, self.preds).tolist()
        }
        
        # Save report
        import json
        with open(self.output_dir / 'performance_report.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("=" * 60)
        print("LANGCELL CLASSIFICATION PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Dataset: {Path(self.results_path).stem}")
        print(f"Number of classes: {self.n_classes}")
        print(f"Total samples: {len(self.labels)}")
        print()
        print("OVERALL METRICS:")
        print(f"  Micro ROC AUC: {self.roc_auc['micro']:.4f}")
        print(f"  Macro ROC AUC: {self.roc_auc['macro']:.4f}")
        print(f"  Micro Average Precision: {self.average_precision['micro']:.4f}")
        print(f"  Macro Average Precision: {self.average_precision['macro']:.4f}")
        print()
        print("PER-CLASS ROC AUC:")
        for i, cell_type in enumerate(self.cell_types):
            print(f"  {cell_type}: {self.roc_auc[i]:.4f}")
        print()
        print("PER-CLASS AVERAGE PRECISION:")
        for i, cell_type in enumerate(self.cell_types):
            print(f"  {cell_type}: {self.average_precision[i]:.4f}")
        print("=" * 60)
        
        return summary
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Preparing data...")
        self.prepare_data()
        
        print("Computing ROC curves...")
        self.compute_roc_curves()
        
        print("Computing Precision-Recall curves...")
        self.compute_pr_curves()
        
        print("Generating plots...")
        self.plot_roc_curves()
        self.plot_pr_curves()
        self.plot_performance_summary()
        
        print("Generating report...")
        self.generate_report()
        
        print(f"Analysis complete! Results saved to: {self.output_dir}")


def main():
    """Main function to run the analysis."""
    # Configuration
    results_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/results_pancreas.pt"
    type2text_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/LangCell/your_data/type2text_pancreas.json"
    output_dir = "LangCell/LangCell-annotation-zeroshot/plots"
    
    # Create evaluator and run analysis
    evaluator = LangCellEvaluator(
        results_path=results_path,
        type2text_path=type2text_path,
        output_dir=output_dir
    )
    
    evaluator.run_full_analysis()


if __name__ == "__main__":
    main() 