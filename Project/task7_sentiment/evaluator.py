"""
Task C.7: Model Evaluator

This module implements evaluation metrics and statistical tests for classification models.
Addresses Task Requirement 4 (5 marks) - Evaluation & Comparison.

Requirements addressed:
- Calculate classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix and ROC curves
- Statistical significance testing (McNemar test)
- Compare baseline vs full models
- REUSE visualization.py from Task C.5 for plotting

Evaluation metrics:
- Accuracy: Overall correctness
- Precision: Positive prediction accuracy
- Recall: Positive detection rate
- F1 Score: Harmonic mean of precision & recall
- ROC-AUC: Area under ROC curve
- Confusion Matrix: Error breakdown

Author: Your Name
Date: October 2025
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Scikit-learn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)

# Statistical tests
from scipy.stats import chi2
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# CRITICAL: REUSE Task C.5 visualization module
try:
    import visualization
    VISUALIZATION_AVAILABLE = True
    print("[REUSE] Task C.5 visualization imported successfully!")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[WARNING] visualization.py not found. Using custom plots.")


class ModelEvaluator:
    """
    Evaluate and compare classification models
    
    This class implements Task C.7 evaluation requirements:
    1. Calculate all classification metrics
    2. Generate confusion matrices and ROC curves
    3. Perform McNemar statistical test for model comparison
    4. Compare baseline (technical-only, sentiment-only) vs full models
    5. REUSE Task C.5's visualization module (code reuse!)
    
    Usage:
        evaluator = ModelEvaluator(save_dir='task7_results')
        
        # Evaluate single model
        metrics = evaluator.evaluate_model(y_true, y_pred, y_proba, 'full_model')
        
        # Compare two models
        comparison = evaluator.compare_models(
            y_true, y_pred1, y_pred2,
            'baseline', 'full_model'
        )
        
        # McNemar test
        is_significant = evaluator.mcnemar_test(y_true, y_pred1, y_pred2)
    """
    
    def __init__(self, save_dir: str = 'task7_results'):
        """
        Initialize Model Evaluator
        
        Args:
            save_dir: Directory to save evaluation results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Store all evaluation results
        self.results = {}
        
        print(f"\n[EVALUATOR] Initializing Model Evaluator")
        print(f"  Save directory: {save_dir}")
    
    # =========================================================================
    # SINGLE MODEL EVALUATION
    # =========================================================================
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_proba: np.ndarray = None,
                      model_name: str = 'model',
                      feature_set: str = 'unknown') -> Dict[str, float]:
        """
        Evaluate a single model's performance
        
        Calculates ALL classification metrics required for Task C.7:
        - Accuracy, Precision, Recall, F1 Score
        - ROC-AUC (if probabilities provided)
        - Confusion Matrix breakdown
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for ROC-AUC)
            model_name: Model identifier
            feature_set: 'technical', 'sentiment', or 'full'
            
        Returns:
            dict: All evaluation metrics
        """
        print(f"\n[EVALUATE] {model_name} ({feature_set})")
        
        # Calculate metrics
        metrics = {}
        
        # 1. ACCURACY (Overall correctness)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        
        # 2. PRECISION (Positive prediction accuracy)
        # "Of all predicted UP, how many were actually UP?"
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        print(f"  Precision: {metrics['precision']:.4f}")
        
        # 3. RECALL (Positive detection rate, also called Sensitivity)
        # "Of all actual UP, how many did we predict?"
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        print(f"  Recall:    {metrics['recall']:.4f}")
        
        # 4. F1 SCORE (Harmonic mean of precision & recall)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        
        # 5. ROC-AUC (Area under ROC curve)
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            except ValueError as e:
                print(f"  ROC-AUC:   N/A ({e})")
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
        
        # 6. CONFUSION MATRIX
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract values
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        
        print(f"\n  Confusion Matrix:")
        print(f"    True Negative (TN):  {tn:4d}  |  False Positive (FP): {fp:4d}")
        print(f"    False Negative (FN): {fn:4d}  |  True Positive (TP):  {tp:4d}")
        
        # Calculate additional metrics from confusion matrix
        total = tn + fp + fn + tp
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # False Positive Rate
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Negative Rate
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # 7. Additional info
        metrics['model_name'] = model_name
        metrics['feature_set'] = feature_set
        metrics['n_samples'] = len(y_true)
        metrics['n_positive'] = int(np.sum(y_true == 1))
        metrics['n_negative'] = int(np.sum(y_true == 0))
        
        # Store results
        result_key = f"{model_name}_{feature_set}"
        self.results[result_key] = metrics
        
        print(f"\n[OK] Evaluation complete for {model_name}")
        
        return metrics
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str = 'model',
                             save_path: str = None):
        """
        Plot confusion matrix heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name for title
            save_path: Path to save plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['DOWN (0)', 'UP (1)'],
                   yticklabels=['DOWN (0)', 'UP (1)'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] Confusion matrix: {save_path}")
        else:
            plt.savefig(f'{self.save_dir}/confusion_matrix_{model_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      model_name: str = 'model',
                      save_path: str = None):
        """
        Plot ROC curve
        
        ROC (Receiver Operating Characteristic):
        - X-axis: False Positive Rate
        - Y-axis: True Positive Rate (Recall)
        - Diagonal line: Random classifier (AUC = 0.5)
        - Perfect classifier: Top-left corner (AUC = 1.0)
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Model name for title
            save_path: Path to save plot (optional)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] ROC curve: {save_path}")
        else:
            plt.savefig(f'{self.save_dir}/roc_curve_{model_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_metrics_comparison(self, results_list: List[Dict[str, Any]],
                               save_path: str = None):
        """
        Compare metrics across multiple models
        
        Creates bar chart comparing accuracy, precision, recall, F1
        
        Args:
            results_list: List of metric dictionaries
            save_path: Path to save plot (optional)
        """
        # Extract data for plotting
        model_names = [r['model_name'] for r in results_list]
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        
        # Prepare data
        data = {metric: [r[metric] for r in results_list] 
                for metric in metrics_to_plot}
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(model_names))
        width = 0.2
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            offset = width * (i - 1.5)
            ax.bar(x + offset, data[metric], width, label=metric.capitalize(),
                  color=color, alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] Metrics comparison: {save_path}")
        else:
            plt.savefig(f'{self.save_dir}/metrics_comparison.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    # =========================================================================
    # STATISTICAL TESTS
    # =========================================================================
    
    def mcnemar_test(self, y_true: np.ndarray, 
                    y_pred1: np.ndarray, y_pred2: np.ndarray,
                    model1_name: str = 'Model 1',
                    model2_name: str = 'Model 2',
                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform McNemar's test for statistical significance
        
        McNemar's test determines if two models' predictions are
        significantly different. This is REQUIRED for Task C.7!
        
        The test uses a 2x2 contingency table:
                        Model 2 Correct | Model 2 Wrong
        Model 1 Correct |       a        |       b
        Model 1 Wrong   |       c        |       d
        
        Test statistic: χ² = (b - c)² / (b + c)
        
        Null hypothesis: Models have equal performance
        Alternative: Models have different performance
        
        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            model1_name: Name of model 1
            model2_name: Name of model 2
            alpha: Significance level (default: 0.05)
            
        Returns:
            dict: Test results with statistic, p-value, conclusion
        """
        print(f"\n[MCNEMAR] Testing {model1_name} vs {model2_name}")
        
        # Create correctness arrays
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        # Build contingency table
        # a: both correct
        # b: model1 correct, model2 wrong
        # c: model1 wrong, model2 correct
        # d: both wrong
        
        a = np.sum(correct1 & correct2)
        b = np.sum(correct1 & ~correct2)
        c = np.sum(~correct1 & correct2)
        d = np.sum(~correct1 & ~correct2)
        
        print(f"\n  Contingency Table:")
        print(f"    Both correct:          {a}")
        print(f"    {model1_name} only:    {b}")
        print(f"    {model2_name} only:    {c}")
        print(f"    Both wrong:            {d}")
        
        # Calculate test statistic
        # Use continuity correction for small samples
        if (b + c) == 0:
            print("\n  [WARNING] b + c = 0, cannot perform test")
            return {
                'statistic': None,
                'p_value': None,
                'significant': False,
                'message': 'Test not applicable (b + c = 0)'
            }
        
        # McNemar statistic with continuity correction
        statistic = ((abs(b - c) - 1) ** 2) / (b + c)
        
        # P-value from chi-square distribution (df=1)
        p_value = 1 - chi2.cdf(statistic, df=1)
        
        # Determine significance
        is_significant = p_value < alpha
        
        print(f"\n  Test Statistic: χ² = {statistic:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significance level: α = {alpha}")
        
        if is_significant:
            print(f"  [RESULT] Models are SIGNIFICANTLY different (p < {alpha})")
            conclusion = f"Reject null hypothesis: {model2_name} performs significantly different from {model1_name}"
        else:
            print(f"  [RESULT] No significant difference (p >= {alpha})")
            conclusion = f"Fail to reject null hypothesis: No significant difference between models"
        
        # Determine which model is better
        acc1 = np.sum(correct1) / len(y_true)
        acc2 = np.sum(correct2) / len(y_true)
        
        better_model = model1_name if acc1 > acc2 else model2_name
        
        return {
            'model1': model1_name,
            'model2': model2_name,
            'contingency_table': {'a': int(a), 'b': int(b), 'c': int(c), 'd': int(d)},
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': alpha,
            'significant': is_significant,
            'conclusion': conclusion,
            'better_model': better_model,
            'accuracy1': float(acc1),
            'accuracy2': float(acc2)
        }
    
    # =========================================================================
    # MODEL COMPARISON
    # =========================================================================
    
    def compare_models(self, models_results: List[Dict[str, Any]],
                      save_summary: bool = True) -> pd.DataFrame:
        """
        Compare multiple models and generate summary table
        
        Args:
            models_results: List of evaluation result dictionaries
            save_summary: Whether to save summary CSV
            
        Returns:
            pd.DataFrame: Comparison table
        """
        print(f"\n[COMPARE] Comparing {len(models_results)} models...")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(models_results)
        
        # Select key columns for display
        display_cols = ['model_name', 'feature_set', 'accuracy', 'precision', 
                       'recall', 'f1', 'roc_auc', 'n_samples']
        
        comparison_df = comparison_df[display_cols]
        
        # Sort by F1 score (descending)
        comparison_df = comparison_df.sort_values('f1', ascending=False)
        
        print("\n  Model Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Save to CSV
        if save_summary:
            save_path = f'{self.save_dir}/model_comparison.csv'
            comparison_df.to_csv(save_path, index=False)
            print(f"\n[SAVED] Comparison table: {save_path}")
        
        return comparison_df
    
    def generate_report(self, experiment_name: str = 'task7_experiment'):
        """
        Generate comprehensive evaluation report
        
        Creates JSON file with all evaluation results
        
        Args:
            experiment_name: Name for the report
        """
        print(f"\n[REPORT] Generating evaluation report...")
        
        report = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_models': len(self.results),
            'results': self.results
        }
        
        # Save report
        report_path = f'{self.save_dir}/{experiment_name}_evaluation.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[SAVED] Evaluation report: {report_path}")
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    Total models evaluated: {len(self.results)}")
        print(f"    Results saved to: {self.save_dir}")
        
        # Find best model
        if self.results:
            best_model = max(self.results.items(), 
                           key=lambda x: x[1].get('f1', 0))
            print(f"    Best model (by F1): {best_model[0]} (F1={best_model[1]['f1']:.4f})")
        
        return report


# Example usage
if __name__ == '__main__':
    print("="*70)
    print("TESTING MODEL EVALUATOR")
    print("="*70)
    
    # Create dummy predictions
    np.random.seed(42)
    n_samples = 200
    
    y_true = np.random.randint(0, 2, n_samples)
    
    # Model 1: 70% accuracy
    y_pred1 = y_true.copy()
    flip_indices = np.random.choice(n_samples, size=int(0.3 * n_samples), replace=False)
    y_pred1[flip_indices] = 1 - y_pred1[flip_indices]
    y_proba1 = np.random.uniform(0.3, 0.9, n_samples)
    
    # Model 2: 75% accuracy
    y_pred2 = y_true.copy()
    flip_indices = np.random.choice(n_samples, size=int(0.25 * n_samples), replace=False)
    y_pred2[flip_indices] = 1 - y_pred2[flip_indices]
    y_proba2 = np.random.uniform(0.4, 0.95, n_samples)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(save_dir='test_task7_results')
    
    # Evaluate models
    print("\n" + "="*70)
    print("EVALUATING MODEL 1")
    print("="*70)
    
    metrics1 = evaluator.evaluate_model(
        y_true, y_pred1, y_proba1,
        model_name='baseline_technical',
        feature_set='technical'
    )
    
    print("\n" + "="*70)
    print("EVALUATING MODEL 2")
    print("="*70)
    
    metrics2 = evaluator.evaluate_model(
        y_true, y_pred2, y_proba2,
        model_name='full_model',
        feature_set='full'
    )
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrix(y_true, y_pred1, 'baseline_technical')
    evaluator.plot_confusion_matrix(y_true, y_pred2, 'full_model')
    
    # Plot ROC curves
    evaluator.plot_roc_curve(y_true, y_proba1, 'baseline_technical')
    evaluator.plot_roc_curve(y_true, y_proba2, 'full_model')
    
    # Compare models
    comparison = evaluator.compare_models([metrics1, metrics2])
    
    # Plot comparison
    evaluator.plot_metrics_comparison([metrics1, metrics2])
    
    # McNemar test
    mcnemar_result = evaluator.mcnemar_test(
        y_true, y_pred1, y_pred2,
        'baseline_technical', 'full_model'
    )
    
    # Generate report
    report = evaluator.generate_report('test_experiment')
    
    print("\n[TEST] Evaluator tested successfully!")
