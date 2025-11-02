#!/usr/bin/env python
"""
Task C.7: Advanced Evaluation & Visualization
Generate comprehensive evaluation report with visualizations

This script creates professional visualizations and reports for academic presentation:
1. Confusion matrices (heatmaps for top 6 models)
2. Model comparison charts (4-subplot comprehensive analysis)
3. Feature importance plots (for tree-based models)
4. Detailed classification reports (text file)
5. Executive summary (f    # Generate visualizations (3 plots)
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_confusion_matrices(all_results)  # Top 6 models confusion matrices
    plot_model_comparison(all_results)  # 4-subplot comprehensive comparison
    
    if models:
        # Plot feature importance for tree-based models
        # Complete list of all possible features (14 technical + 2 sentiment)
        plot_feature_importance(models, 
                               ['return_1d', 'return_5d', 'return_20d', 'volatility_5d', 
                                'volatility_20d', 'ma_5', 'ma_20', 'ma_50', 'rsi', 'macd',
                                'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
                                'sentiment_mean', 'sentiment_std'])ing)

These visualizations help demonstrate:
- Model performance comparison
- Algorithm effectiveness
- Feature importance (which indicators matter most)
- Baseline vs sentiment comparison

Perfect for including in academic reports and presentations!

Author: Anh Vu Le
Date: November 2025
Course: COS30018 - Intelligent Systems
"""

import warnings
warnings.filterwarnings('ignore')  # Suppress matplotlib/seaborn warnings

import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Statistical visualization (built on matplotlib)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)  # Evaluation metrics
from sklearn.model_selection import cross_val_score  # Cross-validation
import joblib  # Load saved models
import os  # File operations
import json  # JSON handling

# Set professional plotting style
sns.set_style("whitegrid")  # Clean grid background
plt.rcParams['figure.figsize'] = (12, 8)  # Default figure size


def plot_confusion_matrices(all_results, save_dir='task7_results/plots'):
    """
    Plot confusion matrices for top 6 models
    
    Confusion matrix shows:
    - True Negatives (TN): Correctly predicted DOWN
    - False Positives (FP): Predicted UP but actually DOWN
    - False Negatives (FN): Predicted DOWN but actually UP  
    - True Positives (TP): Correctly predicted UP
    
    Visual heatmap makes it easy to see:
    - Which models have fewer errors (lighter off-diagonal)
    - Which models are better at detecting UPs vs DOWNs
    
    Args:
        all_results: List of evaluation result dictionaries
        save_dir: Directory to save plots
        
    Returns:
        None (saves confusion_matrices.png)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get top 6 models by F1 score (best performers)
    top_models = sorted(all_results, key=lambda x: x['f1'], reverse=True)[:6]
    
    # Create 2×3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confusion Matrices - Top 6 Models', fontsize=16, fontweight='bold')
    
    # Plot each confusion matrix
    for idx, result in enumerate(top_models):
        ax = axes[idx // 3, idx % 3]  # Calculate subplot position
        cm = np.array(result['confusion_matrix'])  # Get confusion matrix
        
        # Create heatmap with annotations
        # annot=True shows numbers in each cell
        # fmt='d' formats as integers
        # cmap='Blues' uses blue color scheme (darker = more samples)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
        
        # Add title with model name and F1 score
        title = f"{result['model']}\nF1={result['f1']:.3f}"
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()  # Adjust spacing
    plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved confusion matrices to {save_dir}/confusion_matrices.png")
    plt.close()  # Free memory


def plot_roc_curves(models_data, save_dir='task7_results/plots'):
    """Plot ROC curves for all models"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
    
    for idx, (model_name, data) in enumerate(models_data.items()):
        y_test = data['y_test']
        y_pred_proba = data.get('y_pred_proba')
        
        if y_pred_proba is not None and len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, color=colors[idx], lw=2, 
                    label=f'{model_name} (AUC={auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved ROC curves to {save_dir}/roc_curves.png")
    plt.close()


def plot_model_comparison(all_results, save_dir='task7_results/plots'):
    """
    Plot comprehensive model comparison (4 subplots)
    
    This creates a professional 4-panel figure showing:
    1. F1 Score ranking (horizontal bar chart)
    2. Accuracy vs F1 scatter (shows if models are balanced)
    3. Precision vs Recall trade-off (important for imbalanced data)
    4. Average metrics by feature set (technical vs sentiment vs combined)
    
    Colors indicate feature sets:
    - Red: technical_only (baseline)
    - Green: sentiment_only (sentiment experiment)
    - Blue: combined (best expected)
    
    Args:
        all_results: List of evaluation result dictionaries
        save_dir: Directory to save plots
        
    Returns:
        None (saves model_comparison.png)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'confusion_matrix'} 
                       for r in all_results])
    df = df.sort_values('f1', ascending=True)  # Sort for horizontal bar chart
    
    # Create 2×2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. F1 Score comparison (horizontal bar chart)
    # Color bars by feature set to show baseline vs sentiment comparison
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if 'sentiment_only' in m else '#3498db' if 'combined' in m else '#e74c3c' 
              for m in df['model']]  # Green=sentiment, Blue=combined, Red=technical
    ax1.barh(df['model'], df['f1'], color=colors)
    ax1.set_xlabel('F1 Score', fontweight='bold')
    ax1.set_title('F1 Score by Model', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)  # Vertical grid lines
    
    # 2. Accuracy vs F1 scatter plot
    # Points near diagonal = balanced models (accuracy ≈ F1)
    # Points below diagonal = F1 < accuracy (precision/recall imbalance)
    ax2 = axes[0, 1]
    feature_colors = {'technical_only': '#e74c3c', 'sentiment_only': '#2ecc71', 'combined': '#3498db'}
    for feat, group in df.groupby('feature_set'):
        ax2.scatter(group['accuracy'], group['f1'], 
                   label=feat, s=100, alpha=0.7, color=feature_colors[feat])
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Diagonal reference line
    ax2.set_xlabel('Accuracy', fontweight='bold')
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('Accuracy vs F1 Score', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Precision vs Recall trade-off
    # Upper-right is ideal (high precision AND high recall)
    # Shows which models sacrifice precision for recall or vice versa
    ax3 = axes[1, 0]
    for feat, group in df.groupby('feature_set'):
        ax3.scatter(group['recall'], group['precision'], 
                   label=feat, s=100, alpha=0.7, color=feature_colors[feat])
    ax3.set_xlabel('Recall', fontweight='bold')
    ax3.set_ylabel('Precision', fontweight='bold')
    ax3.set_title('Precision vs Recall Trade-off', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Average metrics by feature set (bar chart)
    # This clearly shows: Does sentiment improve performance?
    ax4 = axes[1, 1]
    metrics_by_feat = df.groupby('feature_set')[['accuracy', 'precision', 'recall', 'f1']].mean()
    metrics_by_feat.plot(kind='bar', ax=ax4, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    ax4.set_xlabel('Feature Set', fontweight='bold')
    ax4.set_ylabel('Score', fontweight='bold')
    ax4.set_title('Average Metrics by Feature Set', fontweight='bold')
    ax4.legend(['Accuracy', 'Precision', 'Recall', 'F1'])
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved model comparison to {save_dir}/model_comparison.png")
    plt.close()


def plot_feature_importance(models, feature_names, save_dir='task7_results/plots'):
    """
    Plot feature importance for tree-based models
    
    Feature importance shows which indicators contribute most to predictions:
    - High importance = model relies heavily on this feature
    - Low importance = feature doesn't help much
    
    This helps answer:
    - Which technical indicators matter most?
    - Do sentiment features rank high?
    - Can we simplify by removing low-importance features?
    
    Only works for tree-based models (Random Forest, XGBoost, Gradient Boosting)
    because they have built-in feature_importances_ attribute.
    
    Args:
        models: Dictionary of trained models
        feature_names: List of all feature names
        save_dir: Directory to save plots
        
    Returns:
        None (saves feature_importance.png)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get Random Forest and XGBoost models (have feature importance)
    rf_models = {k: v for k, v in models.items() if 'random_forest' in k}
    xgb_models = {k: v for k, v in models.items() if 'xgboost' in k}
    
    if not rf_models and not xgb_models:
        print("[PLOT] No tree-based models found for feature importance")
        return
    
    fig, axes = plt.subplots(len(rf_models) + len(xgb_models), 1, 
                            figsize=(12, 4 * (len(rf_models) + len(xgb_models))))
    
    if len(rf_models) + len(xgb_models) == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # Plot RF feature importance
    for model_name, model_data in rf_models.items():
        model = model_data['model']
        features = model_data['features']
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:15]  # Top 15
        
        axes[ax_idx].barh(range(len(indices)), importance[indices], color='#2ecc71')
        axes[ax_idx].set_yticks(range(len(indices)))
        axes[ax_idx].set_yticklabels([features[i] for i in indices])
        axes[ax_idx].set_xlabel('Importance', fontweight='bold')
        axes[ax_idx].set_title(f'Feature Importance - {model_name}', fontweight='bold')
        axes[ax_idx].invert_yaxis()
        axes[ax_idx].grid(axis='x', alpha=0.3)
        ax_idx += 1
    
    # Plot XGBoost feature importance
    for model_name, model_data in xgb_models.items():
        model = model_data['model']
        features = model_data['features']
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:15]  # Top 15
        
        axes[ax_idx].barh(range(len(indices)), importance[indices], color='#3498db')
        axes[ax_idx].set_yticks(range(len(indices)))
        axes[ax_idx].set_yticklabels([features[i] for i in indices])
        axes[ax_idx].set_xlabel('Importance', fontweight='bold')
        axes[ax_idx].set_title(f'Feature Importance - {model_name}', fontweight='bold')
        axes[ax_idx].invert_yaxis()
        axes[ax_idx].grid(axis='x', alpha=0.3)
        ax_idx += 1
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved feature importance to {save_dir}/feature_importance.png")
    plt.close()


def generate_classification_reports(all_results, save_dir='task7_results'):
    """Generate detailed classification reports"""
    os.makedirs(save_dir, exist_ok=True)
    
    report_file = f'{save_dir}/classification_reports.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORTS\n")
        f.write("="*80 + "\n\n")
        
        for result in sorted(all_results, key=lambda x: x['f1'], reverse=True):
            f.write(f"\nModel: {result['model']}\n")
            f.write(f"Feature Set: {result['feature_set']}\n")
            f.write(f"Algorithm: {result['algorithm']}\n")
            f.write("-" * 60 + "\n")
            
            # Metrics summary
            f.write(f"Accuracy:  {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall:    {result['recall']:.4f}\n")
            f.write(f"F1 Score:  {result['f1']:.4f}\n")
            
            # Confusion matrix
            cm = np.array(result['confusion_matrix'])
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"                Predicted DOWN  Predicted UP\n")
            f.write(f"Actual DOWN:    {cm[0,0]:6d}         {cm[0,1]:6d}\n")
            f.write(f"Actual UP:      {cm[1,0]:6d}         {cm[1,1]:6d}\n")
            
            # Derived metrics
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            f.write(f"\nAdditional Metrics:\n")
            f.write(f"Specificity (True Negative Rate): {specificity:.4f}\n")
            f.write(f"Negative Predictive Value:        {npv:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
    
    print(f"[REPORT] Saved classification reports to {report_file}")


def create_summary_report(all_results, save_dir='task7_results'):
    """Create executive summary report"""
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'confusion_matrix'} 
                       for r in all_results])
    
    report = []
    report.append("="*80)
    report.append("TASK C.7 - EXECUTIVE SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 80)
    report.append(f"Total Models Evaluated: {len(all_results)}")
    report.append(f"Feature Sets: {df['feature_set'].nunique()} (technical_only, sentiment_only, combined)")
    report.append(f"Algorithms: {df['algorithm'].nunique()} (logistic, random_forest, xgboost)")
    report.append("")
    
    # Best models
    report.append("TOP 5 MODELS (by F1 Score)")
    report.append("-" * 80)
    top5 = df.nlargest(5, 'f1')
    for idx, row in top5.iterrows():
        report.append(f"{row['model']}")
        report.append(f"  Accuracy: {row['accuracy']:.4f} | Precision: {row['precision']:.4f} | "
                     f"Recall: {row['recall']:.4f} | F1: {row['f1']:.4f}")
    report.append("")
    
    # Feature set comparison
    report.append("PERFORMANCE BY FEATURE SET")
    report.append("-" * 80)
    for feat in ['technical_only', 'sentiment_only', 'combined']:
        subset = df[df['feature_set'] == feat]
        if len(subset) > 0:
            report.append(f"\n{feat.upper().replace('_', ' ')}:")
            report.append(f"  Best F1: {subset['f1'].max():.4f} ({subset.loc[subset['f1'].idxmax(), 'algorithm']})")
            report.append(f"  Avg Accuracy: {subset['accuracy'].mean():.4f}")
            report.append(f"  Avg F1: {subset['f1'].mean():.4f}")
    report.append("")
    
    # Baseline comparison
    report.append("BASELINE COMPARISON (SENTIMENT VALUE ASSESSMENT)")
    report.append("-" * 80)
    
    tech_best = df[df['feature_set'] == 'technical_only']['f1'].max()
    sent_best = df[df['feature_set'] == 'sentiment_only']['f1'].max()
    comb_best = df[df['feature_set'] == 'combined']['f1'].max()
    
    improvement = ((sent_best - tech_best) / tech_best * 100) if tech_best > 0 else 0
    
    report.append(f"Baseline (Technical Only):  F1 = {tech_best:.4f}")
    report.append(f"With Sentiment Only:        F1 = {sent_best:.4f}")
    report.append(f"Combined Features:          F1 = {comb_best:.4f}")
    report.append(f"\nSentiment Improvement: {improvement:+.1f}%")
    
    if sent_best > tech_best:
        report.append("\n[OK] CONCLUSION: Sentiment features ADD SIGNIFICANT VALUE!")
    else:
        report.append("\n[WARNING] CONCLUSION: Sentiment features do not improve performance")
    
    report.append("")
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    with open(f'{save_dir}/executive_summary.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"[REPORT] Saved executive summary to {save_dir}/executive_summary.txt")
    print("\n" + report_text)


def main():
    """
    Main evaluation runner
    
    This function orchestrates the creation of all visualizations and reports:
    1. Loads evaluation metrics from JSON (all 21 models)
    2. Loads trained models for feature importance analysis
    3. Generates 3 plots (confusion matrices, comparison, feature importance)
    4. Creates 2 text reports (classification reports, executive summary)
    
    All outputs saved to task7_results/ for easy access and report writing.
    
    Returns:
        None (saves all outputs to task7_results/)
    """
    print("\n" + "="*80)
    print("TASK C.7 - ADVANCED EVALUATION & VISUALIZATION")
    print("="*80)
    
    # Load evaluation results (created by task7_runner.py and task7_extended_models.py)
    print("\n[LOAD] Loading evaluation results...")
    with open('task7_results/evaluation_metrics.json', 'r') as f:
        all_results = json.load(f)
    
    print(f"[OK] Loaded {len(all_results)} model results")
    
    # Load trained models from disk (for feature importance analysis)
    print("\n[LOAD] Loading trained models...")
    models = {}
    for result in all_results:
        model_name = result['model']
        try:
            # Load .pkl file containing model, scaler, and feature names
            model_data = joblib.load(f'task7_models/{model_name}.pkl')
            models[model_name] = model_data
        except Exception as e:
            # Skip if model file not found (shouldn't happen but handle gracefully)
            print(f"[WARNING] Could not load {model_name}: {e}")
    
    print(f"[OK] Loaded {len(models)} models")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_confusion_matrices(all_results)
    plot_model_comparison(all_results)
    
    if models:
        plot_feature_importance(models, 
                               ['return_1d', 'return_5d', 'return_20d', 'volatility_5d', 
                                'volatility_20d', 'ma_5', 'ma_20', 'ma_50', 'rsi', 'macd',
                                'macd_signal', 'bb_middle', 'bb_upper', 'bb_lower',
                                'sentiment_mean', 'sentiment_std'])
    
    # Generate text reports (2 files)
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)
    
    generate_classification_reports(all_results)  # Detailed metrics for all 21 models
    create_summary_report(all_results)  # Executive summary for report writing
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nOutputs:")
    print("  [PLOTS] task7_results/plots/")
    print("     - confusion_matrices.png (heatmaps for top 6 models)")
    print("     - model_comparison.png (4-subplot comprehensive analysis)")
    print("     - feature_importance.png (feature rankings for tree models)")
    print("  [REPORTS] task7_results/")
    print("     - classification_reports.txt (detailed metrics for 21 models)")
    print("     - executive_summary.txt (summary for academic report)")
    print("\n[OK] All visualizations and reports generated!")
    print("📊 Ready for academic presentation and report writing!")


if __name__ == '__main__':
    main()
