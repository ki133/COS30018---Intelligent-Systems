#!/usr/bin/env python
"""
Task C.7: Extended Model Testing
Test additional algorithms: SVM, Gradient Boosting, MLP Neural Network

This script extends the basic pipeline by training 12 additional models:
- SVM with Linear kernel (3 models: technical, sentiment, combined)
- SVM with RBF kernel (3 models: technical, sentiment, combined)
- Gradient Boosting (3 models: technical, sentiment, combined)
- MLP Neural Network (3 models: technical, sentiment, combined)

Total: 12 new models + 9 from main pipeline = 21 models

These advanced algorithms often outperform basic models and provide:
- SVM: Strong theoretical foundation, handles high-dimensional data
- Gradient Boosting: Sequential ensemble, often best performance
- MLP: Neural network, captures complex non-linear patterns

Author: Anh Vu Le
Date: November 2025
Course: COS30018 - Intelligent Systems
"""

import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warnings for cleaner output

import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import joblib  # Model serialization
import os  # File operations
import json  # JSON handling for results
from datetime import datetime  # Timestamp for logging
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.ensemble import GradientBoostingClassifier  # Gradient Boosting
from sklearn.neural_network import MLPClassifier  # Multi-Layer Perceptron (Neural Network)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix)  # Evaluation metrics

# Task 7 custom modules
from task7_sentiment.classifier_models import SentimentClassifierTrainer  # Training utilities


def load_feature_data():
    """
    Load preprocessed feature data from main pipeline
    
    This function loads the complete dataset that was created by task7_runner.py.
    The dataset contains:
    - Technical features (14): returns, volatility, MA, RSI, MACD, Bollinger Bands
    - Sentiment features (2): sentiment_mean, sentiment_std
    - Target variable (1): binary UP/DOWN prediction
    
    Returns:
        tuple: (full_df, technical_features, sentiment_features, y, split_idx)
            - full_df: Complete DataFrame with all features
            - technical_features: List of technical feature column names
            - sentiment_features: List of sentiment feature column names
            - y: Target variable (pandas Series)
            - split_idx: Index for temporal train/test split
    """
    print("\n[LOAD] Loading feature data...")
    
    # Load complete feature dataset (created by task7_runner.py)
    full_df = pd.read_csv('task7_data/news_processed/full_features.csv')
    
    # Identify feature columns by naming convention
    # Technical features: all indicators computed from stock price data
    technical_features = [c for c in full_df.columns if c.startswith((
        'return_', 'volatility_', 'ma_', 'rsi', 'macd', 'bb_'
    ))]
    # Sentiment features: aggregated news sentiment scores
    sentiment_features = [c for c in full_df.columns if c.startswith('sentiment_')]
    
    # Extract target variable (binary classification: UP=1, DOWN=0)
    y = full_df['target']
    
    # Temporal split: 80% train, 20% test (same as main pipeline)
    # IMPORTANT: We use the SAME split to ensure fair comparison with main models
    test_size = 0.2
    split_idx = int(len(full_df) * (1 - test_size))
    
    print(f"[OK] Loaded {len(full_df)} samples")
    print(f"  Technical features: {len(technical_features)}")
    print(f"  Sentiment features: {len(sentiment_features)}")
    print(f"  Split: {split_idx} train, {len(full_df) - split_idx} test")
    
    return full_df, technical_features, sentiment_features, y, split_idx


def train_svm_models(full_df, tech_feats, sent_feats, y, split_idx):
    """
    Train Support Vector Machine (SVM) models with different kernels
    
    SVM is a powerful classifier that finds the optimal hyperplane to separate classes.
    We test two kernel functions:
    - Linear: For linearly separable data (fast, interpretable)
    - RBF (Radial Basis Function): For non-linear patterns (more flexible)
    
    Args:
        full_df: Complete DataFrame with all features
        tech_feats: List of technical feature names
        sent_feats: List of sentiment feature names
        y: Target variable
        split_idx: Index for train/test split
        
    Returns:
        tuple: (results, models)
            - results: List of evaluation metrics for each model
            - models: Dictionary of trained SVM models
    """
    print("\n" + "="*80)
    print("TRAINING SVM MODELS")
    print("="*80)
    
    # Initialize trainer with SMOTE (handles class imbalance)
    trainer = SentimentClassifierTrainer(use_smote=True, random_state=42)
    
    results = []  # Store evaluation metrics
    models = {}  # Store trained models
    
    # Three feature sets to compare (same as main pipeline)
    feature_sets = {
        'technical_only': tech_feats,  # Baseline (no sentiment)
        'sentiment_only': sent_feats,  # Only sentiment
        'combined': tech_feats + sent_feats  # All features
    }
    
    # Two SVM kernels to test
    kernels = ['linear', 'rbf']  # Linear for simple patterns, RBF for complex
    
    # Loop through each feature set
    for feat_name, feat_cols in feature_sets.items():
        X = full_df[feat_cols]  # Extract feature columns
        
        # Temporal split (maintain time order - crucial!)
        X_train = X.iloc[:split_idx]  # First 80% for training
        X_test = X.iloc[split_idx:]  # Last 20% for testing
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Preprocess: StandardScaler + SMOTE
        # StandardScaler is CRITICAL for SVM (sensitive to feature scales)
        X_train_proc, y_train_proc, X_test_proc = trainer.preprocess_data(
            X_train, y_train, X_test, experiment_name=f'svm_{feat_name}'
        )
        
        # Train SVM with each kernel type
        for kernel in kernels:
            model_name = f"{feat_name}_svm_{kernel}"
            print(f"\n[TRAIN] {model_name}...")
            
            # Train SVM classifier
            # probability=True enables predict_proba() for ROC curves
            svm = SVC(kernel=kernel, probability=True, random_state=42)
            svm.fit(X_train_proc, y_train_proc)
            
            # Predict on test set
            y_pred = svm.predict(X_test_proc)
            
            # Calculate comprehensive metrics
            metrics = {
                'model': model_name,
                'feature_set': feat_name,
                'algorithm': f'svm_{kernel}',
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            results.append(metrics)
            
            # Store model with all components needed for prediction
            models[model_name] = {
                'model': svm,  # Trained SVM
                'scaler': trainer.scalers[f'svm_{feat_name}'],  # Fitted scaler
                'features': feat_cols  # Feature names
            }
            
            # Save to disk for later use
            joblib.dump(models[model_name], f'task7_models/{model_name}.pkl')
            
            print(f"  Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
    
    return results, models


def train_gradientboosting_models(full_df, tech_feats, sent_feats, y, split_idx):
    """
    Train Gradient Boosting models
    
    Gradient Boosting builds an ensemble of weak learners (decision trees) sequentially.
    Each new tree corrects errors made by previous trees, leading to strong performance.
    
    Advantages:
    - Often achieves best performance in competitions
    - Handles non-linear relationships well
    - Built-in feature importance
    - Robust to outliers
    
    Args:
        full_df: Complete DataFrame with all features
        tech_feats: List of technical feature names
        sent_feats: List of sentiment feature names
        y: Target variable
        split_idx: Index for train/test split
        
    Returns:
        tuple: (results, models)
            - results: List of evaluation metrics for each model
            - models: Dictionary of trained Gradient Boosting models
    """
    print("\n" + "="*80)
    print("TRAINING GRADIENT BOOSTING MODELS")
    print("="*80)
    
    trainer = SentimentClassifierTrainer(use_smote=True, random_state=42)
    
    results = []
    models = {}
    
    feature_sets = {
        'technical_only': tech_feats,
        'sentiment_only': sent_feats,
        'combined': tech_feats + sent_feats
    }
    
    for feat_name, feat_cols in feature_sets.items():
        X = full_df[feat_cols]
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Preprocess
        X_train_proc, y_train_proc, X_test_proc = trainer.preprocess_data(
            X_train, y_train, X_test, experiment_name=f'gb_{feat_name}'
        )
        
        model_name = f"{feat_name}_gradientboosting"
        print(f"\n[TRAIN] {model_name}...")
        
        # Train Gradient Boosting classifier
        # n_estimators=100: Build 100 sequential trees (good balance of performance vs speed)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_train_proc, y_train_proc)
        
        # Predict
        y_pred = gb.predict(X_test_proc)
        
        # Metrics
        metrics = {
            'model': model_name,
            'feature_set': feat_name,
            'algorithm': 'gradientboosting',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        results.append(metrics)
        
        # Save model
        models[model_name] = {
            'model': gb,
            'scaler': trainer.scalers[f'gb_{feat_name}'],
            'features': feat_cols
        }
        
        joblib.dump(models[model_name], f'task7_models/{model_name}.pkl')
        
        print(f"  Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
    
    return results, models


def train_mlp_models(full_df, tech_feats, sent_feats, y, split_idx):
    """
    Train Multi-Layer Perceptron (Neural Network) models
    
    MLP is a feedforward artificial neural network with multiple layers.
    Architecture: Input → Hidden Layer 1 (64 neurons) → Hidden Layer 2 (32 neurons) → Output
    
    Advantages:
    - Captures complex non-linear patterns
    - Learns hierarchical representations
    - Can approximate any function (universal approximator)
    
    Disadvantages:
    - Requires more data than traditional ML
    - Prone to overfitting on small datasets
    - Longer training time
    
    Args:
        full_df: Complete DataFrame with all features
        tech_feats: List of technical feature names
        sent_feats: List of sentiment feature names
        y: Target variable
        split_idx: Index for train/test split
        
    Returns:
        tuple: (results, models)
            - results: List of evaluation metrics for each model
            - models: Dictionary of trained MLP models
    """
    print("\n" + "="*80)
    print("TRAINING MLP NEURAL NETWORK MODELS")
    print("="*80)
    
    trainer = SentimentClassifierTrainer(use_smote=True, random_state=42)
    
    results = []
    models = {}
    
    feature_sets = {
        'technical_only': tech_feats,
        'sentiment_only': sent_feats,
        'combined': tech_feats + sent_feats
    }
    
    for feat_name, feat_cols in feature_sets.items():
        X = full_df[feat_cols]
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Preprocess
        X_train_proc, y_train_proc, X_test_proc = trainer.preprocess_data(
            X_train, y_train, X_test, experiment_name=f'mlp_{feat_name}'
        )
        
        model_name = f"{feat_name}_mlp"
        print(f"\n[TRAIN] {model_name}...")
        
        # Train MLP Neural Network
        # hidden_layer_sizes=(64, 32): 2 hidden layers with 64 and 32 neurons
        # max_iter=500: Maximum training epochs (iterations)
        # This creates network: Input → 64 neurons → 32 neurons → Output
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        mlp.fit(X_train_proc, y_train_proc)
        
        # Predict
        y_pred = mlp.predict(X_test_proc)
        
        # Metrics
        metrics = {
            'model': model_name,
            'feature_set': feat_name,
            'algorithm': 'mlp',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        results.append(metrics)
        
        # Save model
        models[model_name] = {
            'model': mlp,
            'scaler': trainer.scalers[f'mlp_{feat_name}'],
            'features': feat_cols
        }
        
        joblib.dump(models[model_name], f'task7_models/{model_name}.pkl')
        
        print(f"  Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
    
    return results, models


def update_evaluation_results(new_results):
    """
    Merge new model results with existing evaluation metrics
    
    This function updates the evaluation files to include both:
    - Original 9 models from task7_runner.py
    - New 12 models from this extended testing
    Total: 21 models for comprehensive comparison
    
    Args:
        new_results: List of evaluation dictionaries for new models
        
    Returns:
        None (updates files in task7_results/)
    """
    print("\n" + "="*80)
    print("UPDATING EVALUATION RESULTS")
    print("="*80)
    
    # Load existing results from main pipeline
    with open('task7_results/evaluation_metrics.json', 'r') as f:
        existing_results = json.load(f)
    
    # Merge old and new results
    all_results = existing_results + new_results
    
    # Save complete results (21 models)
    with open('task7_results/evaluation_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Update CSV comparison table (sorted by F1 score)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'confusion_matrix'} 
                       for r in all_results])
    df = df.sort_values('f1', ascending=False)  # Best models first
    df.to_csv('task7_results/model_comparison.csv', index=False)
    
    print(f"[OK] Updated evaluation with {len(new_results)} new models")
    print(f"[OK] Total models: {len(all_results)}")
    
    # Show top 10
    print("\nTOP 10 MODELS (by F1 Score):")
    print("-" * 80)
    for idx, row in df.head(10).iterrows():
        print(f"{row['model']:40s} F1={row['f1']:.4f}  Acc={row['accuracy']:.4f}")


def main():
    """
    Main runner for extended model testing
    
    This function orchestrates the training of 12 additional models:
    1. SVM Linear (3 models)
    2. SVM RBF (3 models)
    3. Gradient Boosting (3 models)
    4. MLP Neural Network (3 models)
    
    All models are trained on the same data splits as the main pipeline
    to ensure fair comparison.
    
    Returns:
        None (saves models to task7_models/, updates task7_results/)
    """
    import time
    start_time = time.time()  # Track execution time
    
    print("\n" + "="*80)
    print("TASK C.7 - EXTENDED MODEL TESTING")
    print("Testing: SVM, Gradient Boosting, MLP Neural Network")
    print("="*80)
    
    # Load preprocessed data (created by task7_runner.py)
    full_df, tech_feats, sent_feats, y, split_idx = load_feature_data()
    
    # Train new models (12 total)
    all_new_results = []  # Collect all evaluation results
    
    # 1. Train SVM models (6 models: 2 kernels × 3 feature sets)
    svm_results, svm_models = train_svm_models(full_df, tech_feats, sent_feats, y, split_idx)
    all_new_results.extend(svm_results)
    
    # 2. Train Gradient Boosting models (3 models: 1 algorithm × 3 feature sets)
    gb_results, gb_models = train_gradientboosting_models(full_df, tech_feats, sent_feats, y, split_idx)
    all_new_results.extend(gb_results)
    
    # 3. Train MLP Neural Network models (3 models: 1 architecture × 3 feature sets)
    mlp_results, mlp_models = train_mlp_models(full_df, tech_feats, sent_feats, y, split_idx)
    all_new_results.extend(mlp_results)
    
    # Merge with existing results (9 from main + 12 from extended = 21 total)
    update_evaluation_results(all_new_results)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("EXTENDED MODEL TESTING COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"New models trained: {len(all_new_results)}")
    print(f"  - SVM (Linear + RBF): {len(svm_results)} models")
    print(f"  - Gradient Boosting: {len(gb_results)} models")
    print(f"  - MLP Neural Network: {len(mlp_results)} models")
    print(f"\nTotal models now: 21 (9 basic + 12 extended)")
    print("\n✅ Run task7_advanced_evaluation.py to generate visualizations!")
    print("📊 Check task7_results/model_comparison.csv to see all rankings")


if __name__ == '__main__':
    main()
