#!/usr/bin/env python
"""
Task C.7: Complete Sentiment-Based Stock Prediction Pipeline
With full evaluation metrics and baseline comparison

This is the main pipeline that orchestrates the entire sentiment-based stock prediction process.
It performs 5 major stages:
1. Load news data from CSV
2. Perform sentiment analysis on news articles
3. Engineer features combining technical indicators and sentiment
4. Train 9 models (3 feature sets × 3 algorithms)
5. Evaluate with full metrics and baseline comparison

Author: Anh Vu Le
Date: November 2025
Course: COS30018 - Intelligent Systems
Institution: Swinburne University of Technology
"""

import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings for cleaner output

import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing
import joblib  # Model serialization (save/load)
import os  # File system operations
from datetime import datetime, timedelta  # Date handling
import yfinance as yf  # Yahoo Finance API for stock data
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report)  # Evaluation metrics

# Task 7 custom modules for sentiment-based prediction
from task7_sentiment.config import Task7Config  # Configuration settings
from task7_sentiment.sentiment_analyzer import SentimentAnalyzer  # Sentiment analysis engine
from task7_sentiment.feature_builder import SentimentFeatureBuilder  # Feature engineering
from task7_sentiment.classifier_models import SentimentClassifierTrainer  # Model training


def main():
    """
    Complete Task C.7 Pipeline
    
    This function orchestrates the entire sentiment-based stock prediction workflow:
    1. Loads pre-collected news articles from CSV
    2. Analyzes sentiment using TextBlob (lexicon-based)
    3. Aggregates daily sentiment scores
    4. Downloads stock data and computes technical indicators
    5. Merges sentiment with technical features
    6. Trains 9 classification models (3 feature sets × 3 algorithms)
    7. Evaluates all models with comprehensive metrics
    8. Compares baseline (technical-only) vs sentiment-enhanced models
    
    No arguments required - all paths are hardcoded for reproducibility.
    
    Returns:
        None (saves results to task7_models/ and task7_results/)
    """
    import time
    start_time = time.time()  # Track execution time
    
    print("="*80)
    print("TASK C.7: SENTIMENT-BASED STOCK PREDICTION")
    print("Complete Pipeline with Full Evaluation")
    print("="*80)
    
    # =========================================================================
    # STAGE 1: LOAD NEWS DATA
    # =========================================================================
    # Load pre-collected news articles from CSV file.
    # These articles were scraped using news_scraper.py (Google News RSS)
    # and contain: date, title, description, content, source, url
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 1: LOADING NEWS DATA")
    print("="*80)
    
    news_file = 'task7_data/news_raw/news_raw.csv'
    print(f"\n[OK] Loading from {news_file}")
    news_df = pd.read_csv(news_file)  # Read CSV with all news articles
    
    print(f"[OK] Loaded {len(news_df)} articles")
    
    # Display source diversity (shows credibility - not from single source)
    source_counts = news_df['source'].value_counts()
    print(f"\nSources: {len(source_counts)} unique")
    for source, count in source_counts.head(10).items():
        print(f"  {source}: {count}")
    
    # =========================================================================
    # STAGE 2: SENTIMENT ANALYSIS
    # =========================================================================
    # Perform sentiment analysis on news articles using TextBlob.
    # TextBlob is a lexicon-based method that scores text from -1 (negative)
    # to +1 (positive). We use article titles because they're concise and
    # capture the main sentiment without noise from long content.
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 2: SENTIMENT ANALYSIS")
    print("="*80)
    
    # Initialize sentiment analyzer with lexicon-based method (fast and reliable)
    analyzer = SentimentAnalyzer(primary_model='lexicon')
    
    # Analyze each article title and get sentiment score (-1 to +1)
    news_df['sentiment_score'] = news_df['title'].apply(
        lambda x: analyzer.analyze(str(x), method='textblob')
    )
    
    # Categorize sentiment into discrete classes for interpretability
    # Thresholds: >0.05 = positive, <-0.05 = negative, else neutral
    news_df['sentiment_category'] = news_df['sentiment_score'].apply(
        lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
    )
    
    # Display sentiment distribution to verify diversity (not all neutral)
    print(f"\n[OK] Analyzed {len(news_df)} articles:")
    print(f"  Positive: {(news_df['sentiment_category'] == 'positive').sum()}")
    print(f"  Neutral: {(news_df['sentiment_category'] == 'neutral').sum()}")
    print(f"  Negative: {(news_df['sentiment_category'] == 'negative').sum()}")
    
    # Aggregate sentiment by day because stock prices are daily
    # Multiple articles per day → need single daily sentiment score
    news_df['date'] = pd.to_datetime(news_df['date'])
    
    daily_sentiment = news_df.groupby(news_df['date'].dt.date).agg({
        'sentiment_score': ['mean', 'std', 'count'],  # Mean sentiment, volatility, article count
        'sentiment_category': lambda x: (x == 'positive').sum() / len(x) if len(x) > 0 else 0  # Positive ratio
    }).reset_index()
    
    # Flatten multi-level columns
    daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'article_count', 'positive_ratio']
    daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)  # Fill NaN with 0 for single-article days
    
    print(f"\nAggregated to {len(daily_sentiment)} trading days")
    
    # Save daily sentiment for future use
    os.makedirs('task7_data/news_processed', exist_ok=True)
    daily_sentiment.to_csv('task7_data/news_processed/daily_sentiment.csv', index=False)
    print("[OK] Saved to task7_data/news_processed/daily_sentiment.csv")
    
    # =========================================================================
    # STAGE 3: FEATURE ENGINEERING
    # =========================================================================
    # Compute technical indicators from stock price data and merge with sentiment.
    # Technical indicators capture price patterns, momentum, and volatility.
    # We compute 14 technical features + 2 sentiment features = 16 total.
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 3: FEATURE ENGINEERING")
    print("="*80)
    
    # Download historical stock data from Yahoo Finance
    ticker = 'CBA.AX'  # Commonwealth Bank of Australia
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # Last 2 years
    
    stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    stock_df.reset_index(inplace=True)
    stock_df.columns = [col[0] if isinstance(col, tuple) else col for col in stock_df.columns]  # Flatten multi-index
    
    print(f"\n[OK] Downloaded {len(stock_df)} days of stock data")
    
    # Build technical features manually (14 features total)
    
    # 1. RETURNS: Measure price changes over different time periods
    stock_df['return_1d'] = stock_df['Close'].pct_change()  # Daily return
    stock_df['return_5d'] = stock_df['Close'].pct_change(5)  # Weekly return
    stock_df['return_20d'] = stock_df['Close'].pct_change(20)  # Monthly return
    
    # 2. VOLATILITY: Measure price instability (risk indicator)
    stock_df['volatility_5d'] = stock_df['return_1d'].rolling(5).std()  # Short-term volatility
    stock_df['volatility_20d'] = stock_df['return_1d'].rolling(20).std()  # Long-term volatility
    
    # 3. MOVING AVERAGES: Smooth price trends
    stock_df['ma_5'] = stock_df['Close'].rolling(5).mean()  # 5-day MA (short-term trend)
    stock_df['ma_20'] = stock_df['Close'].rolling(20).mean()  # 20-day MA (medium-term trend)
    stock_df['ma_50'] = stock_df['Close'].rolling(50).mean()  # 50-day MA (long-term trend)
    
    # 4. RSI (Relative Strength Index): Momentum oscillator (0-100)
    # Values >70 suggest overbought, <30 suggest oversold
    delta = stock_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()  # Average gain over 14 days
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()  # Average loss over 14 days
    rs = gain / loss  # Relative strength
    stock_df['rsi'] = 100 - (100 / (1 + rs))  # RSI formula
    
    # 5. MACD (Moving Average Convergence Divergence): Trend-following momentum
    exp1 = stock_df['Close'].ewm(span=12, adjust=False).mean()  # 12-day EMA
    exp2 = stock_df['Close'].ewm(span=26, adjust=False).mean()  # 26-day EMA
    stock_df['macd'] = exp1 - exp2  # MACD line
    stock_df['macd_signal'] = stock_df['macd'].ewm(span=9, adjust=False).mean()  # Signal line
    
    # 6. BOLLINGER BANDS: Volatility bands (price typically stays within ±2 std)
    stock_df['bb_middle'] = stock_df['Close'].rolling(20).mean()  # Middle band (20-day MA)
    bb_std = stock_df['Close'].rolling(20).std()
    stock_df['bb_upper'] = stock_df['bb_middle'] + (2 * bb_std)  # Upper band
    stock_df['bb_lower'] = stock_df['bb_middle'] - (2 * bb_std)  # Lower band
    
    # Merge stock data with sentiment data on date
    # Left join ensures we keep all stock days (even without news)
    stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.date
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date']).dt.date
    
    full_df = stock_df.merge(daily_sentiment, on='date', how='left')
    
    # Fill missing sentiment values (days without news) with neutral values
    # This is important: missing news doesn't mean negative sentiment!
    full_df['sentiment_mean'] = full_df['sentiment_mean'].fillna(0)  # Neutral sentiment
    full_df['sentiment_std'] = full_df['sentiment_std'].fillna(0)  # No volatility
    full_df['article_count'] = full_df['article_count'].fillna(0)  # No articles
    full_df['positive_ratio'] = full_df['positive_ratio'].fillna(0.5)  # 50% positive (neutral)
    
    # Create binary target variable: 1 if price goes UP tomorrow, 0 if DOWN
    # shift(-1) looks at next day's price (future prediction target)
    full_df['target'] = (full_df['Close'].shift(-1) > full_df['Close']).astype(int)
    
    # Remove rows with NaN (caused by rolling windows and shift operations)
    full_df = full_df.dropna()
    
    print(f"[OK] Feature engineering complete:")
    print(f"  Final samples: {len(full_df)}")
    
    # Identify feature columns by prefix for organized feature sets
    # This allows us to compare: technical-only vs sentiment-only vs combined
    technical_features = [c for c in full_df.columns if c.startswith((
        'return_', 'volatility_', 'ma_', 'rsi', 'macd', 'bb_'
    ))]
    sentiment_features = [c for c in full_df.columns if c.startswith('sentiment_')]
    
    print(f"  Technical features: {len(technical_features)}")
    print(f"  Sentiment features: {len(sentiment_features)}")
    print(f"  Total: {len(technical_features) + len(sentiment_features)}")
    
    # Save complete feature dataset for reproducibility
    full_df.to_csv('task7_data/news_processed/full_features.csv', index=False)
    print("[OK] Saved to task7_data/news_processed/full_features.csv")
    
    # =========================================================================
    # STAGE 4: MODEL TRAINING
    # =========================================================================
    # Train 9 models = 3 feature sets × 3 algorithms
    # Feature sets: technical_only (baseline), sentiment_only, combined
    # Algorithms: Logistic Regression, Random Forest, XGBoost
    # This allows comprehensive comparison to assess sentiment value.
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 4: MODEL TRAINING")
    print("="*80)
    
    # Extract target variable (binary: UP=1, DOWN=0)
    y = full_df['target']
    
    # Define three feature sets for comparison
    # technical_only = baseline (no sentiment)
    # sentiment_only = only sentiment features (interesting experiment)
    # combined = all features (best performance expected)
    feature_sets = {
        'technical_only': technical_features,
        'sentiment_only': sentiment_features,
        'combined': technical_features + sentiment_features
    }
    
    # Three classification algorithms with different strengths:
    # - Logistic: Linear, fast, interpretable
    # - Random Forest: Non-linear, handles interactions, robust
    # - XGBoost: Gradient boosting, often best performance
    algorithms = ['logistic', 'random_forest', 'xgboost']
    
    # Temporal split (80% train, 20% test) - NO SHUFFLE!
    # Time series must maintain order to prevent data leakage
    test_size = 0.2
    split_idx = int(len(full_df) * (1 - test_size))
    
    # Train all combinations (3 feature sets × 3 algorithms = 9 models)
    # use_smote=True handles class imbalance by oversampling minority class
    trainer = SentimentClassifierTrainer(use_smote=True, random_state=42)
    
    all_models = {}  # Store trained models
    all_predictions = {}  # Store predictions for evaluation
    
    os.makedirs('task7_models', exist_ok=True)  # Create directory for saved models
    
    # Loop through each feature set (technical, sentiment, combined)
    for feat_name, feat_cols in feature_sets.items():
        X = full_df[feat_cols]  # Extract feature columns
        
        # Temporal split (maintain time order - crucial for stock data!)
        X_train = X.iloc[:split_idx]  # First 80% for training
        X_test = X.iloc[split_idx:]  # Last 20% for testing (future data)
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"\n[{feat_name.upper()}]")
        print(f"  Features: {len(feat_cols)}")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Preprocess: StandardScaler + SMOTE (class balancing)
        # StandardScaler: Normalize features to mean=0, std=1
        # SMOTE: Synthetic Minority Oversampling (balances UP/DOWN classes)
        X_train_proc, y_train_proc, X_test_proc = trainer.preprocess_data(
            X_train, y_train, X_test, experiment_name=feat_name
        )
        
        for algo in algorithms:
            model_name = f"{feat_name}_{algo}"
            print(f"\n  Training {algo}...")
            
            # Train
            model = trainer.train_model(
                pd.DataFrame(X_train_proc, columns=feat_cols),
                pd.Series(y_train_proc),
                model_type=algo,
                experiment_name=model_name
            )
            
            # Predict
            y_pred = trainer.predict(model, pd.DataFrame(X_test_proc, columns=feat_cols))
            
            # Store
            all_models[model_name] = {
                'model': model,
                'scaler': trainer.scalers[feat_name],
                'features': feat_cols
            }
            all_predictions[model_name] = {
                'y_test': y_test.values,
                'y_pred': y_pred,
                'feature_set': feat_name,
                'algorithm': algo
            }
            
            # Save model
            joblib.dump(all_models[model_name], f'task7_models/{model_name}.pkl')
            
            # Quick accuracy
            acc = accuracy_score(y_test, y_pred)
            print(f"    Accuracy: {acc:.3f}")
    
    print(f"\n[OK] Trained {len(all_models)} models")
    print(f"[OK] Saved to task7_models/")
    
    # =========================================================================
    # STAGE 5: EVALUATION WITH ALL METRICS
    # =========================================================================
    # Compute comprehensive evaluation metrics for each model:
    # - Accuracy: Overall correctness
    # - Precision: How many predicted UPs were actually UP
    # - Recall: How many actual UPs were correctly predicted
    # - F1 Score: Harmonic mean of precision and recall
    # - Confusion Matrix: Detailed breakdown of TP, TN, FP, FN
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 5: COMPREHENSIVE EVALUATION")
    print("="*80)
    
    all_results = []  # Store all evaluation results
    
    # Evaluate each model on test set
    for model_name, preds in all_predictions.items():
        y_test = preds['y_test']  # True labels
        y_pred = preds['y_pred']  # Predicted labels
        
        # Calculate all required metrics for Task C.7
        metrics = {
            'model': model_name,
            'feature_set': preds['feature_set'],
            'algorithm': preds['algorithm'],
            'accuracy': accuracy_score(y_test, y_pred),  # (TP+TN)/(TP+TN+FP+FN)
            'precision': precision_score(y_test, y_pred, zero_division=0),  # TP/(TP+FP)
            'recall': recall_score(y_test, y_pred, zero_division=0),  # TP/(TP+FN)
            'f1': f1_score(y_test, y_pred, zero_division=0)  # 2*(P*R)/(P+R)
        }
        
        # Confusion matrix: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()  # Convert to list for JSON serialization
        
        all_results.append(metrics)
        
        # Display metrics for each model
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")  # How reliable are UP predictions?
        print(f"  Recall:    {metrics['recall']:.3f}")  # How many UPs did we catch?
        print(f"  F1 Score:  {metrics['f1']:.3f}")  # Balanced metric (important for imbalanced data)
        print(f"  Confusion Matrix: [[TN={cm[0,0]}, FP={cm[0,1]}], [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    # =========================================================================
    # BASELINE COMPARISON (REQUIREMENT: Compare with/without sentiment)
    # =========================================================================
    # This is a CRITICAL requirement for Task C.7:
    # We must demonstrate whether sentiment features ADD VALUE over technical-only.
    # Baseline = technical_only (no sentiment)
    # Test = sentiment_only or combined
    # If sentiment improves F1, we prove sentiment is valuable!
    # =========================================================================
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print("="*80)
    
    # Best technical-only model (BASELINE - no sentiment features)
    tech_results = [r for r in all_results if r['feature_set'] == 'technical_only']
    best_tech = max(tech_results, key=lambda x: x['f1']) if tech_results else None
    
    # Best sentiment-only model (interesting experiment)
    sent_results = [r for r in all_results if r['feature_set'] == 'sentiment_only']
    best_sent = max(sent_results, key=lambda x: x['f1']) if sent_results else None
    
    # Best combined model (expected to be best)
    comb_results = [r for r in all_results if r['feature_set'] == 'combined']
    best_comb = max(comb_results, key=lambda x: x['f1']) if comb_results else None
    
    if best_tech:
        print("\nBASELINE (Technical Only - No Sentiment):")
        print(f"  Model: {best_tech['model']}")
        print(f"  Accuracy:  {best_tech['accuracy']:.3f}")
        print(f"  Precision: {best_tech['precision']:.3f}")
        print(f"  Recall:    {best_tech['recall']:.3f}")
        print(f"  F1 Score:  {best_tech['f1']:.3f}")
    
    if best_sent:
        print("\nWITH SENTIMENT (Sentiment Only):")
        print(f"  Model: {best_sent['model']}")
        print(f"  Accuracy:  {best_sent['accuracy']:.3f}")
        print(f"  Precision: {best_sent['precision']:.3f}")
        print(f"  Recall:    {best_sent['recall']:.3f}")
        print(f"  F1 Score:  {best_sent['f1']:.3f}")
    
    if best_comb:
        print("\nCOMBINED (Technical + Sentiment):")
        print(f"  Model: {best_comb['model']}")
        print(f"  Accuracy:  {best_comb['accuracy']:.3f}")
        print(f"  Precision: {best_comb['precision']:.3f}")
        print(f"  Recall:    {best_comb['recall']:.3f}")
        print(f"  F1 Score:  {best_comb['f1']:.3f}")
    
    # Calculate improvement percentage (sentiment vs baseline)
    if best_tech and best_sent:
        print("\n" + "="*80)
        print("SENTIMENT VALUE ASSESSMENT")
        print("="*80)
        
        # Calculate percentage improvement
        imp_acc = (best_sent['accuracy'] - best_tech['accuracy']) / best_tech['accuracy'] * 100 if best_tech['accuracy'] > 0 else 0
        imp_f1 = (best_sent['f1'] - best_tech['f1']) / best_tech['f1'] * 100 if best_tech['f1'] > 0 else 0
        
        print(f"Accuracy Improvement: {imp_acc:+.1f}%")
        print(f"F1 Score Improvement: {imp_f1:+.1f}%")
        
        # Clear conclusion for academic report
        if best_sent['f1'] > best_tech['f1']:
            print("\n✅ CONCLUSION: Sentiment features ADD SIGNIFICANT VALUE!")
            print("   Recommendation: Use sentiment-enhanced models for stock prediction")
        else:
            print("\n⚠️  CONCLUSION: Sentiment features do not improve performance")
            print("   Recommendation: Stick with technical-only features")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    # Save evaluation results to JSON and CSV for:
    # 1. Reproducibility (can recreate results without re-training)
    # 2. Further analysis (can be loaded into visualization scripts)
    # 3. Academic report (can copy metrics directly)
    # =========================================================================
    os.makedirs('task7_results', exist_ok=True)
    
    # Save complete metrics as JSON (includes confusion matrices)
    import json
    with open('task7_results/evaluation_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save comparison table as CSV (easier to view in Excel)
    # Exclude confusion_matrix column because CSV can't handle nested lists
    comparison_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'confusion_matrix'} 
                                  for r in all_results])
    comparison_df = comparison_df.sort_values('f1', ascending=False)  # Sort by F1 (best first)
    comparison_df.to_csv('task7_results/model_comparison.csv', index=False)
    
    print("\n[OK] Saved evaluation_metrics.json")
    print("[OK] Saved model_comparison.csv")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    # Display final summary with execution time and output locations
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"\nOutputs:")
    print(f"  Data: task7_data/news_processed/")
    print(f"    - daily_sentiment.csv (aggregated daily sentiment)")
    print(f"    - full_features.csv (complete dataset with all features)")
    print(f"  Models: task7_models/ ({len(all_models)} models)")
    print(f"    - Each .pkl file contains: model, scaler, feature names")
    print(f"  Results: task7_results/")
    print(f"    - evaluation_metrics.json (all metrics)")
    print(f"    - model_comparison.csv (sortable comparison)")
    
    print("\n✅ Task C.7 COMPLETE with full evaluation metrics!")
    print("📊 Next steps:")
    print("   1. Run task7_extended_models.py to train SVM, Gradient Boosting, MLP")
    print("   2. Run task7_advanced_evaluation.py to generate visualizations")


if __name__ == '__main__':
    main()
