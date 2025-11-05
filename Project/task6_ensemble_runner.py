"""
task6_ensemble_runner.py
Task C.6: Ensemble Model Experiment Runner

Combines statistical models (ARIMA/SARIMA) with deep learning models (LSTM/GRU/RNN)
using various ensemble strategies.

This script:
1. Loads and processes stock data using data_processing.py (Task C.2)
2. Trains statistical models (ARIMA/SARIMA) using ensemble_models.py
3. Trains deep learning models (LSTM/GRU) using model_builder.py (Task C.4)
4. Combines predictions using ensemble_methods.py
5. Evaluates and compares all approaches
6. Saves results, plots, and model artifacts

Author: Anh Vu Le
Date: 12/10/2025
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

# Import our existing modules (Tasks C.2, C.4, C.5)
import data_processing
import model_builder
import visualization

# Import new ensemble modules (Task C.6)
from ensemble_models import ARIMAModel, SARIMAModel
from ensemble_methods import (
    SimpleAverageEnsemble, 
    WeightedAverageEnsemble, 
    StackingEnsemble,
    EnsembleCombiner
)

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TensorFlow for DL models
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

class EnsembleExperimentConfig:
    """Configuration for ensemble experiments"""
    
    # Data parameters
    TICKER = 'CBA.AX'
    START_DATE = '2020-01-01'
    END_DATE = '2024-10-01'
    FEATURES = ['Close']  # Start with univariate
    TARGET_COL = 'Close'
    
    # Sequence parameters for DL models
    SEQUENCE_LENGTH = 60
    
    # ARIMA parameters
    ARIMA_ORDER = (5, 1, 0)  # (p, d, q)
    ARIMA_AUTO_SELECT = False  # Disabled due to pmdarima compatibility issues
    
    # SARIMA parameters  
    SARIMA_ORDER = (1, 1, 1)
    SARIMA_SEASONAL_ORDER = (1, 1, 1, 5)  # Weekly seasonality (5 trading days)
    SARIMA_AUTO_SELECT = False
    
    # DL model parameters
    LSTM_UNITS = [64, 32]
    GRU_UNITS = [64]
    DROPOUT = 0.2
    EPOCHS = 50
    BATCH_SIZE = 32
    PATIENCE = 10
    
    # Ensemble methods to test
    ENSEMBLE_METHODS = ['simple_average', 'weighted', 'stacking']
    
    # Output
    RESULTS_DIR = 'task6_results'
    

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------

def create_sequences_for_dl(data: np.ndarray, sequence_length: int) -> tuple:
    """
    Create sequences for deep learning models
    
    Args:
        data: 1D array of values
        sequence_length: Number of past steps to use
        
    Returns:
        X, y arrays for training
    """
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM: (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape)
    }


#------------------------------------------------------------------------------
# Main Experiment Runner
#------------------------------------------------------------------------------

def run_ensemble_experiment(config: EnsembleExperimentConfig):
    """
    Main function to run complete ensemble experiment
    
    Steps:
    1. Load and prepare data
    2. Train ARIMA model
    3. Train SARIMA model
    4. Train LSTM model
    5. Train GRU model
    6. Combine using ensemble methods
    7. Evaluate and compare
    8. Save results
    """
    
    print("="*80)
    print("TASK C.6: ENSEMBLE MODELING EXPERIMENT")
    print("="*80)
    print(f"Ticker: {config.TICKER}")
    print(f"Period: {config.START_DATE} to {config.END_DATE}")
    print(f"Target: {config.TARGET_COL}")
    print("="*80)
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(config.RESULTS_DIR, f'ensemble_exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    print(f"\n[DIR] Results will be saved to: {exp_dir}")
    
    # -------------------------------------------------------------------------
    # Step 1: Load and Process Data using Task C.2 function
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: Loading and Processing Data (Task C.2)")
    print("="*80)
    
    data_dict = data_processing.load_and_process_data(
        ticker=config.TICKER,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        features=config.FEATURES,
        split_method='ratio',
        split_value=0.8,
        cache_dir='data_cache',
        scale_features=True
    )
    
    train_df = data_dict['train_data']
    test_df = data_dict['test_data']
    scalers = data_dict['scalers']
    
    # Get scaled data for training
    train_data = train_df[config.TARGET_COL].values
    test_data = test_df[config.TARGET_COL].values
    
    print(f"[OK] Data loaded: {len(train_data)} train, {len(test_data)} test samples")
    
    # -------------------------------------------------------------------------
    # Step 2: Train ARIMA Model
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: Training ARIMA Model")
    print("="*80)
    
    arima_model = ARIMAModel(
        order=config.ARIMA_ORDER,
        auto_select=config.ARIMA_AUTO_SELECT
    )
    
    arima_results = arima_model.fit(train_data, verbose=True)
    
    # CRITICAL LINE: Make predictions for test period
    arima_predictions = arima_model.predict(n_steps=len(test_data))
    
    # Calculate metrics
    arima_metrics = calculate_metrics(test_data, arima_predictions)
    print(f"\n[OK] ARIMA Metrics:")
    print(f"   MAE: {arima_metrics['MAE']:.4f}")
    print(f"   RMSE: {arima_metrics['RMSE']:.4f}")
    print(f"   MAPE: {arima_metrics['MAPE']:.2f}%")
    
    # -------------------------------------------------------------------------
    # Step 3: Train SARIMA Model (optional, can skip if slow)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: Training SARIMA Model")
    print("="*80)
    
    sarima_model = SARIMAModel(
        order=config.SARIMA_ORDER,
        seasonal_order=config.SARIMA_SEASONAL_ORDER,
        auto_select=config.SARIMA_AUTO_SELECT
    )
    
    try:
        sarima_results = sarima_model.fit(train_data, verbose=True)
        sarima_predictions = sarima_model.predict(n_steps=len(test_data))
        sarima_metrics = calculate_metrics(test_data, sarima_predictions)
        print(f"\n[OK] SARIMA Metrics:")
        print(f"   MAE: {sarima_metrics['MAE']:.4f}")
        print(f"   RMSE: {sarima_metrics['RMSE']:.4f}")
        print(f"   MAPE: {sarima_metrics['MAPE']:.2f}%")
        use_sarima = True
    except Exception as e:
        print(f"[WARN]  SARIMA training failed: {e}")
        print("   Continuing without SARIMA...")
        use_sarima = False
    
    # -------------------------------------------------------------------------
    # Step 4: Train LSTM Model using Task C.4 function
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: Training LSTM Model (Task C.4)")
    print("="*80)
    
    # Create sequences for LSTM
    X_train_lstm, y_train_lstm = create_sequences_for_dl(train_data, config.SEQUENCE_LENGTH)
    X_test_lstm, y_test_lstm = create_sequences_for_dl(test_data, config.SEQUENCE_LENGTH)
    
    print(f"LSTM sequences: X_train={X_train_lstm.shape}, X_test={X_test_lstm.shape}")
    
    # Build LSTM model using our model_builder (Task C.4)
    lstm_build_info = model_builder.build_sequence_model(
        sequence_length=config.SEQUENCE_LENGTH,
        n_features=1,
        layer_type='lstm',
        layer_units=config.LSTM_UNITS,
        dropout=config.DROPOUT,
        optimizer='adam',
        learning_rate=0.001,
        loss='mse',
        metrics=['mae']
    )
    
    lstm_model = lstm_build_info['model']
    
    # CRITICAL LINE: Train LSTM with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=config.PATIENCE, restore_best_weights=True)
    
    lstm_history = lstm_model.fit(
        X_train_lstm, y_train_lstm,
        validation_split=0.2,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Make predictions
    lstm_predictions_scaled = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    
    # Align with test data (skip first sequence_length samples)
    lstm_predictions = lstm_predictions_scaled
    test_data_aligned = y_test_lstm
    
    lstm_metrics = calculate_metrics(test_data_aligned, lstm_predictions)
    print(f"\n[OK] LSTM Metrics:")
    print(f"   MAE: {lstm_metrics['MAE']:.4f}")
    print(f"   RMSE: {lstm_metrics['RMSE']:.4f}")
    print(f"   MAPE: {lstm_metrics['MAPE']:.2f}%")
    
    # -------------------------------------------------------------------------
    # Step 5: Train GRU Model
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: Training GRU Model (Task C.4)")
    print("="*80)
    
    # Build GRU model
    gru_build_info = model_builder.build_sequence_model(
        sequence_length=config.SEQUENCE_LENGTH,
        n_features=1,
        layer_type='gru',
        layer_units=config.GRU_UNITS,
        dropout=config.DROPOUT,
        optimizer='adam',
        learning_rate=0.001,
        loss='mse',
        metrics=['mae']
    )
    
    gru_model = gru_build_info['model']
    
    # Train GRU
    gru_history = gru_model.fit(
        X_train_lstm, y_train_lstm,
        validation_split=0.2,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Make predictions
    gru_predictions_scaled = gru_model.predict(X_test_lstm, verbose=0).flatten()
    gru_predictions = gru_predictions_scaled
    
    gru_metrics = calculate_metrics(test_data_aligned, gru_predictions)
    print(f"\n[OK] GRU Metrics:")
    print(f"   MAE: {gru_metrics['MAE']:.4f}")
    print(f"   RMSE: {gru_metrics['RMSE']:.4f}")
    print(f"   MAPE: {gru_metrics['MAPE']:.2f}%")
    
    # -------------------------------------------------------------------------
    # Step 6: Ensemble Combinations
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 6: Combining Models with Ensemble Methods")
    print("="*80)
    
    # We need to align predictions (ARIMA predicts full test, LSTM/GRU skip sequence_length)
    # Solution: Truncate ARIMA predictions to match LSTM/GRU length
    arima_pred_aligned = arima_predictions[-len(test_data_aligned):]
    if use_sarima:
        sarima_pred_aligned = sarima_predictions[-len(test_data_aligned):]
    
    # Create ensemble combiner
    combiner = EnsembleCombiner()
    combiner.add_model("ARIMA", arima_pred_aligned)
    if use_sarima:
        combiner.add_model("SARIMA", sarima_pred_aligned)
    combiner.add_model("LSTM", lstm_predictions)
    combiner.add_model("GRU", gru_predictions)
    
    # Test different ensemble methods
    ensemble_results = {}
    
    # 6.1: Simple Average Ensemble
    print("\n[ENSEMBLE] 6.1: Simple Average Ensemble")
    simple_ensemble = SimpleAverageEnsemble()
    if use_sarima:
        simple_pred = simple_ensemble.combine([arima_pred_aligned, sarima_pred_aligned, lstm_predictions, gru_predictions])
    else:
        simple_pred = simple_ensemble.combine([arima_pred_aligned, lstm_predictions, gru_predictions])
    
    simple_metrics = calculate_metrics(test_data_aligned, simple_pred)
    ensemble_results['Simple Average'] = simple_metrics
    print(f"   MAE: {simple_metrics['MAE']:.4f}")
    print(f"   RMSE: {simple_metrics['RMSE']:.4f}")
    
    # 6.2: Weighted Average Ensemble (Performance-based)
    print("\n[ENSEMBLE] 6.2: Weighted Average Ensemble (Performance-based)")
    weighted_ensemble = WeightedAverageEnsemble(method='performance')
    if use_sarima:
        weighted_ensemble.fit([arima_pred_aligned, sarima_pred_aligned, lstm_predictions, gru_predictions], test_data_aligned)
        weighted_pred = weighted_ensemble.combine([arima_pred_aligned, sarima_pred_aligned, lstm_predictions, gru_predictions])
    else:
        weighted_ensemble.fit([arima_pred_aligned, lstm_predictions, gru_predictions], test_data_aligned)
        weighted_pred = weighted_ensemble.combine([arima_pred_aligned, lstm_predictions, gru_predictions])
    
    weighted_metrics = calculate_metrics(test_data_aligned, weighted_pred)
    ensemble_results['Weighted Average'] = weighted_metrics
    print(f"   MAE: {weighted_metrics['MAE']:.4f}")
    print(f"   RMSE: {weighted_metrics['RMSE']:.4f}")
    
    # 6.3: Stacking Ensemble (Ridge meta-learner)
    print("\n[ENSEMBLE] 6.3: Stacking Ensemble (Ridge meta-learner)")
    stacking_ensemble = StackingEnsemble(meta_learner='ridge')
    if use_sarima:
        stacking_ensemble.fit([arima_pred_aligned, sarima_pred_aligned, lstm_predictions, gru_predictions], test_data_aligned)
        stacking_pred = stacking_ensemble.combine([arima_pred_aligned, sarima_pred_aligned, lstm_predictions, gru_predictions])
    else:
        stacking_ensemble.fit([arima_pred_aligned, lstm_predictions, gru_predictions], test_data_aligned)
        stacking_pred = stacking_ensemble.combine([arima_pred_aligned, lstm_predictions, gru_predictions])
    
    stacking_metrics = calculate_metrics(test_data_aligned, stacking_pred)
    ensemble_results['Stacking (Ridge)'] = stacking_metrics
    print(f"   MAE: {stacking_metrics['MAE']:.4f}")
    print(f"   RMSE: {stacking_metrics['RMSE']:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 7: Results Comparison and Visualization
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 7: Results Summary and Comparison")
    print("="*80)
    
    # Compile all results
    all_results = {
        'Individual Models': {
            'ARIMA': arima_metrics,
            'LSTM': lstm_metrics,
            'GRU': gru_metrics
        },
        'Ensemble Models': ensemble_results
    }
    
    if use_sarima:
        all_results['Individual Models']['SARIMA'] = sarima_metrics
    
    # Print comparison table
    print("\n[RESULTS] PERFORMANCE COMPARISON:")
    print("-" * 60)
    print(f"{'Model':<30} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 60)
    
    for category, models in all_results.items():
        print(f"\n{category}:")
        for model_name, metrics in models.items():
            print(f"  {model_name:<28} {metrics['MAE']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['MAPE']:<10.2f}%")
    
    print("-" * 60)
    
    # Find best model
    all_models_flat = {**all_results['Individual Models'], **all_results['Ensemble Models']}
    best_model = min(all_models_flat.items(), key=lambda x: x[1]['MAE'])
    print(f"\n[BEST] BEST MODEL: {best_model[0]} (MAE: {best_model[1]['MAE']:.4f})")
    
    # -------------------------------------------------------------------------
    # Step 8: Save Results and Plots
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 8: Saving Results")
    print("="*80)
    
    # Save metrics to JSON
    results_file = os.path.join(exp_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"[OK] Results saved to: {results_file}")
    
    # Save configuration
    config_file = os.path.join(exp_dir, 'config.json')
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"[OK] Configuration saved to: {config_file}")
    
    # Create comparison plot
    plt.figure(figsize=(14, 8))
    
    # Plot actual vs predictions
    x_axis = range(len(test_data_aligned))
    plt.plot(x_axis, test_data_aligned, 'k-', label='Actual', linewidth=2)
    plt.plot(x_axis, arima_pred_aligned, '--', label='ARIMA', alpha=0.7)
    plt.plot(x_axis, lstm_predictions, '--', label='LSTM', alpha=0.7)
    plt.plot(x_axis, gru_predictions, '--', label='GRU', alpha=0.7)
    plt.plot(x_axis, simple_pred, '-', label='Simple Ensemble', linewidth=2, alpha=0.8)
    plt.plot(x_axis, weighted_pred, '-', label='Weighted Ensemble', linewidth=2, alpha=0.8)
    
    plt.title(f'Task C.6: Ensemble Model Predictions - {config.TICKER}', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Scaled Price')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(exp_dir, 'predictions_comparison.png')
    plt.savefig(plot_file, dpi=300)
    print(f"[OK] Plot saved to: {plot_file}")
    plt.close()
    
    print("\n" + "="*80)
    print("[OK] EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"[DIR] All results saved to: {exp_dir}")
    print("="*80)
    
    return all_results


#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------

if __name__ == "__main__":
    config = EnsembleExperimentConfig()
    results = run_ensemble_experiment(config)
