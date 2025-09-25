"""
task5_adv_prediction.py

Task C.5: Advanced Time Series Prediction
- Multistep Prediction
- Multivariate Prediction
- Combined Multivariate & Multistep Prediction

This script builds on the foundations of C.2, C.3, and C.4, extending the
framework to handle more complex forecasting scenarios.

Main components:
1.  `create_advanced_sequences`: A new sequence generation function capable of
    creating datasets for multistep prediction, where the target `y` is a
    sequence of future values instead of a single value.

2.  Experiment Runner (`if __name__ == '__main__':`):
    - Defines and runs three distinct experiments for Task C.5.
    - Leverages the updated `data_processing.py` for multivariate scaling.
    - Leverages `model_builder.py` for creating models with multistep outputs.
    - Saves results, models, and plots in a structured `task5_results` directory.

Author: Your Name
Date: 2025-09-15
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Project-specific imports
import data_processing
import model_builder
import visualization as plotting

# --- Configuration ---
BASE_RESULTS_DIR = 'task5_results'
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)


def create_advanced_sequences(data, sequence_length, target_col, n_future_steps):
    """
    Creates sequences for multistep time series forecasting.

    Unlike the previous `create_sequences`, this function generates a `y`
    that is a sequence of `n_future_steps` future values of the target column.

    Args:
        data (pd.DataFrame): Input data with features and target.
        sequence_length (int): Number of past time steps for each input sequence (X).
        target_col (str): The name of the column to be predicted.
        n_future_steps (int): Number of future time steps to predict (length of y).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X: Input sequences, shape (n_samples, sequence_length, n_features)
            - y: Target sequences, shape (n_samples, n_future_steps)
    """
    X, y = [], []
    # Find the index of the target column to extract it for the y array
    try:
        target_col_idx = data.columns.get_loc(target_col)
    except KeyError:
        raise ValueError(f"Target column '{target_col}' not found in data columns: {data.columns.tolist()}")

    # The main data array for creating sequences
    data_array = data.values

    for i in range(len(data_array) - sequence_length - n_future_steps + 1):
        # Input sequence (X): `sequence_length` timesteps of all features
        X.append(data_array[i:(i + sequence_length), :])

        # Target sequence (y): `n_future_steps` future values of the `target_col`
        y.append(data_array[(i + sequence_length):(i + sequence_length + n_future_steps), target_col_idx])

    if not X:
        # This can happen if the dataset is too short for the given sequence/future lengths
        return np.array([]).reshape(0, sequence_length, data.shape[1]), np.array([]).reshape(0, n_future_steps)

    return np.array(X), np.array(y)


def run_experiment(experiment_name, ticker, features, target_col,
                   sequence_length, n_future_steps,
                   model_config, training_config):
    """
    Main function to run a single experiment from data loading to evaluation.
    """
    print(f"--- Starting Experiment: {experiment_name} ---")

    # Create a dedicated directory for this experiment's results
    exp_dir = os.path.join(BASE_RESULTS_DIR, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Results will be saved in: {exp_dir}")

    # --- 1. Data Loading and Processing ---
    # For multivariate, scale all features together. Otherwise, scale per feature.
    scale_mode = 'all_features' if len(features) > 1 else 'per_feature'
    
    data_dict = data_processing.load_and_process_data(
        ticker=ticker,
        features=features,
        start_date='2015-01-01',
        end_date='2024-12-31',
        split_method='ratio',
        split_value=0.8,
        scale_features=True,
        scale_mode=scale_mode, # Use the new parameter
        cache_dir='data_cache'
    )
    train_df = data_dict['train_data']
    test_df = data_dict['test_data']
    scalers = data_dict['scalers']

    # --- 2. Sequence Creation ---
    print(f"Creating sequences with length {sequence_length} and {n_future_steps} future steps...")
    X_train, y_train = create_advanced_sequences(train_df[features], sequence_length, target_col, n_future_steps)
    X_test, y_test = create_advanced_sequences(test_df[features], sequence_length, target_col, n_future_steps)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Not enough data to create sequences. Skipping experiment.")
        return

    # --- 3. Model Building ---
    n_features = X_train.shape[2]
    build_info = model_builder.build_sequence_model(
        sequence_length=sequence_length,
        n_features=n_features,
        last_layer_units=n_future_steps, # Critical for multistep
        **model_config
    )
    model = build_info['model']
    print("\nModel Summary:")
    print(build_info['summary_str'])

    # --- 4. Model Training ---
    print("\nStarting model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    history = model.fit(
        X_train, y_train,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # --- 5. Evaluation and Plotting ---
    print("\nEvaluating model and generating plots...")
    predictions = model.predict(X_test)

    # Inverse transform predictions
    # If single-step, y_test and predictions are (n_samples, 1)
    # If multi-step, they are (n_samples, n_future_steps)
    target_scaler = scalers.get(target_col) if scale_mode == 'per_feature' else scalers.get('all')

    if not target_scaler:
         raise ValueError("Could not find the appropriate scaler for the target column.")

    # For 'all_features' mode, the scaler expects a 2D array with all features.
    # We need to create a dummy array of the correct shape, put our prediction
    # in the right column, and then inverse transform.
    if scale_mode == 'all_features':
        target_col_idx = features.index(target_col)
        n_features = len(features)
        n_steps = predictions.shape[1]  # number of future steps
        
        # Initialize output arrays
        inv_predictions = np.zeros_like(predictions)
        inv_y_test = np.zeros_like(y_test)

        for i in range(predictions.shape[0]):
            # Create dummy arrays with shape (n_steps, n_features)
            dummy_pred = np.zeros((n_steps, n_features))
            dummy_test = np.zeros((n_steps, n_features))
            
            # Put the predicted/actual values into the target column
            dummy_pred[:, target_col_idx] = predictions[i, :]
            dummy_test[:, target_col_idx] = y_test[i, :]

            # Inverse transform the whole block
            inv_pred_full = target_scaler.inverse_transform(dummy_pred)
            inv_test_full = target_scaler.inverse_transform(dummy_test)

            # Extract just the target column values
            inv_predictions[i, :] = inv_pred_full[:, target_col_idx]
            inv_y_test[i, :] = inv_test_full[:, target_col_idx]

    else: # per_feature
        inv_predictions = target_scaler.inverse_transform(predictions)
        inv_y_test = target_scaler.inverse_transform(y_test)

    # For plotting, we often just want to see the first prediction step
    plotting.plot_training_history(history, os.path.join(exp_dir, 'training_history.png'))
    plotting.plot_predictions_vs_actual(
        inv_y_test[:, 0], 
        inv_predictions[:, 0], 
        title=f'{experiment_name}: Predictions vs Actual (First Step)',
        save_path=os.path.join(exp_dir, 'predictions_vs_actual.png')
    )

    # --- 6. Save Artifacts ---
    print("Saving model and results...")
    model.save(os.path.join(exp_dir, 'model.h5'))
    
    # --- Save results, model, and plots ---
    
    # Create a dictionary to hold all results and configurations
    results = {
        'data_config': data_dict['config'],
        'model_config': {
            'layer_type': model_config['layer_type'],
            'sequence_length': sequence_length,
            'future_steps': n_future_steps,
            'layer_units': model_config['layer_units'],
            'dropout': model_config['dropout'],
            'bidirectional': model_config['bidirectional'],
        },
        'training_config': training_config,
        'history': {k: [float(val) for val in v] for k, v in history.history.items()},
        'test_loss': model.evaluate(X_test, y_test, verbose=0)[0]
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    # Common settings for all experiments
    TICKER = 'NVDA'
    SEQUENCE_LENGTH = 60  # Use 60 past days to predict

    # =========================================================================
    # Experiment 1: Multistep-Only Prediction
    # Goal: Predict the next 10 days' 'Close' price using past 'Close' prices.
    # =========================================================================
    print("\n--- Task C.5.1: Multistep Prediction ---")
    exp1_name = 'C5_1_multistep_close'
    exp1_features = ['Close']
    exp1_target = 'Close'
    exp1_future_steps = 10

    exp1_model_config = {
        'layer_type': 'lstm',
        'layer_units': [100, 50],
        'dropout': 0.2,
        'bidirectional': True,
    }
    exp1_training_config = {
        'epochs': 100,
        'batch_size': 32,
    }

    run_experiment(
        experiment_name=exp1_name,
        ticker=TICKER,
        features=exp1_features,
        target_col=exp1_target,
        sequence_length=SEQUENCE_LENGTH,
        n_future_steps=exp1_future_steps,
        model_config=exp1_model_config,
        training_config=exp1_training_config
    )

    # =========================================================================
    # Experiment 2: Multivariate-Only Prediction
    # Goal: Predict the next 1 day's 'Close' using multiple features.
    # =========================================================================
    print("\n--- Task C.5.2: Multivariate Prediction ---")
    exp2_name = 'C5_2_multivariate_close'
    exp2_features = ['Close', 'Volume', 'High', 'Low']  # Use actual available features
    exp2_target = 'Close'
    exp2_future_steps = 1 # Single step prediction

    exp2_model_config = {
        'layer_type': 'gru',
        'layer_units': [128, 64],
        'dropout': 0.3,
        'bidirectional': True,
    }
    exp2_training_config = {
        'epochs': 100,
        'batch_size': 32,
    }

    run_experiment(
        experiment_name=exp2_name,
        ticker=TICKER,
        features=exp2_features,
        target_col=exp2_target,
        sequence_length=SEQUENCE_LENGTH,
        n_future_steps=exp2_future_steps,
        model_config=exp2_model_config,
        training_config=exp2_training_config
    )

    # =========================================================================
    # Experiment 3: Combined Multivariate and Multistep Prediction
    # Goal: Predict next 10 days' 'Close' using multiple features.
    # =========================================================================
    print("\n--- Task C.5.3: Combined Multivariate and Multistep Prediction ---")
    exp3_name = 'C5_3_multivar_multistep_close'
    exp3_features = ['Close', 'Volume', 'High', 'Low']  # Use actual available features
    exp3_target = 'Close'
    exp3_future_steps = 10

    exp3_model_config = {
        'layer_type': 'lstm',
        'layer_units': [128, 64, 32],
        'dropout': 0.3,
        'bidirectional': True,
    }
    exp3_training_config = {
        'epochs': 150, # More complex task may need more training
        'batch_size': 32,
    }

    run_experiment(
        experiment_name=exp3_name,
        ticker=TICKER,
        features=exp3_features,
        target_col=exp3_target,
        sequence_length=SEQUENCE_LENGTH,
        n_future_steps=exp3_future_steps,
        model_config=exp3_model_config,
        training_config=exp3_training_config
    )

    print("\nAll Task C.5 experiments have been executed.")
