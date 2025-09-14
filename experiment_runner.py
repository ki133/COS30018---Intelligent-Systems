"""
experiment_runner.py
Task C.4: Run multiple deep learning model experiments (LSTM / GRU / RNN)

Features:
- Loads processed stock data using existing enhanced data loader (data_processing.py)
- Generates supervised sequences (sliding window) for time series prediction
- Splits into train/validation/test sets using time order
- Builds models via model_builder.build_sequence_model
- Trains with early stopping + reduce LR on plateau
- Logs each experiment (config + metrics + history) into a timestamped folder
- Supports batch execution of multiple configurations
- Saves:
    * model_summary.txt
    * config.json
    * training_history.csv
    * metrics.json
    * best_model.keras (saved Keras model)

Usage (example quick run):
    python experiment_runner.py --ticker CBA.AX --epochs 3 --quick

Vietnamese explanatory comments included for learning clarity.
"""
from __future__ import annotations
import os, json, argparse, math, datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# Local imports
from model_builder import build_sequence_model
from data_processing import load_and_process_data  # Assuming Task C.2 file name

# ------------------------------------------------------
# Configuration dataclass
# ------------------------------------------------------
@dataclass
class ExperimentConfig:
    ticker: str = 'CBA.AX'
    start_date: str = '2023-01-01'
    end_date: str = '2024-01-01'
    price_column: str = 'Close'
    sequence_length: int = 60  # số ngày nhìn lại
    predict_horizon: int = 1   # dự đoán 1 ngày tới
    feature_columns: List[str] = None  # filled after load
    layer_type: str = 'lstm'
    layer_units: List[int] = None
    dropout: float = 0.2
    recurrent_dropout: float = 0.0
    bidirectional: bool = False
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    patience: int = 5
    min_delta: float = 1e-4
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    validation_split: float = 0.1
    test_split: float = 0.15  # portion of tail for testing
    loss: str = 'mean_squared_error'
    metrics: List[str] = None
    output_activation: str = 'linear'

    def finalize(self, n_features: int):
        if self.feature_columns is None:
            # will fill after data load
            pass
        if self.layer_units is None:
            self.layer_units = [64, 32]
        if self.metrics is None:
            self.metrics = ['mae']

# ------------------------------------------------------
# Data preparation utilities
# ------------------------------------------------------

def create_supervised_sequences(values: np.ndarray, seq_len: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 1D (or 2D feature) time series into (X, y) supervised windows.
    values: shape (N, F)
    Returns X shape (M, seq_len, F), y shape (M, ) for single-step horizon.
    """
    X, y = [], []
    for i in range(seq_len, len(values) - horizon + 1):
        X.append(values[i - seq_len:i])
        # Predict using target column (assume column 0 if multiple features)
        y.append(values[i + horizon - 1, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ------------------------------------------------------
# Training single experiment
# ------------------------------------------------------

def run_single_experiment(cfg: ExperimentConfig, output_dir: str) -> Dict[str, Any]:
    # 1. Load enhanced processed data
    processed = load_and_process_data(
        ticker=cfg.ticker,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        features=[cfg.price_column]  # start with primary feature first
    )
    # The C.2 function returns scaled train/test separately. We'll recombine (time order) for windowing.
    train_df = processed['train_data'].copy()
    test_df = processed['test_data'].copy()
    # Keep ordering: train then test
    df = pd.concat([train_df, test_df], axis=0)

    # Choose features: for simplicity include scaled price and engineered features if exist
    candidate_columns = [col for col in df.columns if col not in ('Date',)]
    # For now only use the first (target) feature from processed pipeline (can extend later)
    feature_columns = [cfg.price_column]
    cfg.feature_columns = feature_columns
    values = df[feature_columns].values.astype(np.float32)

    # 2. Build supervised windows
    X, y = create_supervised_sequences(values, cfg.sequence_length, cfg.predict_horizon)

    # 3. Split train/val/test (time-based)
    test_size = int(len(X) * cfg.test_split)
    X_train_full, X_test = X[:-test_size], X[-test_size:]
    y_train_full, y_test = y[:-test_size], y[-test_size:]

    val_size = int(len(X_train_full) * cfg.validation_split)
    X_train, X_val = X_train_full[:-val_size], X_train_full[-val_size:]
    y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]

    # 4. Build model
    build_info = build_sequence_model(
        sequence_length=cfg.sequence_length,
        n_features=X.shape[2],
        layer_type=cfg.layer_type,
        layer_units=cfg.layer_units,
        dropout=cfg.dropout,
        recurrent_dropout=cfg.recurrent_dropout,
        bidirectional=cfg.bidirectional,
        last_layer_units=1,
        output_activation=cfg.output_activation,
        optimizer=cfg.optimizer,
        learning_rate=cfg.learning_rate,
        loss=cfg.loss,
        metrics=cfg.metrics,
    )
    model = build_info['model']

    # 5. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.patience, restore_best_weights=True, min_delta=cfg.min_delta),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=cfg.reduce_lr_factor, patience=cfg.reduce_lr_patience, verbose=1)
    ]

    # 6. Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Evaluate
    test_eval = model.evaluate(X_test, y_test, verbose=0)
    metrics_names = model.metrics_names
    metrics_dict = {name: float(val) for name, val in zip(metrics_names, test_eval)}

    # 8. Predictions for additional metrics (RMSE, MAPE)
    y_pred = model.predict(X_test, verbose=0).reshape(-1)
    rmse = float(np.sqrt(np.mean((y_pred - y_test)**2)))
    mae = float(np.mean(np.abs(y_pred - y_test)))
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100)
    metrics_dict.update({'rmse': rmse, 'mape': mape})

    # 9. Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(build_info['summary_str'])
    # History
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    # Metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2)
    # Save model
    model.save(os.path.join(output_dir, 'best_model.keras'))

    return {
        'config': asdict(cfg),
        'metrics': metrics_dict,
        'history_tail': hist_df.tail(1).to_dict(orient='records')[0],
        'output_dir': output_dir
    }

# ------------------------------------------------------
# Batch runner
# ------------------------------------------------------

def run_batch(configs: List[ExperimentConfig], base_dir: str) -> List[Dict[str, Any]]:
    results = []
    for cfg in configs:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{cfg.layer_type.upper()}_{'-'.join(map(str,cfg.layer_units))}_seq{cfg.sequence_length}_ep{cfg.epochs}".replace('/', '-')
        out_dir = os.path.join(base_dir, f"{timestamp}_{exp_name}")
        print(f"\n🚀 Running experiment: {exp_name}\n   Output -> {out_dir}")
        res = run_single_experiment(cfg, out_dir)
        results.append(res)
    # Aggregate metrics
    agg = pd.DataFrame([r['metrics'] | {'model': r['config']['layer_type'], 'layers': '-'.join(map(str,r['config']['layer_units']))} for r in results])
    agg.to_csv(os.path.join(base_dir, 'batch_summary.csv'), index=False)
    return results

# ------------------------------------------------------
# CLI
# ------------------------------------------------------

def build_cli_configs(args) -> List[ExperimentConfig]:
    base = ExperimentConfig(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        sequence_length=args.sequence_length,
        predict_horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    configs: List[ExperimentConfig] = []
    if args.quick:
        # Minimal quick set
        configs.append(ExperimentConfig(layer_type='lstm', layer_units=[64,32], epochs=args.epochs, batch_size=args.batch_size, sequence_length=args.sequence_length))
        configs.append(ExperimentConfig(layer_type='gru', layer_units=[64], epochs=args.epochs, batch_size=args.batch_size, sequence_length=args.sequence_length))
        configs.append(ExperimentConfig(layer_type='rnn', layer_units=[128,64], epochs=args.epochs, batch_size=args.batch_size, sequence_length=args.sequence_length))
    else:
        for lt in ['lstm','gru','rnn']:
            for stack in ([64],[64,32],[128,64]):
                cfg = ExperimentConfig(layer_type=lt, layer_units=stack, epochs=args.epochs, batch_size=args.batch_size, sequence_length=args.sequence_length)
                configs.append(cfg)
    return configs

def main():
    parser = argparse.ArgumentParser(description='Task C.4 Experiment Runner')
    parser.add_argument('--ticker', default='CBA.AX')
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default='2024-01-01')
    parser.add_argument('--sequence-length', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', default='experiments')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    configs = build_cli_configs(args)
    results = run_batch(configs, args.output)
    print('\n✅ All experiments completed. Summary saved to batch_summary.csv')
    for r in results:
        print(f" - {r['config']['layer_type'].upper()} {r['config']['layer_units']} RMSE={r['metrics']['rmse']:.4f} MAPE={r['metrics']['mape']:.2f}% -> {r['output_dir']}")

if __name__ == '__main__':
    main()
