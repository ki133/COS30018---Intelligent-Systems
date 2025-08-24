# Simplified P1 comparison using existing v0.1 data preprocessing but P1 model architecture
import sys
import os
sys.path.append('..')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from collections import deque

# Set seed for reproducibility
np.random.seed(314)
tf.random.set_seed(314)

def load_data_p1_style(ticker, n_steps=50, lookup_step=1, test_size=0.2):
    """Load data in P1 style but using yfinance and similar preprocessing to v0.1"""
    
    # Download data (same period as v0.1)
    train_start = '2020-01-01'
    train_end = '2023-08-01'
    test_start = '2023-08-02' 
    test_end = '2024-07-02'
    
    # Get training data
    train_data = yf.download(ticker, train_start, train_end)
    test_data = yf.download(ticker, test_start, test_end)
    
    # Combine for continuous processing
    full_data = pd.concat([train_data, test_data])
    
    # Use multiple features (P1 approach)
    feature_columns = ['Close', 'Volume', 'Open', 'High', 'Low']
    df = full_data[feature_columns].copy()
    
    # Scale all features
    scalers = {}
    for col in feature_columns:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        scalers[col] = scaler
    
    # Create target (shifted Close price)
    df['future'] = df['Close'].shift(-lookup_step)
    df.dropna(inplace=True)
    
    # Create sequences
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    
    for i, row in df.iterrows():
        seq_data = row[feature_columns].values
        target = row['future']
        sequences.append(seq_data)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    
    # Split into X and y
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split by date (P1 style)
    train_size = len(train_data) - n_steps
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scalers': scalers,
        'last_sequence': X[-1:],  # For next day prediction
        'feature_columns': feature_columns
    }

def create_p1_model(n_steps, n_features):
    """Create P1-style model with multiple layers and features"""
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=256, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.4))
    
    # Second LSTM layer  
    model.add(LSTM(units=256, return_sequences=False))
    model.add(Dropout(0.4))
    
    # Output layer
    model.add(Dense(1, activation="linear"))
    
    # Compile with different loss function (P1 uses huber_loss)
    model.compile(loss='huber', metrics=['mean_absolute_error'], optimizer='adam')
    
    return model

# Main comparison
print("P1 Reference Implementation Comparison")
print("="*50)

ticker = "CBA.AX"
n_steps = 50
epochs = 25

print(f"Loading data for {ticker}...")

# Load data P1 style
data = load_data_p1_style(ticker, n_steps)

print(f"Training samples: {len(data['X_train'])}")
print(f"Testing samples: {len(data['X_test'])}")
print(f"Features: {data['feature_columns']}")
print(f"Input shape: {data['X_train'].shape}")

# Create P1 model
model = create_p1_model(n_steps, len(data['feature_columns']))

print("\nP1 Model Architecture:")
model.summary()

print(f"\nTraining for {epochs} epochs...")

# Train model
history = model.fit(
    data['X_train'], data['y_train'],
    batch_size=64,
    epochs=epochs,
    validation_data=(data['X_test'], data['y_test']),
    verbose=1
)

print("\nMaking predictions...")

# Test predictions
test_pred = model.predict(data['X_test'])

# Next day prediction
next_day_pred = model.predict(data['last_sequence'])

# Scale back predictions
close_scaler = data['scalers']['Close']
test_pred_actual = close_scaler.inverse_transform(test_pred)
next_day_actual = close_scaler.inverse_transform(next_day_pred)
y_test_actual = close_scaler.inverse_transform(data['y_test'].reshape(-1, 1))

# Calculate metrics
mse = mean_squared_error(y_test_actual, test_pred_actual)
mae = mean_absolute_error(y_test_actual, test_pred_actual)

print(f"\nP1 Results:")
print(f"Next day prediction: ${next_day_actual[0][0]:.2f}")
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: ${mae:.2f}")

print("\nComparison with v0.1:")
print("v0.1 prediction: $114.64")
print(f"P1 prediction: ${next_day_actual[0][0]:.2f}")
print(f"Difference: ${abs(next_day_actual[0][0] - 114.64):.2f}")

# Key differences
print("\nKey Differences:")
print("1. Features: v0.1 uses only Close price, P1 uses [Close, Volume, Open, High, Low]")
print("2. Model: v0.1 has 3 LSTM layers (50 units), P1 has 2 LSTM layers (256 units)")
print("3. Loss: v0.1 uses MSE, P1 uses Huber loss")
print("4. Dropout: v0.1 uses 0.2, P1 uses 0.4")
print("5. Architecture: P1 is more complex with more parameters")

# Plot comparison
print("\nGenerating plots...")

plt.figure(figsize=(12, 8))

# Plot actual vs predicted prices
plt.plot(y_test_actual, color="black", label=f"Actual {ticker} Price", linewidth=2)
plt.plot(test_pred_actual, color="blue", label=f"P1 Predicted {ticker} Price", linewidth=2)
plt.title(f"P1 Reference: {ticker} Share Price Prediction")
plt.xlabel("Time")
plt.ylabel(f"{ticker} Share Price (AUD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nP1 Reference visualization completed!")
