# Simple test to compare P1 vs v0.1
import os
import time
from stock_prediction import create_model, load_data

# Modified parameters for quick test (reduce epochs to compare approaches)
N_STEPS = 50
LOOKUP_STEP = 1  # Same as v0.1: next day prediction
SCALE = True
SHUFFLE = True
SPLIT_BY_DATE = False
TEST_SIZE = 0.2
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# Model parameters
N_LAYERS = 2
UNITS = 256
DROPOUT = 0.4
BIDIRECTIONAL = False

# Training parameters (reduced for quick test)
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 25  # Same as v0.1 for fair comparison

# Same stock as v0.1
ticker = "CBA.AX"

print(f"P1 Reference - Loading data for {ticker}...")

# Load the data
try:
    data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, 
                    shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                    feature_columns=FEATURE_COLUMNS)
    
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Testing samples: {len(data['X_test'])}")
    print(f"Features used: {FEATURE_COLUMNS}")
    
    # Create model
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, 
                        n_layers=N_LAYERS, dropout=DROPOUT, optimizer=OPTIMIZER, 
                        bidirectional=BIDIRECTIONAL)
    
    print(f"\nModel architecture:")
    model.summary()
    
    print(f"\nTraining for {EPOCHS} epochs...")
    
    # Train model
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        verbose=1)
    
    # Make predictions
    predictions = model.predict(data["X_test"])
    
    # Get last sequence for next day prediction
    last_sequence = data["last_sequence"][-N_STEPS:].reshape((1, N_STEPS, len(FEATURE_COLUMNS)))
    next_day_prediction = model.predict(last_sequence)
    
    # Scale back the prediction
    if SCALE:
        next_day_prediction = data["column_scaler"]["adjclose"].inverse_transform(next_day_prediction)
    
    print(f"\nP1 Next day prediction: ${next_day_prediction[0][0]:.2f}")
    
    # Calculate some basic metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    y_test_scaled = data["y_test"]
    if SCALE:
        y_test_actual = data["column_scaler"]["adjclose"].inverse_transform(y_test_scaled.reshape(-1, 1))
        pred_actual = data["column_scaler"]["adjclose"].inverse_transform(predictions)
    else:
        y_test_actual = y_test_scaled
        pred_actual = predictions
    
    mse = mean_squared_error(y_test_actual, pred_actual)
    mae = mean_absolute_error(y_test_actual, pred_actual)
    
    print(f"P1 Test MSE: {mse:.4f}")
    print(f"P1 Test MAE: {mae:.4f}")
    
except Exception as e:
    print(f"Error running P1: {e}")
    import traceback
    traceback.print_exc()
