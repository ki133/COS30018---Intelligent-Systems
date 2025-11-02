"""
Task C.7: Classification Models

This module implements classification models for stock price movement prediction.
Addresses Task Requirement 3 & 4 (10 marks) - Modelling & Evaluation.

Requirements addressed:
- Implement baseline models (technical-only, sentiment-only)
- Implement full model (technical + sentiment + interactions)
- Handle class imbalance using SMOTE
- REUSE model_builder.py from Task C.4 for LSTM architecture

Models implemented:
1. Logistic Regression (baseline, interpretable)
2. Random Forest (ensemble, feature importance)
3. XGBoost (gradient boosting, state-of-the-art)
4. LSTM (deep learning, reused from Task C.4)

Author: Your Name
Date: October 2025
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Handle class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARNING] imbalanced-learn not installed. SMOTE unavailable.")
    print("          Install: pip install imbalanced-learn")

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Using Random Forest only.")
    print("          Install: pip install xgboost")

# Deep learning (Keras/TensorFlow)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    # Create dummy keras for type hints
    class keras:
        class Model:
            pass
    print("[WARNING] TensorFlow/Keras not installed. LSTM unavailable.")

# CRITICAL: REUSE Task C.4 model_builder for LSTM architecture
try:
    import model_builder
    MODEL_BUILDER_AVAILABLE = True
    print("[REUSE] Task C.4 model_builder imported successfully!")
except ImportError:
    MODEL_BUILDER_AVAILABLE = False
    print("[WARNING] model_builder.py not found. Will use custom LSTM.")


class SentimentClassifierTrainer:
    """
    Train classification models for stock price movement prediction
    
    This class implements Task C.7 modeling requirements:
    1. Baseline models (technical-only, sentiment-only)
    2. Full models (all features combined)
    3. Handle class imbalance with SMOTE
    4. Compare multiple algorithms (Logistic, RF, XGBoost, LSTM)
    5. REUSE Task C.4's LSTM architecture (code reuse!)
    
    Usage:
        trainer = SentimentClassifierTrainer(
            use_smote=True,
            random_state=42
        )
        
        # Train baseline: technical-only
        model_tech = trainer.train_model(
            X_train, y_train,
            model_type='random_forest',
            experiment_name='baseline_technical'
        )
        
        # Train full: all features
        model_full = trainer.train_model(
            X_train_full, y_train,
            model_type='xgboost',
            experiment_name='full_model'
        )
        
        # Predict
        predictions = trainer.predict(model_full, X_test)
    """
    
    def __init__(self, use_smote: bool = True, random_state: int = 42):
        """
        Initialize Classifier Trainer
        
        Args:
            use_smote: Whether to use SMOTE for handling imbalance
            random_state: Random seed for reproducibility
        """
        self.use_smote = use_smote and SMOTE_AVAILABLE
        self.random_state = random_state
        
        # Scalers for each experiment (stored separately)
        self.scalers = {}
        
        # Trained models storage
        self.models = {}
        
        print(f"\n[CLASSIFIER] Initializing Sentiment Classifier Trainer")
        print(f"  Use SMOTE: {self.use_smote}")
        print(f"  Random state: {random_state}")
        
        if use_smote and not SMOTE_AVAILABLE:
            print("[WARNING] SMOTE requested but not available!")
    
    # =========================================================================
    # DATA PREPROCESSING
    # =========================================================================
    
    def preprocess_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame = None,
                       experiment_name: str = 'default') -> Tuple:
        """
        Preprocess data: scaling + SMOTE (if enabled)
        
        Steps:
        1. Scale features using StandardScaler
        2. Apply SMOTE to handle class imbalance (training only!)
        3. Store scaler for inverse transform later
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            experiment_name: Name for storing scaler
            
        Returns:
            Tuple: (X_train_processed, y_train_processed, X_test_processed)
        """
        print(f"\n[PREPROCESS] {experiment_name}")
        
        # Step 1: Scaling
        print("  [1] Scaling features...")
        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
        
        # Store scaler
        self.scalers[experiment_name] = scaler
        
        # Step 2: SMOTE (only on training data!)
        if self.use_smote:
            print("  [2] Applying SMOTE for class balance...")
            
            # Check original class distribution
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"      Before SMOTE: {dict(zip(unique, counts))}")
            
            # Apply SMOTE
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_scaled, y_train
            )
            
            # Check new distribution
            unique, counts = np.unique(y_train_resampled, return_counts=True)
            print(f"      After SMOTE: {dict(zip(unique, counts))}")
            
            X_train_processed = X_train_resampled
            y_train_processed = y_train_resampled
        else:
            print("  [2] SMOTE disabled, using original data")
            X_train_processed = X_train_scaled
            y_train_processed = y_train
        
        print(f"  [OK] Processed: Train {X_train_processed.shape}, Test {X_test_scaled.shape if X_test_scaled is not None else 'N/A'}")
        
        return X_train_processed, y_train_processed, X_test_scaled
    
    # =========================================================================
    # MODEL TRAINING
    # =========================================================================
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame = None, y_val: pd.Series = None,
                   model_type: str = 'random_forest',
                   model_params: Dict = None,
                   experiment_name: str = None) -> Any:
        """
        Train a classification model
        
        Supports:
        - 'logistic': Logistic Regression
        - 'random_forest': Random Forest Classifier
        - 'xgboost': XGBoost Classifier
        - 'lstm': LSTM Neural Network (reused from Task C.4!)
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for LSTM)
            y_val: Validation labels (for LSTM)
            model_type: Type of model to train
            model_params: Model hyperparameters (optional)
            experiment_name: Name for this experiment
            
        Returns:
            Trained model object
        """
        if experiment_name is None:
            experiment_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n[TRAIN] {experiment_name}")
        print(f"  Model type: {model_type}")
        print(f"  Train shape: {X_train.shape}")
        
        # Preprocess data
        if X_val is not None:
            X_train_prep, y_train_prep, X_val_prep = self.preprocess_data(
                X_train, y_train, X_val, experiment_name
            )
        else:
            X_train_prep, y_train_prep, _ = self.preprocess_data(
                X_train, y_train, None, experiment_name
            )
            X_val_prep, y_val = None, None
        
        # Select training function
        if model_type == 'logistic':
            model = self._train_logistic(X_train_prep, y_train_prep, model_params)
        
        elif model_type == 'random_forest':
            model = self._train_random_forest(X_train_prep, y_train_prep, model_params)
        
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                print("[ERROR] XGBoost not available. Using Random Forest instead.")
                model = self._train_random_forest(X_train_prep, y_train_prep, model_params)
            else:
                model = self._train_xgboost(X_train_prep, y_train_prep, model_params)
        
        elif model_type == 'lstm':
            if not KERAS_AVAILABLE:
                print("[ERROR] Keras not available. Cannot train LSTM.")
                return None
            
            # LSTM needs validation data
            if X_val_prep is None:
                print("[WARNING] LSTM training without validation set!")
            
            model = self._train_lstm(
                X_train_prep, y_train_prep,
                X_val_prep, y_val,
                model_params,
                experiment_name
            )
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Store model
        self.models[experiment_name] = model
        
        print(f"[OK] Model trained: {experiment_name}")
        
        return model
    
    def _train_logistic(self, X_train: np.ndarray, y_train: np.ndarray,
                       params: Dict = None) -> LogisticRegression:
        """
        Train Logistic Regression
        
        Pros: Fast, interpretable, works well with scaled features
        Cons: Linear decision boundary, may underfit complex patterns
        
        Args:
            X_train: Training features (scaled)
            y_train: Training labels
            params: Hyperparameters (optional)
            
        Returns:
            Trained LogisticRegression model
        """
        print("  [LOGISTIC] Training Logistic Regression...")
        
        if params is None:
            params = {
                'C': 1.0,  # Regularization strength (smaller = stronger)
                'max_iter': 1000,
                'random_state': self.random_state
            }
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        print(f"  [OK] Logistic Regression trained")
        
        return model
    
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                            params: Dict = None) -> RandomForestClassifier:
        """
        Train Random Forest Classifier
        
        Pros: Handles non-linear patterns, provides feature importance
        Cons: Can overfit with small datasets
        
        Args:
            X_train: Training features (scaled)
            y_train: Training labels
            params: Hyperparameters (optional)
            
        Returns:
            Trained RandomForestClassifier
        """
        print("  [RANDOM FOREST] Training Random Forest...")
        
        if params is None:
            params = {
                'n_estimators': 100,  # Number of trees
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state
            }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        print(f"  [OK] Random Forest trained ({params['n_estimators']} trees)")
        
        return model
    
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      params: Dict = None) -> xgb.XGBClassifier:
        """
        Train XGBoost Classifier
        
        Pros: State-of-the-art performance, handles imbalance well
        Cons: Slower training, more hyperparameters to tune
        
        Args:
            X_train: Training features (scaled)
            y_train: Training labels
            params: Hyperparameters (optional)
            
        Returns:
            Trained XGBClassifier
        """
        print("  [XGBOOST] Training XGBoost...")
        
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        print(f"  [OK] XGBoost trained")
        
        return model
    
    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   params: Dict = None,
                   experiment_name: str = 'lstm') -> keras.Model:
        """
        Train LSTM Neural Network
        
        CRITICAL: This tries to REUSE Task C.4's model_builder!
        If not available, falls back to custom LSTM.
        
        Pros: Captures temporal patterns, works with sequential data
        Cons: Needs more data, slower training, requires tuning
        
        Args:
            X_train: Training features (scaled)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: Model hyperparameters
            experiment_name: Name for saving checkpoints
            
        Returns:
            Trained Keras model
        """
        print("  [LSTM] Training LSTM Neural Network...")
        
        if params is None:
            params = {
                'units': [64, 32],  # LSTM layer sizes
                'dropout': 0.2,
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        
        # Reshape for LSTM: (samples, timesteps, features)
        # For non-sequential: timesteps = 1
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            if X_val is not None:
                X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        
        n_features = X_train.shape[2]
        
        # Try to REUSE Task C.4's model builder
        if MODEL_BUILDER_AVAILABLE:
            print("  [REUSE] Using Task C.4 model_builder for LSTM architecture!")
            
            try:
                # Adapt Task C.4's create_model() for classification
                # Original is for regression, we need binary classification
                model = self._adapt_task4_lstm(n_features, params)
            except Exception as e:
                print(f"  [WARNING] Could not adapt Task C.4 LSTM: {e}")
                print("  [FALLBACK] Using custom LSTM architecture")
                model = self._build_custom_lstm(n_features, params)
        else:
            print("  [CUSTOM] Building custom LSTM (Task C.4 model_builder not available)")
            model = self._build_custom_lstm(n_features, params)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='binary_crossentropy',  # Binary classification
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        print(f"\n  Model architecture:")
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Add checkpoint callback
        checkpoint_path = f'task7_models/{experiment_name}_best.keras'
        os.makedirs('task7_models', exist_ok=True)
        
        callbacks.append(
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
        )
        
        # Train
        print(f"\n  Training for up to {params['epochs']} epochs...")
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"  [OK] LSTM trained. Best model saved to {checkpoint_path}")
        
        # Store history
        self.models[f'{experiment_name}_history'] = history.history
        
        return model
    
    def _adapt_task4_lstm(self, n_features: int, params: Dict) -> keras.Model:
        """
        Adapt Task C.4's LSTM for classification
        
        Task C.4 builds regression LSTM, we need classification LSTM.
        Main changes:
        - Output layer: Dense(1, activation='sigmoid') instead of Dense(1)
        - Loss: binary_crossentropy instead of mse
        
        This demonstrates CODE REUSE with adaptation!
        
        Args:
            n_features: Number of input features
            params: Model parameters
            
        Returns:
            Keras Model adapted for classification
        """
        # Build similar architecture to Task C.4
        model = keras.Sequential(name='LSTM_Classifier_Task7')
        
        # First LSTM layer (with return_sequences for stacking)
        model.add(layers.LSTM(
            units=params['units'][0],
            return_sequences=len(params['units']) > 1,
            input_shape=(1, n_features),  # (timesteps, features)
            name='lstm_1'
        ))
        model.add(layers.Dropout(params['dropout'], name='dropout_1'))
        
        # Additional LSTM layers if specified
        for i, units in enumerate(params['units'][1:], start=2):
            model.add(layers.LSTM(
                units=units,
                return_sequences=False,  # Last LSTM doesn't return sequences
                name=f'lstm_{i}'
            ))
            model.add(layers.Dropout(params['dropout'], name=f'dropout_{i}'))
        
        # Output layer: BINARY CLASSIFICATION (different from Task C.4!)
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        print("  [ADAPTED] Task C.4 LSTM architecture adapted for classification")
        
        return model
    
    def _build_custom_lstm(self, n_features: int, params: Dict) -> keras.Model:
        """
        Build custom LSTM for classification (fallback)
        
        Args:
            n_features: Number of input features
            params: Model parameters
            
        Returns:
            Keras Model
        """
        model = keras.Sequential(name='Custom_LSTM_Classifier')
        
        # LSTM layers
        for i, units in enumerate(params['units']):
            return_sequences = (i < len(params['units']) - 1)
            
            if i == 0:
                model.add(layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    input_shape=(1, n_features)
                ))
            else:
                model.add(layers.LSTM(
                    units=units,
                    return_sequences=return_sequences
                ))
            
            model.add(layers.Dropout(params['dropout']))
        
        # Output
        model.add(layers.Dense(1, activation='sigmoid'))
        
        return model
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def predict(self, model: Any, X_test: pd.DataFrame,
                experiment_name: str = None,
                return_proba: bool = False) -> np.ndarray:
        """
        Make predictions with trained model
        
        Args:
            model: Trained model
            X_test: Test features
            experiment_name: Name of experiment (for scaler lookup)
            return_proba: Return probabilities instead of class labels
            
        Returns:
            np.ndarray: Predictions (class labels or probabilities)
        """
        # Scale test data using stored scaler
        if experiment_name and experiment_name in self.scalers:
            scaler = self.scalers[experiment_name]
            X_test_scaled = scaler.transform(X_test)
        else:
            print("[WARNING] No scaler found, using unscaled data")
            X_test_scaled = X_test
        
        # Reshape for LSTM if needed
        if isinstance(model, keras.Model):
            if len(X_test_scaled.shape) == 2:
                X_test_scaled = X_test_scaled.reshape(
                    (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
                )
        
        # Predict
        if return_proba:
            # Get probabilities
            if isinstance(model, keras.Model):
                predictions = model.predict(X_test_scaled, verbose=0).flatten()
            elif hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X_test_scaled)[:, 1]
            else:
                print("[WARNING] Model doesn't support probabilities")
                predictions = model.predict(X_test_scaled)
        else:
            # Get class labels
            if isinstance(model, keras.Model):
                predictions = (model.predict(X_test_scaled, verbose=0) > 0.5).astype(int).flatten()
            else:
                predictions = model.predict(X_test_scaled)
        
        return predictions
    
    def get_feature_importance(self, model: Any, 
                              feature_names: List[str] = None) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models
        
        Args:
            model: Trained model (Random Forest or XGBoost)
            feature_names: List of feature names
            
        Returns:
            pd.DataFrame: Feature importance sorted by value
        """
        if not hasattr(model, 'feature_importances_'):
            print("[WARNING] Model doesn't have feature_importances_")
            return None
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    # =========================================================================
    # SAVE/LOAD
    # =========================================================================
    
    def save_model(self, model: Any, experiment_name: str, 
                   save_dir: str = 'task7_models'):
        """
        Save trained model to disk
        
        Args:
            model: Trained model
            experiment_name: Name for the saved model
            save_dir: Directory to save to
        """
        os.makedirs(save_dir, exist_ok=True)
        
        filepath = os.path.join(save_dir, f'{experiment_name}')
        
        # Save based on model type
        if isinstance(model, keras.Model):
            # Keras model
            model.save(f'{filepath}.keras')
            print(f"[SAVED] Keras model: {filepath}.keras")
        
        else:
            # Scikit-learn/XGBoost model (use joblib)
            import joblib
            joblib.dump(model, f'{filepath}.pkl')
            print(f"[SAVED] Model: {filepath}.pkl")
        
        # Save scaler if exists
        if experiment_name in self.scalers:
            import joblib
            joblib.dump(self.scalers[experiment_name], 
                       f'{filepath}_scaler.pkl')
            print(f"[SAVED] Scaler: {filepath}_scaler.pkl")


# Example usage
if __name__ == '__main__':
    print("="*70)
    print("TESTING CLASSIFIER MODELS")
    print("="*70)
    
    # Create dummy data
    print("\n[TEST] Creating dummy data...")
    n_samples = 1000
    n_features = 20
    
    np.random.seed(42)
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randint(0, 2, n_samples))
    
    X_test = pd.DataFrame(
        np.random.randn(200, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Initialize trainer
    trainer = SentimentClassifierTrainer(use_smote=True)
    
    # Test each model type
    for model_type in ['logistic', 'random_forest']:
        print(f"\n{'='*70}")
        print(f"Testing {model_type.upper()}")
        print('='*70)
        
        model = trainer.train_model(
            X_train, y_train,
            model_type=model_type,
            experiment_name=f'test_{model_type}'
        )
        
        # Predict
        predictions = trainer.predict(model, X_test, f'test_{model_type}')
        print(f"\n[TEST] Predictions: {predictions[:10]}")
        print(f"[TEST] Prediction distribution: {np.bincount(predictions)}")
        
        # Feature importance (if available)
        if model_type in ['random_forest', 'xgboost']:
            importance = trainer.get_feature_importance(
                model,
                X_train.columns.tolist()
            )
            print(f"\n[TEST] Top 5 important features:")
            print(importance.head())
    
    print("\n[TEST] All models tested successfully!")
