"""
ensemble_methods.py
Task C.6: Ensemble Methods - Combining Multiple Models

This module implements different ensemble strategies to combine predictions
from statistical models (ARIMA/SARIMA) and deep learning models (LSTM/GRU/RNN).

Ensemble Methods Implemented:
1. Simple Average: Equal weight to all models
2. Weighted Average: Custom weights based on performance
3. Stacking: Meta-learner trained on base model predictions

Reference Article:
https://medium.com/analytics-vidhya/combining-time-series-analysis-with-artificial-intelligence-the-future-of-forecasting-5196f57db913

Author: [Your Name]
Date: 14/10/2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


#------------------------------------------------------------------------------
# Ensemble Strategy Base Class
#------------------------------------------------------------------------------

class EnsembleStrategy:
    """
    Base class for ensemble strategies
    Provides interface for combining predictions from multiple models
    """
    
    def __init__(self, name: str = "Base Ensemble"):
        self.name = name
        self.is_fitted = False
        
    def fit(self, predictions: List[np.ndarray], y_true: np.ndarray):
        """
        Fit the ensemble strategy (if needed)
        
        Args:
            predictions: List of prediction arrays from different models
            y_true: True target values
        """
        raise NotImplementedError
        
    def combine(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Combine predictions from multiple models
        
        Args:
            predictions: List of prediction arrays from different models
            
        Returns:
            Combined prediction array
        """
        raise NotImplementedError


#------------------------------------------------------------------------------
# Simple Average Ensemble
#------------------------------------------------------------------------------

class SimpleAverageEnsemble(EnsembleStrategy):
    """
    Simple Average Ensemble Strategy
    
    Combines predictions by taking the arithmetic mean.
    All models have equal weight = 1/n where n is number of models.
    
    This is the simplest ensemble method and often works surprisingly well!
    
    Formula: y_pred = (pred1 + pred2 + ... + predn) / n
    
    Advantages:
    - No training needed
    - No overfitting risk
    - Easy to understand and implement
    
    Example:
        ensemble = SimpleAverageEnsemble()
        combined = ensemble.combine([arima_pred, lstm_pred, gru_pred])
    """
    
    def __init__(self):
        super().__init__(name="Simple Average")
        
    def fit(self, predictions: List[np.ndarray], y_true: np.ndarray):
        """Simple average doesn't need fitting"""
        self.is_fitted = True
        return self
        
    def combine(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        CRITICAL LINE: Simple averaging of all predictions
        
        Takes arithmetic mean across all model predictions.
        This naturally reduces variance and can improve stability.
        """
        # Stack predictions into 2D array: (n_models, n_samples)
        stacked = np.array(predictions)
        
        # Take mean along model axis (axis=0)
        # Reference: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
        combined = np.mean(stacked, axis=0)
        
        return combined


#------------------------------------------------------------------------------
# Weighted Average Ensemble
#------------------------------------------------------------------------------

class WeightedAverageEnsemble(EnsembleStrategy):
    """
    Weighted Average Ensemble Strategy
    
    Combines predictions using learned or specified weights.
    Better performing models get higher weights.
    
    Formula: y_pred = w1*pred1 + w2*pred2 + ... + wn*predn
    Where: sum(weights) = 1
    
    Weight Selection Methods:
    - 'performance': Inverse of validation error (better model = higher weight)
    - 'custom': User-specified weights
    - 'optimal': Solve for weights that minimize training error
    
    Example:
        # Performance-based weights
        ensemble = WeightedAverageEnsemble(method='performance')
        ensemble.fit(train_predictions, y_train)
        combined = ensemble.combine(test_predictions)
        
        # Custom weights (e.g., trust LSTM 60%, ARIMA 40%)
        ensemble = WeightedAverageEnsemble(weights=[0.4, 0.6])
        combined = ensemble.combine([arima_pred, lstm_pred])
    """
    
    def __init__(self, weights: Optional[List[float]] = None, method: str = 'performance'):
        """
        Args:
            weights: Custom weights (must sum to 1), or None for automatic
            method: 'performance', 'optimal', or 'custom'
        """
        super().__init__(name="Weighted Average")
        self.weights = weights
        self.method = method
        
        if weights is not None:
            # Normalize weights to sum to 1
            self.weights = np.array(weights) / np.sum(weights)
            self.is_fitted = True
            
    def fit(self, predictions: List[np.ndarray], y_true: np.ndarray):
        """
        Learn optimal weights based on validation performance
        
        Args:
            predictions: List of prediction arrays from different models
            y_true: True target values for weight optimization
        """
        if self.weights is not None:
            # Using custom weights, no fitting needed
            self.is_fitted = True
            return self
        
        n_models = len(predictions)
        
        if self.method == 'performance':
            # CRITICAL LINE: Weight by inverse error (lower error = higher weight)
            # Reference: Ensemble learning theory
            errors = []
            for pred in predictions:
                # Calculate MAE for each model
                mae = mean_absolute_error(y_true, pred)
                errors.append(mae)
            
            # Inverse of error (add small epsilon to avoid division by zero)
            inverse_errors = 1.0 / (np.array(errors) + 1e-8)
            
            # Normalize to sum to 1
            self.weights = inverse_errors / np.sum(inverse_errors)
            
            print(f"Performance-based weights: {self.weights}")
            for i, (w, e) in enumerate(zip(self.weights, errors)):
                print(f"  Model {i+1}: weight={w:.3f}, MAE={e:.4f}")
                
        elif self.method == 'optimal':
            # CRITICAL LINE: Solve for optimal weights using linear regression
            # Reference: Stacked Generalization (Wolpert, 1992)
            # We treat this as: y_true = w1*pred1 + w2*pred2 + ... + wn*predn
            
            # Stack predictions as features: shape (n_samples, n_models)
            X = np.column_stack(predictions)
            
            # Fit linear regression with non-negative weights constraint
            # We use Ridge with positive constraint
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=0.01, fit_intercept=False, positive=True)
            reg.fit(X, y_true)
            
            # Get weights and normalize
            self.weights = reg.coef_
            self.weights = self.weights / np.sum(self.weights)
            
            print(f"Optimal weights: {self.weights}")
            
        else:
            # Default: equal weights
            self.weights = np.ones(n_models) / n_models
            
        self.is_fitted = True
        return self
        
    def combine(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        CRITICAL LINE: Weighted combination of predictions
        
        Multiply each prediction by its weight and sum.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first or provide weights.")
        
        # Stack and apply weights
        stacked = np.array(predictions)  # (n_models, n_samples)
        
        # Broadcast weights and compute weighted sum
        # weights shape: (n_models, 1), stacked shape: (n_models, n_samples)
        weighted = stacked * self.weights.reshape(-1, 1)
        combined = np.sum(weighted, axis=0)
        
        return combined


#------------------------------------------------------------------------------
# Stacking Ensemble (Meta-Learning)
#------------------------------------------------------------------------------

class StackingEnsemble(EnsembleStrategy):
    """
    Stacking Ensemble Strategy (Meta-Learning)
    
    Uses a meta-learner (second-level model) to learn how to best combine
    the predictions from base models.
    
    The meta-learner is trained on:
    - Input: Predictions from all base models
    - Output: True target values
    
    This allows the ensemble to learn complex combination patterns that
    simple averaging cannot capture.
    
    Common meta-learners:
    - Linear Regression: Simple and interpretable
    - Ridge Regression: Regularized linear model
    - Random Forest: Can capture non-linear patterns
    
    Reference:
    - Stacked Generalization: Wolpert (1992)
    - https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
    
    Example:
        # Using Random Forest as meta-learner
        ensemble = StackingEnsemble(meta_learner='rf')
        ensemble.fit(train_predictions, y_train)
        combined = ensemble.combine(test_predictions)
    """
    
    def __init__(self, meta_learner: str = 'linear'):
        """
        Args:
            meta_learner: Type of meta-learner
                - 'linear': Linear Regression
                - 'ridge': Ridge Regression (L2 regularization)
                - 'rf': Random Forest Regressor
        """
        super().__init__(name=f"Stacking ({meta_learner})")
        self.meta_learner_type = meta_learner
        self.meta_model = None
        
    def fit(self, predictions: List[np.ndarray], y_true: np.ndarray):
        """
        CRITICAL METHOD: Train the meta-learner
        
        The meta-learner learns to combine base model predictions optimally.
        
        Args:
            predictions: List of prediction arrays from base models (training set)
            y_true: True target values (training set)
        """
        # Stack predictions as features
        # Each column is predictions from one base model
        X_meta = np.column_stack(predictions)  # Shape: (n_samples, n_models)
        
        # Initialize meta-learner based on type
        if self.meta_learner_type == 'linear':
            self.meta_model = LinearRegression()
            
        elif self.meta_learner_type == 'ridge':
            # Ridge adds L2 regularization to prevent overfitting
            self.meta_model = Ridge(alpha=1.0)
            
        elif self.meta_learner_type == 'rf':
            # Random Forest can learn non-linear combinations
            # Reference: https://scikit-learn.org/stable/modules/ensemble.html#forest
            self.meta_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
        else:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner_type}")
        
        # CRITICAL LINE: Fit meta-model on base predictions
        # This learns how to best combine the base models
        print(f"Training {self.meta_learner_type} meta-learner...")
        self.meta_model.fit(X_meta, y_true)
        
        # Print learned combination (for linear models)
        if hasattr(self.meta_model, 'coef_'):
            print(f"Meta-learner coefficients: {self.meta_model.coef_}")
            if hasattr(self.meta_model, 'intercept_'):
                print(f"Meta-learner intercept: {self.meta_model.intercept_}")
        
        self.is_fitted = True
        return self
        
    def combine(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        CRITICAL LINE: Use meta-model to combine predictions
        
        Instead of simple averaging, the meta-model makes the final prediction.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Stack predictions as features
        X_meta = np.column_stack(predictions)
        
        # Use meta-model to make final predictions
        combined = self.meta_model.predict(X_meta)
        
        return combined


#------------------------------------------------------------------------------
# Ensemble Combiner (Main Interface)
#------------------------------------------------------------------------------

class EnsembleCombiner:
    """
    Main interface for combining multiple model predictions
    
    Supports multiple ensemble strategies and provides utilities for
    performance comparison and analysis.
    
    Example:
        combiner = EnsembleCombiner()
        
        # Add predictions from different models
        combiner.add_model("ARIMA", arima_predictions)
        combiner.add_model("LSTM", lstm_predictions)
        combiner.add_model("GRU", gru_predictions)
        
        # Try different ensemble methods
        simple_avg = combiner.combine(method='simple_average')
        weighted = combiner.combine(method='weighted', strategy='performance')
        stacked = combiner.combine(method='stacking', meta_learner='rf')
    """
    
    def __init__(self):
        self.model_predictions = {}
        self.model_names = []
        
    def add_model(self, name: str, predictions: np.ndarray):
        """Add predictions from a model"""
        self.model_predictions[name] = predictions
        self.model_names.append(name)
        
    def combine(self, 
                method: str = 'simple_average',
                y_true: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Combine all model predictions using specified method
        
        Args:
            method: 'simple_average', 'weighted', or 'stacking'
            y_true: True values (required for weighted and stacking)
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Combined predictions
        """
        predictions_list = [self.model_predictions[name] for name in self.model_names]
        
        if method == 'simple_average':
            ensemble = SimpleAverageEnsemble()
            ensemble.fit(predictions_list, y_true)
            
        elif method == 'weighted':
            ensemble = WeightedAverageEnsemble(**kwargs)
            if y_true is not None:
                ensemble.fit(predictions_list, y_true)
                
        elif method == 'stacking':
            if y_true is None:
                raise ValueError("y_true required for stacking ensemble")
            ensemble = StackingEnsemble(**kwargs)
            ensemble.fit(predictions_list, y_true)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return ensemble.combine(predictions_list)
    
    def evaluate_all(self, y_true: np.ndarray, metric: str = 'mae') -> pd.DataFrame:
        """
        Evaluate all individual models and ensemble combinations
        
        Returns DataFrame with performance comparison
        """
        results = []
        
        # Evaluate individual models
        for name in self.model_names:
            pred = self.model_predictions[name]
            mae = mean_absolute_error(y_true, pred)
            rmse = np.sqrt(mean_squared_error(y_true, pred))
            results.append({
                'Model': name,
                'Type': 'Individual',
                'MAE': mae,
                'RMSE': rmse
            })
        
        # Evaluate ensembles
        predictions_list = [self.model_predictions[name] for name in self.model_names]
        
        # Simple average
        simple = SimpleAverageEnsemble()
        simple_pred = simple.combine(predictions_list)
        mae = mean_absolute_error(y_true, simple_pred)
        rmse = np.sqrt(mean_squared_error(y_true, simple_pred))
        results.append({
            'Model': 'Simple Average Ensemble',
            'Type': 'Ensemble',
            'MAE': mae,
            'RMSE': rmse
        })
        
        # Weighted average
        weighted = WeightedAverageEnsemble(method='performance')
        weighted.fit(predictions_list, y_true)
        weighted_pred = weighted.combine(predictions_list)
        mae = mean_absolute_error(y_true, weighted_pred)
        rmse = np.sqrt(mean_squared_error(y_true, weighted_pred))
        results.append({
            'Model': 'Weighted Average Ensemble',
            'Type': 'Ensemble',
            'MAE': mae,
            'RMSE': rmse
        })
        
        # Stacking
        stacking = StackingEnsemble(meta_learner='ridge')
        stacking.fit(predictions_list, y_true)
        stacking_pred = stacking.combine(predictions_list)
        mae = mean_absolute_error(y_true, stacking_pred)
        rmse = np.sqrt(mean_squared_error(y_true, stacking_pred))
        results.append({
            'Model': 'Stacking Ensemble (Ridge)',
            'Type': 'Ensemble',
            'MAE': mae,
            'RMSE': rmse
        })
        
        return pd.DataFrame(results)


#------------------------------------------------------------------------------
# Example Usage and Testing
#------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*70)
    print("TASK C.6: Testing Ensemble Methods")
    print("="*70)
    
    # Generate synthetic test data
    np.random.seed(42)
    n = 100
    y_true = np.sin(np.linspace(0, 4*np.pi, n)) * 10 + 50
    
    # Simulate predictions from 3 different models with different error patterns
    arima_pred = y_true + np.random.normal(0, 3, n)  # Higher variance
    lstm_pred = y_true + np.random.normal(0, 2, n) + 1  # Slight bias
    gru_pred = y_true + np.random.normal(0, 2.5, n) - 0.5  # Small bias
    
    print(f"\n📊 Test data: {n} samples")
    print(f"Individual model errors:")
    print(f"  ARIMA MAE: {mean_absolute_error(y_true, arima_pred):.3f}")
    print(f"  LSTM MAE:  {mean_absolute_error(y_true, lstm_pred):.3f}")
    print(f"  GRU MAE:   {mean_absolute_error(y_true, gru_pred):.3f}")
    
    # Test Simple Average
    print("\n" + "="*70)
    print("Test 1: Simple Average Ensemble")
    print("="*70)
    simple = SimpleAverageEnsemble()
    simple_combined = simple.combine([arima_pred, lstm_pred, gru_pred])
    simple_mae = mean_absolute_error(y_true, simple_combined)
    print(f"[OK] Simple Average MAE: {simple_mae:.3f}")
    
    # Test Weighted Average
    print("\n" + "="*70)
    print("Test 2: Weighted Average Ensemble (performance-based)")
    print("="*70)
    weighted = WeightedAverageEnsemble(method='performance')
    weighted.fit([arima_pred, lstm_pred, gru_pred], y_true)
    weighted_combined = weighted.combine([arima_pred, lstm_pred, gru_pred])
    weighted_mae = mean_absolute_error(y_true, weighted_combined)
    print(f"[OK] Weighted Average MAE: {weighted_mae:.3f}")
    
    # Test Stacking
    print("\n" + "="*70)
    print("Test 3: Stacking Ensemble (Ridge meta-learner)")
    print("="*70)
    stacking = StackingEnsemble(meta_learner='ridge')
    stacking.fit([arima_pred, lstm_pred, gru_pred], y_true)
    stacking_combined = stacking.combine([arima_pred, lstm_pred, gru_pred])
    stacking_mae = mean_absolute_error(y_true, stacking_combined)
    print(f"[OK] Stacking MAE: {stacking_mae:.3f}")
    
    # Compare all methods
    print("\n" + "="*70)
    print("COMPARISON SUMMARY:")
    print("="*70)
    print(f"Individual Models:")
    print(f"  ARIMA: {mean_absolute_error(y_true, arima_pred):.3f}")
    print(f"  LSTM:  {mean_absolute_error(y_true, lstm_pred):.3f}")
    print(f"  GRU:   {mean_absolute_error(y_true, gru_pred):.3f}")
    print(f"\nEnsemble Methods:")
    print(f"  Simple Average: {simple_mae:.3f}")
    print(f"  Weighted Avg:   {weighted_mae:.3f}")
    print(f"  Stacking:       {stacking_mae:.3f}")
    print("="*70)
    
    improvement = (mean_absolute_error(y_true, arima_pred) - simple_mae) / mean_absolute_error(y_true, arima_pred) * 100
    print(f"\n[OK] Ensemble improvement over best individual: {improvement:.1f}%")
