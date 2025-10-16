"""
ensemble_models.py
Task C.6: Ensemble Methods - ARIMA/SARIMA Statistical Models

This module provides wrapper classes for statistical time series models (ARIMA, SARIMA)
that integrate seamlessly with our existing deep learning pipeline.

Key Features:
- Auto ARIMA parameter selection using pmdarima
- SARIMA for seasonal patterns
- Compatible with data from data_processing.py
- Unified interface matching our DL models

Author: Anh Vu Le
Date: 12/10/2025
References:
- ARIMA Theory: https://otexts.com/fpp2/arima.html
- Statsmodels ARIMA: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
- Auto ARIMA: http://alkaline-ml.com/pmdarima/
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, Tuple, Optional

# Statistical time series models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not installed. Install with: pip install statsmodels")

# pmdarima is optional (has compatibility issues with newer numpy)
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except (ImportError, ValueError):
    PMDARIMA_AVAILABLE = False
    auto_arima = None
    warnings.warn("pmdarima not available or incompatible. Auto ARIMA selection disabled.")

#------------------------------------------------------------------------------
# ARIMA Model Wrapper
#------------------------------------------------------------------------------

class ARIMAModel:
    """
    ARIMA (AutoRegressive Integrated Moving Average) Model Wrapper
    
    ARIMA models are classical statistical methods for time series forecasting.
    They model a time series based on its own past values and past forecast errors.
    
    ARIMA(p, d, q) where:
    - p: number of lag observations (AR order)
    - d: degree of differencing (I order)
    - q: size of moving average window (MA order)
    
    Example:
        model = ARIMAModel(order=(5,1,0))  # AR(5) with 1st order differencing
        model.fit(train_data)
        predictions = model.predict(n_steps=10)
    """
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0), auto_select: bool = False):
        """
        Initialize ARIMA model
        Args:
            order: (p, d, q) - ARIMA parameters
            auto_select: If True, use auto_arima to find best parameters
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Please install: pip install statsmodels")
        
        self.order = order
        self.auto_select = auto_select
        self.model = None
        self.fitted_model = None
        self.train_data = None
        self.best_order = None
        
    def fit(self, train_data: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        """
        Fit ARIMA model to training data
        
        Args:
            train_data: 1D array of time series values
            verbose: Print fitting progress
            
        Returns:
            Dictionary with fitting results and diagnostics
        """
        if verbose:
            print(f"=== Fitting ARIMA Model ===")
        
        # Ensure 1D array
        if len(train_data.shape) > 1:
            train_data = train_data.flatten()
        
        self.train_data = train_data
        
        # AUTO ARIMA: Automatically find best (p,d,q) parameters
        # This is a KEY feature - saves manual parameter tuning!
        if self.auto_select:
            if not PMDARIMA_AVAILABLE:
                warnings.warn("pmdarima not available. Falling back to manual ARIMA order.")
                self.auto_select = False
            else:
                if verbose:
                    print("🔍 Running Auto ARIMA to find optimal parameters...")
                    print("   This may take a few minutes...")
                
                # auto_arima uses AIC/BIC to select best parameters
                # Reference: http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
                self.fitted_model = auto_arima(
                    train_data,
                    start_p=1, start_q=1,      # Starting search values
                    max_p=5, max_q=5,          # Maximum values to try
                    d=None,                     # Let it determine differencing order
                    seasonal=False,             # Set True for SARIMA
                    stepwise=True,              # Faster stepwise search
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=verbose               # Print search progress
                )
                
                self.best_order = self.fitted_model.order
                self.order = self.best_order
                
                if verbose:
                    print(f"[OK] Best ARIMA order found: {self.best_order}")
                    print(f"   AIC: {self.fitted_model.aic():.2f}")
                    print(f"   BIC: {self.fitted_model.bic():.2f}")
        
        if not self.auto_select:
            # Manual ARIMA with specified order
            if verbose:
                print(f"[ARIMA] Fitting ARIMA{self.order}...")
            
            # CRITICAL LINE: Create and fit ARIMA model
            # Reference: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
            self.model = ARIMA(train_data, order=self.order)
            self.fitted_model = self.model.fit()
            
            if verbose:
                print(f"[OK] ARIMA{self.order} fitted successfully")
                print(f"   AIC: {self.fitted_model.aic:.2f}")
                print(f"   BIC: {self.fitted_model.bic:.2f}")
        
        # Prepare return diagnostics
        results = {
            'order': self.order,
            'aic': float(self.fitted_model.aic),
            'bic': float(self.fitted_model.bic),
            'converged': self.fitted_model.mle_retvals is not None,
            'n_obs': len(train_data)
        }
        
        return results
    
    def predict(self, n_steps: int = 1, return_conf_int: bool = False) -> np.ndarray:
        """
        Make predictions for n future steps
        
        Args:
            n_steps: Number of steps to forecast
            return_conf_int: Return confidence intervals
            
        Returns:
            Array of predictions (and optionally confidence intervals)
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # CRITICAL LINE: Forecast future values
        # Reference: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.forecast.html
        forecast_result = self.fitted_model.forecast(steps=n_steps)
        
        if return_conf_int:
            # Get prediction intervals
            # This requires using get_forecast instead
            forecast_obj = self.fitted_model.get_forecast(steps=n_steps)
            predictions = forecast_obj.predicted_mean.values
            conf_int = forecast_obj.conf_int().values
            return predictions, conf_int
        else:
            # Return just predictions as numpy array
            if isinstance(forecast_result, pd.Series):
                return forecast_result.values
            return np.array(forecast_result)
    
    def get_in_sample_predictions(self) -> np.ndarray:
        """
        Get fitted values (in-sample predictions) for training data
        Useful for residual analysis and model diagnostics
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.fitted_model.fittedvalues
    
    def summary(self) -> str:
        """Get model summary statistics"""
        if self.fitted_model is None:
            return "Model not fitted yet"
        
        return str(self.fitted_model.summary())


#------------------------------------------------------------------------------
# SARIMA Model Wrapper (Seasonal ARIMA)
#------------------------------------------------------------------------------

class SARIMAModel:
    """
    SARIMA (Seasonal ARIMA) Model Wrapper
    
    SARIMA extends ARIMA to handle seasonal patterns in time series.
    Useful for stock data with weekly/monthly patterns.
    
    SARIMA(p,d,q)(P,D,Q,s) where:
    - (p,d,q): Non-seasonal parameters (same as ARIMA)
    - (P,D,Q,s): Seasonal parameters
    - s: Seasonal period (e.g., 12 for monthly data, 7 for weekly)
    
    Example:
        # Weekly seasonality (5 trading days)
        model = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,5))
        model.fit(train_data)
        predictions = model.predict(n_steps=5)
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
                 auto_select: bool = False):
        """
        Initialize SARIMA model
        
        Args:
            order: (p, d, q) - Non-seasonal ARIMA parameters
            seasonal_order: (P, D, Q, s) - Seasonal parameters
            auto_select: Use auto_arima with seasonal=True
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Please install: pip install statsmodels pmdarima")
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_select = auto_select
        self.model = None
        self.fitted_model = None
        self.train_data = None
        
    def fit(self, train_data: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        """
        Fit SARIMA model to training data
        
        Args:
            train_data: 1D array of time series values
            verbose: Print fitting progress
            
        Returns:
            Dictionary with fitting results
        """
        if verbose:
            print(f"=== Fitting SARIMA Model ===")
        
        # Ensure 1D array
        if len(train_data.shape) > 1:
            train_data = train_data.flatten()
        
        self.train_data = train_data
        
        if self.auto_select:
            if verbose:
                print("🔍 Running Auto ARIMA with seasonal components...")
                print("   This may take several minutes...")
            
            # CRITICAL LINE: Auto ARIMA with seasonality
            # Reference: http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
            self.fitted_model = auto_arima(
                train_data,
                start_p=1, start_q=1,
                max_p=3, max_q=3,          # Lower limits for seasonal (slower)
                d=None,
                seasonal=True,              # Enable seasonal modeling
                m=self.seasonal_order[3],   # Seasonal period
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                D=None,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=verbose
            )
            
            self.order = self.fitted_model.order
            self.seasonal_order = self.fitted_model.seasonal_order
            
            if verbose:
                print(f"[OK] Best SARIMA order: {self.order} x {self.seasonal_order}")
                print(f"   AIC: {self.fitted_model.aic():.2f}")
        else:
            if verbose:
                print(f"[SARIMA] Fitting SARIMA{self.order}x{self.seasonal_order}...")
            
            # CRITICAL LINE: Create and fit SARIMA model
            # Reference: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
            self.model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,  # Avoid convergence issues
                enforce_invertibility=False
            )
            self.fitted_model = self.model.fit(disp=False)
            
            if verbose:
                print(f"[OK] SARIMA fitted successfully")
                print(f"   AIC: {self.fitted_model.aic:.2f}")
        
        results = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': float(self.fitted_model.aic),
            'bic': float(self.fitted_model.bic),
            'n_obs': len(train_data)
        }
        
        return results
    
    def predict(self, n_steps: int = 1, return_conf_int: bool = False) -> np.ndarray:
        """Make predictions for n future steps"""
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # CRITICAL LINE: Forecast with seasonal model
        forecast_result = self.fitted_model.forecast(steps=n_steps)
        
        if return_conf_int:
            forecast_obj = self.fitted_model.get_forecast(steps=n_steps)
            predictions = forecast_obj.predicted_mean.values
            conf_int = forecast_obj.conf_int().values
            return predictions, conf_int
        else:
            if isinstance(forecast_result, pd.Series):
                return forecast_result.values
            return np.array(forecast_result)
    
    def get_in_sample_predictions(self) -> np.ndarray:
        """Get fitted values for training data"""
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.fitted_model.fittedvalues
    
    def summary(self) -> str:
        """Get model summary statistics"""
        if self.fitted_model is None:
            return "Model not fitted yet"
        
        return str(self.fitted_model.summary())


#------------------------------------------------------------------------------
# Unified Time Series Model Interface
#------------------------------------------------------------------------------

class TimeSeriesModel:
    """
    Unified wrapper for different time series models
    Provides consistent interface for both statistical and DL models
    """
    
    def __init__(self, model_type: str = 'arima', **kwargs):
        """
        Args:
            model_type: 'arima' or 'sarima'
            **kwargs: Parameters passed to specific model
        """
        self.model_type = model_type.lower()
        
        if self.model_type == 'arima':
            self.model = ARIMAModel(**kwargs)
        elif self.model_type == 'sarima':
            self.model = SARIMAModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, train_data: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        """Fit the model"""
        return self.model.fit(train_data, verbose=verbose)
    
    def predict(self, n_steps: int = 1, **kwargs) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(n_steps=n_steps, **kwargs)
    
    def get_in_sample_predictions(self) -> np.ndarray:
        """Get in-sample predictions"""
        return self.model.get_in_sample_predictions()


#------------------------------------------------------------------------------
# Example Usage and Testing
#------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*70)
    print("TASK C.6: Testing ARIMA/SARIMA Models")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    n = 200
    trend = np.linspace(100, 150, n)
    seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, n))
    noise = np.random.normal(0, 5, n)
    sample_data = trend + seasonal + noise
    
    # Split into train/test
    train_size = int(0.8 * len(sample_data))
    train_data = sample_data[:train_size]
    test_data = sample_data[train_size:]
    
    print(f"\n[DATA] Data: {len(train_data)} train, {len(test_data)} test samples")
    
    # Test 1: ARIMA with manual parameters
    print("\n" + "="*70)
    print("Test 1: ARIMA with manual order (5,1,0)")
    print("="*70)
    arima_manual = ARIMAModel(order=(5, 1, 0), auto_select=False)
    arima_manual.fit(train_data)
    pred_manual = arima_manual.predict(n_steps=len(test_data))
    mae_manual = np.mean(np.abs(pred_manual - test_data))
    print(f"[OK] Manual ARIMA MAE: {mae_manual:.2f}")
    
    # Test 2: Auto ARIMA
    print("\n" + "="*70)
    print("Test 2: Auto ARIMA (automatic parameter selection)")
    print("="*70)
    arima_auto = ARIMAModel(auto_select=True)
    arima_auto.fit(train_data)
    pred_auto = arima_auto.predict(n_steps=len(test_data))
    mae_auto = np.mean(np.abs(pred_auto - test_data))
    print(f"[OK] Auto ARIMA MAE: {mae_auto:.2f}")
    
    # Test 3: SARIMA
    print("\n" + "="*70)
    print("Test 3: SARIMA with seasonal period = 20")
    print("="*70)
    sarima = SARIMAModel(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 20),
        auto_select=False
    )
    sarima.fit(train_data)
    pred_sarima = sarima.predict(n_steps=len(test_data))
    mae_sarima = np.mean(np.abs(pred_sarima - test_data))
    print(f"[OK] SARIMA MAE: {mae_sarima:.2f}")
    
    print("\n" + "="*70)
    print("COMPARISON:")
    print(f"  Manual ARIMA: MAE = {mae_manual:.2f}")
    print(f"  Auto ARIMA:   MAE = {mae_auto:.2f}")
    print(f"  SARIMA:       MAE = {mae_sarima:.2f}")
    print("="*70)
    print("\n[OK] All tests completed successfully!")
