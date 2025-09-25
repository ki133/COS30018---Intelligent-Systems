# File: data_processing.py
# Authors: [Your Name]
# Date: 31/08/2025
# Task: COS30018 - Option C - Task C.2: Enhanced Data Processing

# Code developed based on requirements from Task C.2 and learning from:
# (P1) https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction

# This file contains the enhanced data processing function that addresses
# the limitations of v0.1 stock_prediction.py mentioned in the TODO comments

import numpy as np
import pandas as pd
import yfinance as yf
import os
import json
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#------------------------------------------------------------------------------
# Task C.2: Enhanced Data Processing Function
# This function fulfills requirements (a) through (e) as specified in Task C.2
#------------------------------------------------------------------------------

def load_and_process_data(ticker, start_date, end_date, 
                         features=['Close'], 
                         split_method='ratio', split_value=0.8,
                         cache_dir='data_cache', scale_features=True,
                         scale_mode='per_feature'):
    """
    Enhanced data loading and processing function for Task C.2, extended for C.5
    
    This function addresses the major limitations of v0.1 and is updated for C.5:
    - Supports multiple features for multivariate prediction (C.5)
    - Handles NaN values properly  
    - Provides flexible train/test splitting methods
    - Implements local caching to avoid repeated downloads
    - Applies proper scaling with multiple strategies (per_feature or all_features) (C.5)
    - Prevents data leakage by fitting scalers only on training data
    
    Parameters:
    -----------
    ticker (str): Stock ticker symbol (e.g., 'CBA.AX', 'AAPL')
    start_date (str): Start date in 'YYYY-MM-DD' format for the entire dataset *** REQUIREMENT (a) ***
    end_date (str): End date in 'YYYY-MM-DD' format for the entire dataset *** REQUIREMENT (a) ***
    features (list): List of feature column names to use. 
                    Available: ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    Default: ['Close'] to maintain compatibility with v0.1
    split_method (str): Method to split data into train/test sets *** REQUIREMENT (c) ***
                       - 'ratio': Split by ratio (train_size)
                       - 'date': Split by specific date (train_end)  
                       - 'random': Random split with fixed random_state
    split_value: Split parameter depending on method *** REQUIREMENT (c) ***:
                - For 'ratio': float between 0 and 1 (e.g., 0.8 = 80% train, 20% test)
                - For 'date': string date 'YYYY-MM-DD' (train_end)
                - For 'random': float between 0 and 1 (train_size for random split)
    cache_dir (str): Directory to store cached data files *** REQUIREMENT (d) ***
    scale_features (bool): Whether to apply MinMax scaling (0,1) to features *** REQUIREMENT (e) ***
    scale_mode (str): Scaling strategy for multivariate data (Task C.5 extension)
                     - 'per_feature': Each feature gets its own scaler (original behavior)
                     - 'all_features': A single scaler is fitted to all features together
    
    Returns:
    --------
    dict: Dictionary containing all processed data and metadata:
        - 'train_data': Training set dataframe (scaled if scale_features=True)
        - 'test_data': Test set dataframe (scaled if scale_features=True)
        - 'raw_train_data': Original unscaled training data  
        - 'raw_test_data': Original unscaled test data
        - 'scalers': Dictionary of fitted MinMaxScaler objects. 
                     If scale_mode is 'all_features', the key is 'all'.
        - 'metadata': Information about processing parameters
    """
    
    print(f"=== Task C.2: Loading and Processing Data for {ticker} ===")
    
    #--------------------------------------------------------------------------
    # REQUIREMENT (a): Allow user to specify start date and end date for whole dataset
    # Input validation for date parameters
    #--------------------------------------------------------------------------
    
    try:
        # Validate date format by attempting to parse them
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
        print(f"✓ Requirement (a): Date range specified - {start_date} to {end_date}")
    except:
        raise ValueError("start_date and end_date must be in 'YYYY-MM-DD' format")
    
    if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
        raise ValueError("start_date must be earlier than end_date")
    
    #--------------------------------------------------------------------------
    # REQUIREMENT (d): Store downloaded data locally and load from cache
    # Setup caching system to avoid repeated downloads
    #--------------------------------------------------------------------------
    
    print(f"✓ Requirement (d): Setting up local data caching")
    
    # Create cache directory if it doesn't exist
    # This allows us to store downloaded data locally for future use
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created cache directory: {cache_dir}")
    
    # Generate unique cache key based on ticker and date range
    # This ensures we can cache different datasets separately
    cache_key = f"{ticker}_{start_date}_{end_date}"
    cache_csv_path = os.path.join(cache_dir, f"{cache_key}.csv")
    cache_meta_path = os.path.join(cache_dir, f"{cache_key}_meta.json")
    
    # Check if cached data exists and load it
    if os.path.exists(cache_csv_path) and os.path.exists(cache_meta_path):
        print(f"✓ Loading from cache: {cache_csv_path}")
        # Load the CSV file with proper date parsing
        # index_col=0 means first column (Date) becomes the index
        # parse_dates=True converts the index to datetime objects
        data = pd.read_csv(cache_csv_path, index_col=0, parse_dates=True)
        
        # Load metadata to verify cache validity
        with open(cache_meta_path, 'r') as f:
            cache_metadata = json.load(f)
            print(f"Cache created: {cache_metadata['cache_date']}")
    else:
        print(f"✓ Downloading fresh data for {ticker} from {start_date} to {end_date}")
        
        # Download data using yfinance
        # yfinance is more reliable than pandas_datareader for Yahoo Finance data
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Handle potential multi-level column structure from yfinance
        # Sometimes yfinance returns MultiIndex columns, we want simple column names
        if isinstance(data.columns, pd.MultiIndex):
            # Get the first level of column names (the actual feature names)
            data.columns = data.columns.get_level_values(0)
        
        # Save to cache for future use
        data.to_csv(cache_csv_path)
        print(f"✓ Data cached to: {cache_csv_path}")
        
        # Save metadata about this cached dataset
        cache_metadata = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'cache_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_columns': list(data.columns),
            'data_shape': data.shape
        }
        
        with open(cache_meta_path, 'w') as f:
            json.dump(cache_metadata, f, indent=2)
        print(f"✓ Metadata cached to: {cache_meta_path}")
    
    print(f"Original data shape: {data.shape}")
    print(f"Available columns: {list(data.columns)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    #--------------------------------------------------------------------------
    # Handle feature column name mapping and validation
    # yfinance uses specific column names, we need to handle different naming conventions
    #--------------------------------------------------------------------------
    
    # This mapping ensures compatibility with different naming styles
    # As mentioned in the instructions, we handle both 'adjclose' and 'Adj Close' formats
    column_mapping = {
        'adjclose': 'Adj Close',
        'AdjClose': 'Adj Close', 
        'close': 'Close',
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'volume': 'Volume'
    }
    
    # Normalize feature names to match yfinance column names
    normalized_features = []
    for feature in features:
        if feature in column_mapping:
            normalized_feature = column_mapping[feature]
        else:
            normalized_feature = feature
            
        if normalized_feature in data.columns:
            normalized_features.append(normalized_feature)
        else:
            print(f"Warning: Feature '{feature}' (normalized to '{normalized_feature}') not found in data")
    
    if not normalized_features:
        raise ValueError("No valid features found in the dataset")
    
    print(f"Using features: {normalized_features}")
    
    # Select only the requested features from the dataset
    feature_data = data[normalized_features].copy()
    
    #--------------------------------------------------------------------------
    # REQUIREMENT (b): Deal with NaN issue in the data
    # Handle NaN (missing) values properly
    #--------------------------------------------------------------------------
    
    print(f"✓ Requirement (b): Handling NaN values in the data")
    
    # Check for NaN values in our selected features
    nan_counts = feature_data.isnull().sum()
    total_nans = nan_counts.sum()
    
    if total_nans > 0:
        print(f"Found {total_nans} NaN values:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  {col}: {count} NaN values")
        
        # Strategy 1: Forward fill (use previous day's value)
        # This is reasonable for stock prices as they tend to be continuous
        # method='ffill' means forward fill - propagate last valid observation forward
        feature_data_clean = feature_data.fillna(method='ffill')
        
        # Strategy 2: Backward fill for any remaining NaN at the beginning
        # This handles cases where the first few rows have NaN values
        # method='bfill' means backward fill - use next valid observation to fill gap
        feature_data_clean = feature_data_clean.fillna(method='bfill')
        
        # Strategy 3: Drop any remaining rows with NaN (as last resort)
        # If there are still NaN values after forward and backward fill, remove those rows
        rows_before_drop = len(feature_data_clean)
        feature_data_clean = feature_data_clean.dropna()
        rows_after_drop = len(feature_data_clean)
        
        if rows_before_drop != rows_after_drop:
            print(f"Dropped {rows_before_drop - rows_after_drop} rows with remaining NaN values")
        
        remaining_nans = feature_data_clean.isnull().sum().sum()
        print(f"✓ After cleaning: {remaining_nans} NaN values remaining")
        print(f"Data shape after NaN handling: {feature_data_clean.shape}")
        
    else:
        print("✓ No NaN values found in the selected features")
        feature_data_clean = feature_data.copy()
    
    #--------------------------------------------------------------------------
    # REQUIREMENT (c): Use different methods to split data into train/test
    # Split data into training and testing sets using flexible methods
    #--------------------------------------------------------------------------
    
    print(f"✓ Requirement (c): Splitting data using '{split_method}' method")
    
    if split_method == 'ratio':
        # Method 1: Split by ratio - first X% for training, remaining for testing
        # This maintains temporal order which is important for time series data
        # We don't shuffle the data because time order matters in stock prediction
        split_index = int(len(feature_data_clean) * split_value)
        
        raw_train_data = feature_data_clean.iloc[:split_index].copy()
        raw_test_data = feature_data_clean.iloc[split_index:].copy()
        
        print(f"Ratio split ({split_value:.1%}):")
        print(f"  Training: {len(raw_train_data)} samples ({raw_train_data.index[0]} to {raw_train_data.index[-1]})")
        print(f"  Testing: {len(raw_test_data)} samples ({raw_test_data.index[0]} to {raw_test_data.index[-1]})")
        
    elif split_method == 'date':
        # Method 2: Split by specific date - all data before split_value for training
        # This is useful when you want to test on a specific time period
        # For example, train on 2020-2023 data, test on 2023-2024 data
        split_date = pd.to_datetime(split_value)
        
        raw_train_data = feature_data_clean[feature_data_clean.index < split_date].copy()
        raw_test_data = feature_data_clean[feature_data_clean.index >= split_date].copy()
        
        print(f"Date split at {split_value}:")
        print(f"  Training: {len(raw_train_data)} samples ({raw_train_data.index[0]} to {raw_train_data.index[-1]})")
        print(f"  Testing: {len(raw_test_data)} samples ({raw_test_data.index[0]} to {raw_test_data.index[-1]})")
        
    elif split_method == 'random':
        # Method 3: Random split - randomly assign data points to train/test
        # Note: This breaks temporal order, which may not be ideal for time series
        # But it's useful for testing if temporal patterns are important
        # We use random_state=42 for reproducibility
        
        # Create indices for splitting
        indices = feature_data_clean.index
        
        # Use train_test_split from sklearn with fixed random_state for reproducibility
        # random_state=42 ensures we get the same split every time we run the code
        # This is important for comparing different experiments
        train_indices, test_indices = train_test_split(
            indices, 
            train_size=split_value, 
            random_state=42,
            shuffle=True
        )
        
        # Sort indices to maintain some temporal structure within each set
        # Even though we split randomly, we sort within each set to keep some order
        train_indices = sorted(train_indices)
        test_indices = sorted(test_indices)
        
        raw_train_data = feature_data_clean.loc[train_indices].copy()
        raw_test_data = feature_data_clean.loc[test_indices].copy()
        
        print(f"Random split ({split_value:.1%} train, random_state=42):")
        print(f"  Training: {len(raw_train_data)} samples")
        print(f"  Testing: {len(raw_test_data)} samples")
        
    else:
        raise ValueError("split_method must be 'ratio', 'date', or 'random'")
    
    # Validate that we have data in both sets
    if len(raw_train_data) == 0:
        raise ValueError("Training set is empty after splitting")
    if len(raw_test_data) == 0:
        raise ValueError("Test set is empty after splitting")
    
    #--------------------------------------------------------------------------
    # REQUIREMENT (e): Scale feature columns and store scalers
    # Apply feature scaling with proper handling to prevent data leakage
    # This addresses ISSUE #2 mentioned in v0.1 comments
    #--------------------------------------------------------------------------
    
    scalers = {}
    
    if scale_features:
        print(f"✓ Requirement (e): Applying MinMax scaling and storing scalers")
        print(f"Scaling mode: {scale_mode}")
        print("IMPORTANT: Fitting scalers on training data only to prevent data leakage")
        
        # Initialize scaled datasets as copies of raw data
        train_data = raw_train_data.copy()
        test_data = raw_test_data.copy()
        
        if scale_mode == 'all_features':
            # Task C.5: Scale all features together using one scaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Fit on the entire training dataframe
            scaler.fit(raw_train_data[normalized_features])
            
            # Transform both train and test data
            train_data[normalized_features] = scaler.transform(raw_train_data[normalized_features])
            test_data[normalized_features] = scaler.transform(raw_test_data[normalized_features])
            
            # Store the single scaler
            scalers['all'] = scaler
            print("✓ Applied a single scaler to all features.")

        elif scale_mode == 'per_feature':
            # Original Task C.2 behavior: Scale each feature separately
            for feature in normalized_features:
                print(f"Processing feature: {feature}")
                
                scaler = MinMaxScaler(feature_range=(0, 1))
                
                train_values_2d = raw_train_data[feature].values.reshape(-1, 1)
                scaler.fit(train_values_2d)
                
                train_data[feature] = scaler.transform(train_values_2d).flatten()
                test_values_2d = raw_test_data[feature].values.reshape(-1, 1)
                test_data[feature] = scaler.transform(test_values_2d).flatten()
                
                scalers[feature] = scaler
                
                # Print scaling statistics for verification and debugging
                train_min, train_max = train_values_2d.min(), train_values_2d.max()
                test_min, test_max = test_values_2d.min(), test_values_2d.max()
                scaled_test_min, scaled_test_max = test_data[feature].min(), test_data[feature].max()
                
                print(f"  {feature}:")
                print(f"    Train range: [{train_min:.2f}, {train_max:.2f}] -> [0.00, 1.00]")
                print(f"    Test range: [{test_min:.2f}, {test_max:.2f}] -> [{scaled_test_min:.2f}, {scaled_test_max:.2f}]")
                
                # Warning if test data goes outside [0,1] range
                # This indicates that the test period has different price ranges than training period
                # This is the "ISSUE #2" mentioned in v0.1 comments
                if scaled_test_min < -0.1 or scaled_test_max > 1.1:
                    print(f"    ⚠️  WARNING: Test data for {feature} extends outside [0,1] range!")
                    print(f"       This suggests test period has different {feature} range than training period")
                    print(f"       This is the 'ISSUE #2' mentioned in v0.1 comments")
                    print(f"       The model may struggle with values outside its training range")
        
        else:
            raise ValueError("scale_mode must be 'per_feature' or 'all_features'")

        # Save scalers to cache
        scalers_cache_path = os.path.join(cache_dir, f"{cache_key}_scalers.pkl")
        with open(scalers_cache_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"✓ Scalers saved to: {scalers_cache_path}")
        
    else:
        print("✓ Scaling disabled - using raw feature values")
        train_data = raw_train_data.copy()
        test_data = raw_test_data.copy()
        scalers = None
    
    #--------------------------------------------------------------------------
    # Prepare comprehensive return data structure
    #--------------------------------------------------------------------------
    
    # Create comprehensive metadata about the processing
    # This metadata is useful for debugging and understanding what was done
    metadata = {
        'ticker': ticker,
        'start_date': start_date, 
        'end_date': end_date,
        'features_requested': features,
        'features_used': normalized_features,
        'split_method': split_method,
        'split_value': split_value,
        'scaling_applied': scale_features,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'train_date_range': f"{train_data.index[0]} to {train_data.index[-1]}",
        'test_date_range': f"{test_data.index[0]} to {test_data.index[-1]}",
        'cache_dir': cache_dir,
        'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task_requirements_fulfilled': {
            'a_specify_dates': True,
            'b_handle_nan': True, 
            'c_flexible_splitting': True,
            'd_local_caching': True,
            'e_scaling_and_scalers': scale_features
        }
    }
    
    # Prepare the comprehensive return dictionary
    # This structure gives users access to both processed and raw data
    result = {
        'train_data': train_data,           # Scaled training data (if scaling enabled)
        'test_data': test_data,             # Scaled test data (if scaling enabled)  
        'raw_train_data': raw_train_data,   # Original unscaled training data
        'raw_test_data': raw_test_data,     # Original unscaled test data
        'scalers': scalers,                 # Dictionary of fitted scalers per feature
        'config': metadata                # Processing information and parameters
    }
    
    print("=== ✅ All Task C.2 Requirements (a-e) Completed Successfully ===")
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Features processed: {len(normalized_features)}")
    print(f"Scaling applied: {scale_features}")
    print(f"Cache directory: {cache_dir}")
    
    return result

#------------------------------------------------------------------------------
# Example usage demonstrating all requirements (a-e):
#
# # Requirement (a): Specify start and end dates for whole dataset
# # Requirement (b): Handle NaN values automatically
# # Requirement (c): Use ratio split method
# # Requirement (d): Enable local caching
# # Requirement (e): Apply scaling and store scalers
# basic_data = load_and_process_data(
#     ticker='CBA.AX',
#     start_date='2020-01-01',  # (a) Start date
#     end_date='2024-07-02',    # (a) End date
#     features=['Close'],       # Single feature like v0.1
#     split_method='ratio',     # (c) Split by ratio
#     split_value=0.8,          # (c) 80% train, 20% test
#     cache_dir='data_cache',   # (d) Local caching directory
#     scale_features=True       # (e) Apply scaling
# )
#
# # Advanced usage with multiple features and date split
# advanced_data = load_and_process_data(
#     ticker='CBA.AX',
#     start_date='2020-01-01',                                    # (a)
#     end_date='2024-07-02',                                      # (a)
#     features=['Open', 'High', 'Low', 'Close', 'Volume'],       # Multiple features
#     split_method='date',                                        # (c) Date split
#     split_value='2023-08-01',                                   # (c) Split date
#     cache_dir='my_cache',                                       # (d)
#     scale_features=True                                         # (e)
# )
#
# # Random split example
# random_data = load_and_process_data(
#     ticker='AAPL',
#     start_date='2020-01-01',                                    # (a)
#     end_date='2024-07-02',                                      # (a)  
#     features=['Close', 'Volume'],                               # Two features
#     split_method='random',                                      # (c) Random split
#     split_value=0.75,                                           # (c) 75% train
#     cache_dir='data_cache',                                     # (d)
#     scale_features=True                                         # (e)
# )
#
# # Access the processed data
# train_data = basic_data['train_data']      # Scaled training data
# test_data = basic_data['test_data']        # Scaled test data  
# scalers = basic_data['scalers']            # For inverse transformation
# metadata = basic_data['metadata']          # Processing information
#------------------------------------------------------------------------------
