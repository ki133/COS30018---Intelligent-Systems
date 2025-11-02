"""
Task C.7: Feature Builder

This module combines technical stock features with sentiment features.
Addresses Task Requirement 3 (5 marks) - Feature Engineering & Modelling.

Requirements addressed:
- Create input features combining historical stock data and sentiment scores
- REUSE data_processing.py from Task C.2 for stock data loading
- Add technical indicators (RSI, MACD, moving averages, volatility)
- Merge sentiment features with stock data (time-aligned)
- Create classification target: UP (1) or DOWN (0)

Author: Your Name
Date: October 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime

# Add parent directory to path to import from previous tasks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# REUSE Task C.2 (data_processing.py) for stock data loading
import data_processing

# Technical indicator libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARNING] TA-Lib not installed. Using manual calculations.")
    print("          For better indicators, install: pip install TA-Lib")


class SentimentFeatureBuilder:
    """
    Feature builder combining technical stock features + sentiment features
    
    This class implements feature engineering for Task C.7 by:
    1. Loading stock data using Task C.2's data_processing module (CODE REUSE)
    2. Calculating technical indicators (RSI, MACD, volatility, etc.)
    3. Merging sentiment features from news analysis
    4. Creating interaction features (sentiment × volume, etc.)
    5. Generating classification target (UP/DOWN)
    
    Usage:
        builder = SentimentFeatureBuilder(
            ticker='CBA.AX',
            start_date='2023-01-01',
            end_date='2024-10-01'
        )
        
        # Add sentiment features
        full_features = builder.merge_sentiment(daily_sentiment_df)
        
        # Create target
        full_features = builder.create_target(full_features)
        
        # Get feature sets for experiments
        X_tech, X_sent, X_full, y = builder.get_feature_sets(full_features)
    """
    
    def __init__(self, ticker: str, start_date: str, end_date: str, 
                 features: List[str] = None):
        """
        Initialize Feature Builder
        
        This uses Task C.2's data_processing.load_and_process_data()
        to load stock data - demonstrating CODE REUSE requirement!
        
        Args:
            ticker: Stock ticker symbol (e.g., 'CBA.AX')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            features: Stock features to load (default: ['Close', 'Volume', 'High', 'Low'])
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        if features is None:
            features = ['Close', 'Volume', 'High', 'Low', 'Open']
        
        self.base_features = features
        
        print(f"\n[FEATURE BUILDER] Initializing for {ticker}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Base features: {features}")
        
        # CRITICAL: REUSE Task C.2's data_processing module
        # This demonstrates integration with previous work!
        print("\n[REUSE] Loading stock data using Task C.2 data_processing module...")
        
        try:
            self.data_dict = data_processing.load_and_process_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                features=features,
                split_method='none',  # We'll split later after merging sentiment
                scale_features=False,  # Scale after creating all features
                cache_dir='data_cache'
            )
            
            # Extract full dataset (train + test combined since split_method='none')
            self.stock_df = pd.concat([
                self.data_dict['train_data'],
                self.data_dict['test_data']
            ]).sort_index().reset_index(drop=True)
            
            print(f"[OK] Loaded {len(self.stock_df)} trading days of stock data")
            
        except Exception as e:
            print(f"[ERROR] Failed to load stock data: {e}")
            print("[INFO] Falling back to manual yfinance download...")
            self._load_stock_data_fallback()
        
        # Store for later use
        self.technical_features = []
        self.sentiment_features = []
        self.interaction_features = []
    
    def _load_stock_data_fallback(self):
        """Fallback: Load stock data manually if data_processing fails"""
        import yfinance as yf
        
        stock = yf.Ticker(self.ticker)
        self.stock_df = stock.history(start=self.start_date, end=self.end_date)
        self.stock_df = self.stock_df.reset_index()
        self.stock_df = self.stock_df.rename(columns={'Date': 'date'})
        self.stock_df['date'] = pd.to_datetime(self.stock_df['date']).dt.date
        
        print(f"[OK] Loaded {len(self.stock_df)} days via yfinance")
    
    # =========================================================================
    # TECHNICAL INDICATORS (Stock Market Features)
    # =========================================================================
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to stock data
        
        These are standard stock market indicators used by traders:
        - Price-based: Returns, Moving Averages
        - Volatility: Rolling standard deviation
        - Momentum: RSI (Relative Strength Index)
        - Trend: MACD (Moving Average Convergence Divergence)
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            pd.DataFrame: DataFrame with added technical features
        """
        print("\n[TECHNICAL] Adding technical indicators...")
        
        df = df.copy()
        
        # Ensure we have required columns
        required = ['Close', 'High', 'Low', 'Volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"[WARNING] Missing columns: {missing}. Some indicators skipped.")
            return df
        
        # 1. RETURNS (Price changes)
        print("  [1] Calculating returns (1-day, 5-day, 10-day)...")
        df['returns_1d'] = df['Close'].pct_change(1)
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_10d'] = df['Close'].pct_change(10)
        
        # 2. VOLATILITY (Risk measure)
        print("  [2] Calculating volatility (5-day, 20-day)...")
        df['volatility_5d'] = df['Close'].rolling(window=5).std()
        df['volatility_20d'] = df['Close'].rolling(window=20).std()
        
        # 3. MOVING AVERAGES (Trend indicators)
        print("  [3] Calculating moving averages (5, 20, 50 days)...")
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_50'] = df['Close'].rolling(window=50).mean()
        
        # MA crossovers (trading signals)
        df['ma_5_20_cross'] = df['ma_5'] - df['ma_20']  # Golden/Death cross
        df['ma_20_50_cross'] = df['ma_20'] - df['ma_50']
        
        # 4. RSI (Relative Strength Index) - Momentum indicator
        # RSI > 70: Overbought (might fall)
        # RSI < 30: Oversold (might rise)
        print("  [4] Calculating RSI (14-day)...")
        if TALIB_AVAILABLE:
            df['rsi'] = talib.RSI(df['Close'].values, timeperiod=14)
        else:
            df['rsi'] = self._calculate_rsi_manual(df['Close'], period=14)
        
        # 5. MACD (Moving Average Convergence Divergence) - Trend strength
        print("  [5] Calculating MACD...")
        if TALIB_AVAILABLE:
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['Close'].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
        else:
            macd_data = self._calculate_macd_manual(df['Close'])
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['signal']
            df['macd_hist'] = macd_data['histogram']
        
        # 6. BOLLINGER BANDS (Volatility bands)
        print("  [6] Calculating Bollinger Bands...")
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (rolling_std * 2)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # 7. VOLUME INDICATORS
        print("  [7] Calculating volume indicators...")
        df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_20']  # Volume spike detector
        
        # 8. HIGH-LOW RANGE
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_to_high'] = df['Close'] / df['High']
        df['close_to_low'] = df['Close'] / df['Low']
        
        # Track technical feature names
        self.technical_features = [
            'Close', 'Volume', 'High', 'Low',
            'returns_1d', 'returns_5d', 'returns_10d',
            'volatility_5d', 'volatility_20d',
            'ma_5', 'ma_20', 'ma_50',
            'ma_5_20_cross', 'ma_20_50_cross',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
            'volume_ma_20', 'volume_ratio',
            'high_low_ratio', 'close_to_high', 'close_to_low'
        ]
        
        print(f"[OK] Added {len(self.technical_features)} technical features")
        
        return df
    
    def _calculate_rsi_manual(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI manually (when TA-Lib not available)
        
        RSI Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Args:
            prices: Price series
            period: RSI period (default 14)
            
        Returns:
            pd.Series: RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd_manual(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate MACD manually
        
        MACD = EMA(12) - EMA(26)
        Signal = EMA(MACD, 9)
        Histogram = MACD - Signal
        
        Args:
            prices: Price series
            
        Returns:
            dict: {'macd', 'signal', 'histogram'}
        """
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }
    
    # =========================================================================
    # SENTIMENT FEATURE INTEGRATION
    # =========================================================================
    
    def merge_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge sentiment features with stock data
        
        This is CRITICAL for Task C.7: combining technical + sentiment!
        
        Steps:
        1. Add technical indicators to stock data
        2. Merge with sentiment data on 'date'
        3. Handle missing sentiment (days without news → neutral)
        4. Create interaction features (sentiment × technical)
        
        Args:
            sentiment_df: Daily sentiment DataFrame with columns:
                          [date, sentiment_score, sentiment_std, positive_ratio,
                           article_count, etc.]
        
        Returns:
            pd.DataFrame: Combined features (technical + sentiment + interactions)
        """
        print("\n[MERGE] Merging stock data with sentiment features...")
        
        # Step 1: Add technical indicators
        stock_with_tech = self.add_technical_indicators(self.stock_df.copy())
        
        # Ensure date column is datetime.date for matching
        stock_with_tech['date'] = pd.to_datetime(stock_with_tech['date']).dt.date
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        
        # Step 2: Merge on date (left join to keep all trading days)
        print(f"  Stock data: {len(stock_with_tech)} days")
        print(f"  Sentiment data: {len(sentiment_df)} days")
        
        merged = pd.merge(
            stock_with_tech,
            sentiment_df,
            on='date',
            how='left'  # Keep all trading days, even if no news
        )
        
        print(f"  Merged: {len(merged)} days")
        
        # Step 3: Handle missing sentiment (IMPORTANT!)
        # Days without news get neutral sentiment
        sentiment_cols = [col for col in sentiment_df.columns if col != 'date']
        
        print(f"\n  Handling missing sentiment for {merged[sentiment_cols].isna().any(axis=1).sum()} days...")
        
        # Fill sentiment_score with 0 (neutral)
        if 'sentiment_score' in merged.columns:
            merged['sentiment_score'] = merged['sentiment_score'].fillna(0)
        
        # Fill ratios with 0.33 (equal distribution)
        for col in ['positive_ratio', 'negative_ratio', 'neutral_ratio']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.33)
        
        # Fill counts with 0
        if 'article_count' in merged.columns:
            merged['article_count'] = merged['article_count'].fillna(0)
        
        # Fill volatility metrics with 0
        if 'sentiment_std' in merged.columns:
            merged['sentiment_std'] = merged['sentiment_std'].fillna(0)
        
        # Track sentiment feature names
        self.sentiment_features = sentiment_cols
        
        # Step 4: Create interaction features
        merged = self._create_interaction_features(merged)
        
        print(f"[OK] Merged dataset ready: {len(merged)} days, {len(merged.columns)} features")
        
        return merged
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features (sentiment × technical)
        
        Interaction features capture relationships between sentiment and
        stock movements. For example:
        - High sentiment + High volume = Strong signal
        - Negative sentiment + High volatility = Panic selling?
        
        Args:
            df: DataFrame with technical + sentiment features
            
        Returns:
            pd.DataFrame: DataFrame with added interaction features
        """
        print("\n[INTERACTION] Creating sentiment × technical interaction features...")
        
        interactions = []
        
        # Check if sentiment_score exists
        if 'sentiment_score' not in df.columns:
            print("[WARNING] No sentiment_score column. Skipping interactions.")
            return df
        
        # 1. Sentiment × Volume (news volume interaction)
        if 'Volume' in df.columns:
            df['sentiment_x_volume'] = df['sentiment_score'] * df['Volume']
            interactions.append('sentiment_x_volume')
        
        # 2. Sentiment × Volatility (risk interaction)
        if 'volatility_5d' in df.columns:
            df['sentiment_x_volatility'] = df['sentiment_score'] * df['volatility_5d']
            interactions.append('sentiment_x_volatility')
        
        # 3. Sentiment × Returns (momentum interaction)
        if 'returns_1d' in df.columns:
            df['sentiment_x_returns'] = df['sentiment_score'] * df['returns_1d']
            interactions.append('sentiment_x_returns')
        
        # 4. Sentiment change (momentum in sentiment)
        df['sentiment_change'] = df['sentiment_score'].diff(1)
        interactions.append('sentiment_change')
        
        # 5. Sentiment moving averages (smoothed sentiment)
        df['sentiment_ma_3'] = df['sentiment_score'].rolling(window=3).mean()
        df['sentiment_ma_7'] = df['sentiment_score'].rolling(window=7).mean()
        interactions.extend(['sentiment_ma_3', 'sentiment_ma_7'])
        
        # 6. Article count × Returns (news attention)
        if 'article_count' in df.columns and 'returns_1d' in df.columns:
            df['news_attention'] = df['article_count'] * abs(df['returns_1d'])
            interactions.append('news_attention')
        
        self.interaction_features = interactions
        
        print(f"[OK] Created {len(interactions)} interaction features")
        
        return df
    
    # =========================================================================
    # TARGET VARIABLE CREATION
    # =========================================================================
    
    def create_target(self, df: pd.DataFrame, lookahead: int = 1) -> pd.DataFrame:
        """
        Create classification target: UP (1) or DOWN (0)
        
        This is the key difference from Tasks C.4-C.6 (regression):
        - Previous tasks: Predict exact price (regression)
        - Task C.7: Predict direction UP/DOWN (classification)
        
        Args:
            df: DataFrame with 'Close' column
            lookahead: Days ahead to predict (default: 1 = next day)
            
        Returns:
            pd.DataFrame: DataFrame with 'target' column added
        """
        print(f"\n[TARGET] Creating classification target (lookahead={lookahead} days)...")
        
        df = df.copy()
        
        # Calculate future close price
        df['Close_future'] = df['Close'].shift(-lookahead)
        
        # Create binary target:
        # 1 if future close > current close (UP)
        # 0 if future close <= current close (DOWN)
        df['target'] = (df['Close_future'] > df['Close']).astype(int)
        
        # Remove last rows (no future data)
        df = df[:-lookahead]
        
        # Check class balance
        if 'target' in df.columns:
            class_counts = df['target'].value_counts()
            print(f"\n  Class distribution:")
            print(f"    DOWN (0): {class_counts.get(0, 0)} samples ({100*class_counts.get(0,0)/len(df):.1f}%)")
            print(f"    UP (1):   {class_counts.get(1, 0)} samples ({100*class_counts.get(1,0)/len(df):.1f}%)")
            
            # Warn if severely imbalanced
            imbalance_ratio = max(class_counts) / min(class_counts)
            if imbalance_ratio > 1.5:
                print(f"  [WARNING] Classes are imbalanced (ratio: {imbalance_ratio:.2f})")
                print(f"            Consider using SMOTE or class weights")
        
        # Drop helper column
        df = df.drop(columns=['Close_future'])
        
        print(f"[OK] Target created. Dataset: {len(df)} samples")
        
        return df
    
    # =========================================================================
    # FEATURE SET PREPARATION
    # =========================================================================
    
    def get_feature_sets(self, df: pd.DataFrame, 
                         test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                            pd.DataFrame, pd.Series,
                                                            pd.Series, pd.Series,
                                                            pd.Series, pd.Series]:
        """
        Split data into feature sets for experiments
        
        Returns 3 feature sets for Task C.7 requirements:
        1. Technical features only (Baseline 1)
        2. Sentiment features only (Baseline 2)
        3. All features combined (Full model)
        
        Args:
            df: Full DataFrame with all features and target
            test_size: Proportion for test set (default: 0.2 = 20%)
            
        Returns:
            Tuple: (X_tech_train, X_tech_test,
                    X_sent_train, X_sent_test,
                    X_full_train, X_full_test,
                    y_train, y_test)
        """
        print(f"\n[SPLIT] Preparing feature sets (test_size={test_size})...")
        
        # Remove rows with NaN (from rolling calculations)
        df_clean = df.dropna().reset_index(drop=True)
        
        print(f"  After removing NaN: {len(df_clean)} samples")
        
        # Temporal split (CRITICAL: No shuffle for time series!)
        split_idx = int(len(df_clean) * (1 - test_size))
        
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        print(f"  Train: {len(train_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        
        # Extract target
        y_train = train_df['target']
        y_test = test_df['target']
        
        # Extract feature sets
        # 1. Technical features only
        X_tech_train = train_df[self.technical_features]
        X_tech_test = test_df[self.technical_features]
        
        # 2. Sentiment features only
        X_sent_train = train_df[self.sentiment_features]
        X_sent_test = test_df[self.sentiment_features]
        
        # 3. All features (technical + sentiment + interactions)
        all_feature_cols = (self.technical_features + 
                           self.sentiment_features + 
                           self.interaction_features)
        
        X_full_train = train_df[all_feature_cols]
        X_full_test = test_df[all_feature_cols]
        
        print(f"\n  Feature set sizes:")
        print(f"    Technical: {len(self.technical_features)} features")
        print(f"    Sentiment: {len(self.sentiment_features)} features")
        print(f"    Interaction: {len(self.interaction_features)} features")
        print(f"    Total: {len(all_feature_cols)} features")
        
        return (X_tech_train, X_tech_test,
                X_sent_train, X_sent_test,
                X_full_train, X_full_test,
                y_train, y_test)
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get feature names by category
        
        Returns:
            dict: {'technical': [...], 'sentiment': [...], 'interaction': [...]}
        """
        return {
            'technical': self.technical_features,
            'sentiment': self.sentiment_features,
            'interaction': self.interaction_features,
            'all': self.technical_features + self.sentiment_features + self.interaction_features
        }


# Example usage
if __name__ == '__main__':
    # Test feature builder
    print("="*70)
    print("TESTING FEATURE BUILDER")
    print("="*70)
    
    # Initialize
    builder = SentimentFeatureBuilder(
        ticker='CBA.AX',
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    # Create dummy sentiment data for testing
    print("\n[TEST] Creating dummy sentiment data...")
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    dummy_sentiment = pd.DataFrame({
        'date': dates,
        'sentiment_score': np.random.randn(len(dates)) * 0.3,
        'sentiment_std': np.abs(np.random.randn(len(dates)) * 0.1),
        'positive_ratio': np.random.uniform(0.2, 0.8, len(dates)),
        'negative_ratio': np.random.uniform(0.1, 0.4, len(dates)),
        'article_count': np.random.randint(0, 20, len(dates)),
    })
    
    # Merge features
    full_features = builder.merge_sentiment(dummy_sentiment)
    
    # Create target
    full_features = builder.create_target(full_features)
    
    # Get feature sets
    (X_tech_train, X_tech_test,
     X_sent_train, X_sent_test,
     X_full_train, X_full_test,
     y_train, y_test) = builder.get_feature_sets(full_features)
    
    print("\n[TEST] Feature sets created successfully!")
    print(f"  X_tech_train shape: {X_tech_train.shape}")
    print(f"  X_sent_train shape: {X_sent_train.shape}")
    print(f"  X_full_train shape: {X_full_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    
    # Show feature names
    feature_names = builder.get_feature_names()
    print("\n[TEST] Feature categories:")
    for category, names in feature_names.items():
        print(f"  {category}: {len(names)} features")
        if len(names) <= 10:
            print(f"    {names}")
