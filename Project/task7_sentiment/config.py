"""
Task C.7: Configuration Settings

Centralizes all configuration parameters for Task 7.
This file contains settings for data collection, sentiment analysis,
feature engineering, and model training.

Following best practices: All hardcoded values are defined here,
making it easy to modify without touching code logic.

Author: Your Name
Date: October 2025
"""

import os


class Task7Config:
    """
    Configuration class for Task C.7 Sentiment-Based Prediction
    
    All parameters needed for the experiment are defined here.
    Modify these values to change experiment settings.
    """
    
    # =========================================================================
    # STOCK DATA CONFIGURATION (Reuses Tasks C.2-C.6)
    # =========================================================================
    
    TICKER = 'CBA.AX'  # Commonwealth Bank of Australia
    START_DATE = '2023-11-01'  # Start of data collection (2 years ago)
    END_DATE = '2025-11-01'    # End of data collection (today)
    
    # Technical features to extract from stock data
    # These will be calculated using data_processing.py (Task C.2)
    TECHNICAL_FEATURES = [
        'Close',           # Closing price
        'Volume',          # Trading volume
        'High',            # Daily high
        'Low',             # Daily low
        'returns_1d',      # 1-day returns
        'returns_5d',      # 5-day returns
        'returns_10d',     # 10-day returns
        'volatility_5d',   # 5-day rolling volatility
        'volatility_20d',  # 20-day rolling volatility
        'ma_5',            # 5-day moving average
        'ma_20',           # 20-day moving average
        'ma_50',           # 50-day moving average
        'rsi',             # Relative Strength Index (14-day)
        'macd',            # Moving Average Convergence Divergence
        'macd_signal',     # MACD signal line
    ]
    
    # =========================================================================
    # NEWS DATA COLLECTION CONFIGURATION (Task Requirement 1: 5 marks)
    # =========================================================================
    
    # API Keys (USER MUST SET THESE!)
    # Get NewsAPI key from: https://newsapi.org/register
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', 'b35d9f08ee214ea597d693b536d43a9f')
    
    # Get Twitter API keys from: https://developer.twitter.com/
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', 'YOUR_TWITTER_API_KEY_HERE')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', 'YOUR_TWITTER_API_SECRET_HERE')
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', 'YOUR_BEARER_TOKEN_HERE')
    
    # News search keywords for CBA.AX
    NEWS_KEYWORDS = [
        'Commonwealth Bank',
        'CBA Australia',
        'CommBank',
        'Australian banking',
        'ASX CBA'
    ]
    
    # News sources to prioritize (financial news)
    PREFERRED_SOURCES = [
        'financial-times',
        'bloomberg',
        'reuters',
        'the-wall-street-journal',
        'australian-financial-review'
    ]
    
    # Data collection limits
    MAX_ARTICLES_PER_DAY = 50      # Limit to avoid noise
    MIN_ARTICLE_LENGTH = 100       # Filter out very short articles
    
    # =========================================================================
    # SENTIMENT ANALYSIS CONFIGURATION (Task Requirement 2: 5 marks)
    # =========================================================================
    
    # Sentiment models to compare
    SENTIMENT_MODELS = {
        'finbert': 'ProsusAI/finbert',           # Best for financial text
        'finbert_tone': 'yiyanghkust/finbert-tone',  # Alternative FinBERT
        'vader': 'vader',                         # Baseline for social media
        'textblob': 'textblob',                   # Simple baseline
    }
    
    # Primary sentiment model (best for financial news)
    PRIMARY_SENTIMENT_MODEL = 'finbert'
    
    # Sentiment aggregation settings
    SENTIMENT_FEATURES = [
        'sentiment_score',      # Daily average sentiment (-1 to 1)
        'sentiment_std',        # Sentiment volatility
        'sentiment_max',        # Most positive sentiment
        'sentiment_min',        # Most negative sentiment
        'positive_ratio',       # % of positive articles
        'negative_ratio',       # % of negative articles
        'neutral_ratio',        # % of neutral articles
        'article_count',        # Number of articles
        'weighted_sentiment',   # Weighted by article importance
    ]
    
    # Sentiment thresholds
    POSITIVE_THRESHOLD = 0.1   # Sentiment > 0.1 considered positive
    NEGATIVE_THRESHOLD = -0.1  # Sentiment < -0.1 considered negative
    
    # =========================================================================
    # FEATURE ENGINEERING CONFIGURATION (Task Requirement 3: 5 marks)
    # =========================================================================
    
    # Feature interaction terms (combining technical + sentiment)
    INTERACTION_FEATURES = [
        'sentiment_x_volume',      # Sentiment * Volume
        'sentiment_x_volatility',  # Sentiment * Volatility
        'sentiment_x_returns',     # Sentiment * Returns
        'sentiment_change',        # Change in sentiment from prev day
        'sentiment_ma_3',          # 3-day sentiment moving average
        'sentiment_ma_7',          # 7-day sentiment moving average
    ]
    
    # All features combined
    ALL_FEATURES = TECHNICAL_FEATURES + SENTIMENT_FEATURES + INTERACTION_FEATURES
    
    # Target variable definition
    TARGET_DEFINITION = 'next_day_direction'  # UP (1) or DOWN (0)
    
    # Train/test split
    TRAIN_RATIO = 0.8  # 80% train, 20% test (temporal split, no shuffle!)
    
    # =========================================================================
    # MODEL CONFIGURATION (Task Requirement 3 & 4: 10 marks)
    # =========================================================================
    
    # Models to train and compare
    MODELS_TO_TEST = {
        'logistic': {
            'name': 'Logistic Regression',
            'type': 'sklearn',
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }
        },
        'random_forest': {
            'name': 'Random Forest',
            'type': 'sklearn',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'random_state': 42
            }
        },
        'xgboost': {
            'name': 'XGBoost',
            'type': 'xgboost',
            'params': {
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        },
        'lstm': {
            'name': 'LSTM Classifier',
            'type': 'keras',  # Reuses Task C.4 model_builder
            'params': {
                'sequence_length': 60,
                'layer_units': [64, 32],
                'dropout': 0.2,
                'bidirectional': True,
                'epochs': 50,
                'batch_size': 32
            }
        }
    }
    
    # Class imbalance handling
    USE_SMOTE = True  # Oversample minority class
    SMOTE_RATIO = 'auto'  # Balance classes to 1:1
    
    # Feature scaling
    SCALER_TYPE = 'standard'  # 'standard' or 'minmax'
    
    # =========================================================================
    # EVALUATION CONFIGURATION (Task Requirement 4: 5 marks)
    # =========================================================================
    
    # Metrics to calculate
    EVALUATION_METRICS = [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'roc_auc',
        'confusion_matrix',
        'classification_report'
    ]
    
    # Baseline configurations for comparison
    BASELINES = {
        'technical_only': {
            'features': TECHNICAL_FEATURES,
            'description': 'Baseline 1: Technical indicators only (no sentiment)'
        },
        'sentiment_only': {
            'features': SENTIMENT_FEATURES,
            'description': 'Baseline 2: Sentiment features only (no technical)'
        }
    }
    
    # Statistical significance test
    USE_MCNEMAR_TEST = True  # Test if improvement is statistically significant
    SIGNIFICANCE_LEVEL = 0.05  # p-value threshold
    
    # =========================================================================
    # INDEPENDENT RESEARCH CONFIGURATION (Task Requirement 5: 5 marks)
    # =========================================================================
    
    # FinBERT Fine-tuning settings
    FINETUNE_FINBERT = True
    FINETUNE_EPOCHS = 3
    FINETUNE_BATCH_SIZE = 16
    FINETUNE_LEARNING_RATE = 2e-5
    
    # Financial lexicon enhancement
    FINANCIAL_LEXICON = {
        # Very positive terms
        'profit upgrade': 0.9,
        'dividend increase': 0.8,
        'strong earnings': 0.8,
        'beat expectations': 0.7,
        'record profit': 0.9,
        
        # Positive terms
        'growth': 0.5,
        'expansion': 0.5,
        'acquisition': 0.4,
        'partnership': 0.4,
        
        # Neutral terms
        'announcement': 0.0,
        'statement': 0.0,
        
        # Negative terms
        'regulatory investigation': -0.7,
        'lawsuit': -0.6,
        'scandal': -0.8,
        'losses': -0.6,
        
        # Very negative terms
        'profit warning': -0.9,
        'missed expectations': -0.7,
        'downgrade': -0.8,
        'CEO resignation': -0.6,
        'fraud': -0.9,
    }
    
    # Aspect-based sentiment weights
    ASPECT_WEIGHTS = {
        'financial_performance': 0.6,  # Most important for stock price
        'management_changes': 0.2,
        'regulatory_issues': 0.15,
        'general_news': 0.05
    }
    
    # =========================================================================
    # OUTPUT CONFIGURATION (Task Requirement 6: 5 marks)
    # =========================================================================
    
    # Directory paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'task7_data')
    MODELS_DIR = os.path.join(BASE_DIR, 'task7_models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'task7_results')
    
    # Data subdirectories
    NEWS_RAW_DIR = os.path.join(DATA_DIR, 'news_raw')
    NEWS_PROCESSED_DIR = os.path.join(DATA_DIR, 'news_processed')
    SENTIMENT_DIR = os.path.join(DATA_DIR, 'sentiment_scores')
    
    # Output file formats
    SAVE_FORMAT = 'both'  # 'json', 'csv', or 'both'
    
    # Visualization settings
    PLOT_STYLE = 'seaborn'
    FIGURE_DPI = 300  # High quality for report
    FIGURE_FORMAT = 'png'
    
    # Logging
    LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    VERBOSE = True
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    @classmethod
    def create_directories(cls):
        """
        Create all required directories if they don't exist.
        Called at the start of experiments.
        """
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.NEWS_RAW_DIR,
            cls.NEWS_PROCESSED_DIR,
            cls.SENTIMENT_DIR,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_feature_groups(cls):
        """
        Returns a dictionary of feature groups for easy access.
        
        Returns:
            dict: Feature groups (technical, sentiment, interaction, all)
        """
        return {
            'technical': cls.TECHNICAL_FEATURES,
            'sentiment': cls.SENTIMENT_FEATURES,
            'interaction': cls.INTERACTION_FEATURES,
            'all': cls.ALL_FEATURES
        }
    
    @classmethod
    def validate_config(cls):
        """
        Validates configuration settings and checks for required API keys.
        
        Returns:
            bool: True if configuration is valid
        
        Raises:
            ValueError: If critical configuration is missing
        """
        # Check API keys
        if cls.NEWSAPI_KEY == 'YOUR_NEWSAPI_KEY_HERE':
            print("[WARNING] NewsAPI key not set. Set environment variable NEWSAPI_KEY")
            print("          Get key from: https://newsapi.org/register")
        
        if cls.TWITTER_BEARER_TOKEN == 'YOUR_BEARER_TOKEN_HERE':
            print("[WARNING] Twitter API token not set. Twitter data collection will be skipped.")
            print("          Get token from: https://developer.twitter.com/")
        
        # Check date range
        from datetime import datetime
        start = datetime.strptime(cls.START_DATE, '%Y-%m-%d')
        end = datetime.strptime(cls.END_DATE, '%Y-%m-%d')
        
        if start >= end:
            raise ValueError(f"START_DATE ({cls.START_DATE}) must be before END_DATE ({cls.END_DATE})")
        
        days_diff = (end - start).days
        if days_diff < 180:  # Less than 6 months
            print(f"[WARNING] Short date range ({days_diff} days). Recommended: At least 180 days")
        
        print("[OK] Configuration validated successfully")
        return True
    
    @classmethod
    def print_config_summary(cls):
        """
        Prints a summary of the current configuration.
        Useful for documenting experiments.
        """
        print("=" * 70)
        print("TASK C.7 CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"Stock Ticker:        {cls.TICKER}")
        print(f"Date Range:          {cls.START_DATE} to {cls.END_DATE}")
        print(f"Technical Features:  {len(cls.TECHNICAL_FEATURES)}")
        print(f"Sentiment Features:  {len(cls.SENTIMENT_FEATURES)}")
        print(f"Total Features:      {len(cls.ALL_FEATURES)}")
        print(f"Models to Test:      {len(cls.MODELS_TO_TEST)}")
        print(f"Primary Model:       {cls.PRIMARY_SENTIMENT_MODEL}")
        print(f"Train/Test Split:    {cls.TRAIN_RATIO:.0%} / {1-cls.TRAIN_RATIO:.0%}")
        print(f"Use SMOTE:           {cls.USE_SMOTE}")
        print(f"Results Directory:   {cls.RESULTS_DIR}")
        print("=" * 70)


# Example usage and validation
if __name__ == '__main__':
    # Validate configuration
    Task7Config.validate_config()
    
    # Print summary
    Task7Config.print_config_summary()
    
    # Create directories
    Task7Config.create_directories()
    print("\n[OK] All directories created")
    
    # Show feature groups
    feature_groups = Task7Config.get_feature_groups()
    print(f"\n[INFO] Feature groups available:")
    for group_name, features in feature_groups.items():
        print(f"  - {group_name}: {len(features)} features")
