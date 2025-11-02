"""
Task C.7: Sentiment-Based Stock Price Movement Prediction

This package implements a classification model that predicts whether
stock prices will rise or fall based on:
1. Historical stock data (technical features)
2. News sentiment analysis (sentiment features)

Modules:
- config: Configuration settings
- news_collector: Data collection from NewsAPI and Twitter (Task requirement 1)
- sentiment_analyzer: Sentiment scoring with FinBERT/VADER/TextBlob (Task requirement 2)
- feature_builder: Feature engineering combining technical + sentiment (Task requirement 3)
- classifier_models: Classification models for UP/DOWN prediction (Task requirement 3)
- evaluator: Model evaluation and comparison (Task requirement 4)
- finbert_tuner: FinBERT fine-tuning for Australian banking context (Task requirement 5)

Author: Your Name
Date: October 2025
Course: COS30018 Intelligent Systems
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

# Package-level imports for convenience
from .config import Task7Config
from .news_collector import NewsCollector
from .sentiment_analyzer import SentimentAnalyzer
from .feature_builder import SentimentFeatureBuilder
from .classifier_models import SentimentClassifierTrainer
from .evaluator import ModelEvaluator

__all__ = [
    'Task7Config',
    'NewsCollector',
    'SentimentAnalyzer',
    'SentimentFeatureBuilder',
    'SentimentClassifierTrainer',
    'ModelEvaluator',
]
