"""
Task C.7: Sentiment Analyzer

This module implements sentiment analysis using multiple tools for comparison.
Addresses Task Requirement 2 (5 marks).

Requirements addressed:
- Use sentiment analysis tools to generate sentiment scores
- Aggregate sentiment at daily level to match stock data frequency
- Experiment with different tools and discuss suitability

Tools implemented:
1. FinBERT - Best for financial news (primary model)
2. VADER - Fast, good for social media (baseline)
3. TextBlob - Simple baseline for comparison

Author: Your Name
Date: October 2025
References:
- FinBERT: https://huggingface.co/ProsusAI/finbert
- VADER: https://github.com/cjhutto/vaderSentiment
- TextBlob: https://textblob.readthedocs.io/
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# FinBERT (Transformer-based, best for financial text)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    print("[WARNING] transformers not installed. FinBERT disabled.")
    print("          Install with: pip install transformers torch")

# VADER (Lexicon-based, good for social media)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("[WARNING] vaderSentiment not installed. VADER disabled.")
    print("          Install with: pip install vaderSentiment")

# TextBlob (Simple baseline)
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("[WARNING] textblob not installed. TextBlob disabled.")
    print("          Install with: pip install textblob")


class SentimentAnalyzer:
    """
    Multi-tool sentiment analyzer for financial text
    
    This class implements and compares three sentiment analysis approaches:
    1. FinBERT: State-of-the-art transformer model trained on financial text
    2. VADER: Fast lexicon-based model, good for social media
    3. TextBlob: Simple rule-based model as baseline
    
    Usage:
        analyzer = SentimentAnalyzer(primary_model='finbert')
        
        # Analyze single text
        score = analyzer.analyze(text, method='finbert')
        
        # Analyze DataFrame of articles
        df['sentiment'] = analyzer.analyze_batch(df['text'])
        
        # Compare all methods
        comparison = analyzer.compare_methods(df['text'])
        
        # Aggregate daily sentiment
        daily_sentiment = analyzer.aggregate_daily_sentiment(df)
    """
    
    def __init__(self, primary_model: str = 'finbert', device: str = 'cpu'):
        """
        Initialize sentiment analyzer
        
        Args:
            primary_model: Primary model to use ('finbert', 'vader', 'textblob')
            device: 'cpu' or 'cuda' for GPU acceleration (FinBERT only)
        """
        self.primary_model = primary_model
        self.device = device
        
        # Initialize models
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.vader_analyzer = None
        
        # Load primary model
        if primary_model == 'finbert' and FINBERT_AVAILABLE:
            self._load_finbert()
        elif primary_model == 'vader' and VADER_AVAILABLE:
            self._load_vader()
        elif primary_model == 'textblob' and TEXTBLOB_AVAILABLE:
            print("[OK] Using TextBlob for sentiment analysis")
        else:
            print(f"[WARNING] Primary model '{primary_model}' not available")
        
        # Statistics
        self.analysis_stats = {
            'total_analyzed': 0,
            'finbert_count': 0,
            'vader_count': 0,
            'textblob_count': 0,
            'errors': 0
        }
    
    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    
    def _load_finbert(self):
        """
        Load FinBERT model for financial sentiment analysis
        
        FinBERT is a BERT model fine-tuned on financial news for
        sentiment analysis. It's specifically trained to understand
        financial terminology and context.
        
        Model: ProsusAI/finbert
        Output: Probabilities for [positive, negative, neutral]
        """
        print("[LOADING] FinBERT model (this may take a few minutes)...")
        
        try:
            model_name = 'ProsusAI/finbert'
            
            # Load tokenizer (converts text to model input)
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model (neural network weights)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to device (CPU or GPU)
            self.finbert_model.to(self.device)
            
            # Set to evaluation mode (disables dropout, etc.)
            self.finbert_model.eval()
            
            print(f"[OK] FinBERT loaded successfully (device: {self.device})")
            
        except Exception as e:
            print(f"[ERROR] Failed to load FinBERT: {e}")
            self.finbert_model = None
    
    def _load_vader(self):
        """
        Load VADER sentiment analyzer
        
        VADER (Valence Aware Dictionary for Sentiment Reasoning) is a
        lexicon and rule-based sentiment analysis tool specifically
        attuned to social media text.
        
        Output: Compound score from -1 (negative) to +1 (positive)
        """
        print("[LOADING] VADER sentiment analyzer...")
        
        try:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            print("[OK] VADER loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load VADER: {e}")
            self.vader_analyzer = None
    
    # =========================================================================
    # SENTIMENT ANALYSIS METHODS
    # =========================================================================
    
    def analyze_finbert(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT
        
        This is the BEST method for financial news as FinBERT is
        specifically trained on financial text and understands
        domain-specific terminology.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: {
                'positive': float (0-1),
                'negative': float (0-1),
                'neutral': float (0-1),
                'compound': float (-1 to 1, positive - negative)
            }
            
        Example:
            >>> text = "Commonwealth Bank reports strong quarterly profits"
            >>> analyzer.analyze_finbert(text)
            {'positive': 0.92, 'negative': 0.02, 'neutral': 0.06, 'compound': 0.90}
        """
        if not self.finbert_model or not self.finbert_tokenizer:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        try:
            # Tokenize input text
            # max_length=512 is BERT's limit
            # truncation=True ensures text fits within limit
            inputs = self.finbert_tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions (no gradient calculation for inference)
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
            
            # Convert logits to probabilities using softmax
            # Softmax ensures probabilities sum to 1.0
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]  # Move to CPU and extract first result
            
            # FinBERT outputs: [positive, negative, neutral]
            # Map to named dictionary
            result = {
                'positive': float(probs[0]),
                'negative': float(probs[1]),
                'neutral': float(probs[2]),
                'compound': float(probs[0] - probs[1])  # Net sentiment
            }
            
            self.analysis_stats['finbert_count'] += 1
            return result
            
        except Exception as e:
            print(f"[ERROR] FinBERT analysis failed: {e}")
            self.analysis_stats['errors'] += 1
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        VADER is fast and works well for social media text.
        Good baseline for comparison with FinBERT.
        
        Args:
            text: Input text
            
        Returns:
            dict: {'positive', 'negative', 'neutral', 'compound'}
            
        Note:
            VADER compound score interpretation:
            - >= 0.05: positive
            - <= -0.05: negative
            - between -0.05 and 0.05: neutral
        """
        if not self.vader_analyzer:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        try:
            # VADER returns scores for pos, neu, neg, compound
            scores = self.vader_analyzer.polarity_scores(text)
            
            result = {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': scores['compound']
            }
            
            self.analysis_stats['vader_count'] += 1
            return result
            
        except Exception as e:
            print(f"[ERROR] VADER analysis failed: {e}")
            self.analysis_stats['errors'] += 1
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
    
    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        TextBlob is the simplest method, good as a baseline.
        Returns polarity from -1 (negative) to +1 (positive).
        
        Args:
            text: Input text
            
        Returns:
            dict: {'compound': polarity score}
        """
        if not TEXTBLOB_AVAILABLE:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # Convert to positive/negative/neutral format
            if polarity > 0.05:
                positive = polarity
                negative = 0.0
            elif polarity < -0.05:
                positive = 0.0
                negative = abs(polarity)
            else:
                positive = 0.0
                negative = 0.0
            
            neutral = 1.0 - (positive + negative)
            
            result = {
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'compound': polarity
            }
            
            self.analysis_stats['textblob_count'] += 1
            return result
            
        except Exception as e:
            print(f"[ERROR] TextBlob analysis failed: {e}")
            self.analysis_stats['errors'] += 1
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
    
    def analyze(self, text: str, method: str = None) -> float:
        """
        Analyze sentiment using specified method (or primary model)
        
        Args:
            text: Input text
            method: 'finbert', 'vader', 'textblob', or None (use primary)
            
        Returns:
            float: Compound sentiment score (-1 to 1)
        """
        if method is None:
            method = self.primary_model
        
        if method == 'finbert':
            result = self.analyze_finbert(text)
        elif method == 'vader':
            result = self.analyze_vader(text)
        elif method == 'textblob':
            result = self.analyze_textblob(text)
        else:
            print(f"[WARNING] Unknown method '{method}', using FinBERT")
            result = self.analyze_finbert(text)
        
        self.analysis_stats['total_analyzed'] += 1
        return result['compound']
    
    def analyze_batch(self, texts: pd.Series, method: str = None,
                      show_progress: bool = True) -> pd.Series:
        """
        Analyze batch of texts (more efficient than looping)
        
        Args:
            texts: Pandas Series of texts
            method: Sentiment method to use
            show_progress: Whether to print progress
            
        Returns:
            pd.Series: Sentiment scores
        """
        if method is None:
            method = self.primary_model
        
        print(f"\n[ANALYSIS] Analyzing {len(texts)} texts with {method.upper()}...")
        
        scores = []
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(texts)} ({100*(i+1)/len(texts):.1f}%)")
            
            score = self.analyze(text, method=method)
            scores.append(score)
        
        print(f"[OK] Analysis complete!")
        return pd.Series(scores, index=texts.index)
    
    # =========================================================================
    # COMPARISON & AGGREGATION
    # =========================================================================
    
    def compare_methods(self, texts: pd.Series) -> pd.DataFrame:
        """
        Compare all three sentiment methods on same texts
        
        This addresses Task Requirement 2: "experiment with different
        tools and discuss their suitability"
        
        Args:
            texts: Pandas Series of texts to analyze
            
        Returns:
            pd.DataFrame with columns: [finbert, vader, textblob]
        """
        print(f"\n[COMPARISON] Comparing all sentiment methods on {len(texts)} texts...")
        
        comparison_df = pd.DataFrame(index=texts.index)
        
        # Analyze with each method
        if FINBERT_AVAILABLE:
            comparison_df['finbert'] = self.analyze_batch(texts, method='finbert', show_progress=True)
        
        if VADER_AVAILABLE:
            comparison_df['vader'] = self.analyze_batch(texts, method='vader', show_progress=True)
        
        if TEXTBLOB_AVAILABLE:
            comparison_df['textblob'] = self.analyze_batch(texts, method='textblob', show_progress=True)
        
        # Calculate correlation between methods
        print("\n[CORRELATION] Between sentiment methods:")
        print(comparison_df.corr())
        
        return comparison_df
    
    def aggregate_daily_sentiment(self, news_df: pd.DataFrame,
                                   text_col: str = 'full_text',
                                   sentiment_col: str = 'sentiment_score') -> pd.DataFrame:
        """
        Aggregate sentiment scores by day
        
        This is CRITICAL for Task Requirement 2: "aggregate at daily level
        to match frequency of stock data"
        
        Aggregation features calculated:
        - avg_sentiment: Mean daily sentiment
        - sentiment_std: Volatility of sentiment
        - sentiment_max: Most positive article
        - sentiment_min: Most negative article
        - positive_ratio: % of positive articles
        - negative_ratio: % of negative articles
        - article_count: Number of articles
        
        Args:
            news_df: DataFrame with 'date' and sentiment columns
            text_col: Column containing text
            sentiment_col: Column containing sentiment scores
            
        Returns:
            pd.DataFrame: Daily aggregated sentiment features
        """
        print(f"\n[AGGREGATION] Aggregating sentiment by day...")
        
        # Ensure sentiment scores exist
        if sentiment_col not in news_df.columns:
            print(f"  Calculating sentiment scores for {len(news_df)} articles...")
            news_df[sentiment_col] = self.analyze_batch(news_df[text_col])
        
        # Classify as positive/negative/neutral
        news_df['is_positive'] = news_df[sentiment_col] > 0.05
        news_df['is_negative'] = news_df[sentiment_col] < -0.05
        news_df['is_neutral'] = ((news_df[sentiment_col] >= -0.05) & 
                                  (news_df[sentiment_col] <= 0.05))
        
        # Group by date and aggregate
        daily_agg = news_df.groupby('date').agg({
            sentiment_col: ['mean', 'std', 'min', 'max', 'count'],
            'is_positive': 'mean',  # Ratio of positive articles
            'is_negative': 'mean',  # Ratio of negative articles
            'is_neutral': 'mean',   # Ratio of neutral articles
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = [
            'date',
            'sentiment_score',      # Mean sentiment
            'sentiment_std',        # Sentiment volatility
            'sentiment_min',        # Most negative
            'sentiment_max',        # Most positive
            'article_count',        # Number of articles
            'positive_ratio',       # % positive
            'negative_ratio',       # % negative
            'neutral_ratio',        # % neutral
        ]
        
        # Fill missing std (happens when only 1 article per day)
        daily_agg['sentiment_std'] = daily_agg['sentiment_std'].fillna(0)
        
        print(f"[OK] Aggregated to {len(daily_agg)} trading days")
        print(f"\nDaily sentiment statistics:")
        print(daily_agg[['sentiment_score', 'sentiment_std', 'article_count']].describe())
        
        return daily_agg
    
    # =========================================================================
    # STATISTICS & UTILITIES
    # =========================================================================
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics"""
        return self.analysis_stats.copy()
    
    def save_comparison_plot(self, comparison_df: pd.DataFrame, save_path: str):
        """
        Create comparison plot for different sentiment methods
        
        Args:
            comparison_df: Output from compare_methods()
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Distribution comparison
        ax = axes[0, 0]
        for col in comparison_df.columns:
            comparison_df[col].hist(alpha=0.5, bins=30, label=col, ax=ax)
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Sentiment Score Distributions')
        ax.legend()
        
        # Plot 2: Scatter plot (FinBERT vs VADER)
        if 'finbert' in comparison_df.columns and 'vader' in comparison_df.columns:
            ax = axes[0, 1]
            ax.scatter(comparison_df['vader'], comparison_df['finbert'], alpha=0.3)
            ax.set_xlabel('VADER Score')
            ax.set_ylabel('FinBERT Score')
            ax.set_title('FinBERT vs VADER Comparison')
            ax.plot([-1, 1], [-1, 1], 'r--', alpha=0.5)  # Diagonal line
        
        # Plot 3: Correlation heatmap
        ax = axes[1, 0]
        sns.heatmap(comparison_df.corr(), annot=True, cmap='coolwarm', center=0,
                    ax=ax, vmin=-1, vmax=1)
        ax.set_title('Correlation Between Methods')
        
        # Plot 4: Method agreement
        ax = axes[1, 1]
        if 'finbert' in comparison_df.columns and 'vader' in comparison_df.columns:
            agreement = (np.sign(comparison_df['finbert']) == np.sign(comparison_df['vader'])).mean()
            disagreement = 1 - agreement
            ax.bar(['Agreement', 'Disagreement'], [agreement, disagreement])
            ax.set_ylabel('Proportion')
            ax.set_title(f'FinBERT-VADER Agreement: {agreement:.1%}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Comparison plot saved to {save_path}")
        plt.close()


# Example usage
if __name__ == '__main__':
    from config import Task7Config
    import os
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(primary_model='finbert')
    
    # Test on sample texts
    sample_texts = [
        "Commonwealth Bank reports record quarterly profits, beating analyst expectations",
        "CBA faces regulatory investigation over money laundering concerns",
        "Commonwealth Bank announces new digital banking platform",
        "Investors disappointed as CBA misses earnings targets",
    ]
    
    print("\n" + "="*70)
    print("TESTING SENTIMENT ANALYSIS")
    print("="*70)
    
    # Test each method
    for i, text in enumerate(sample_texts, 1):
        print(f"\n[{i}] Text: {text[:60]}...")
        
        if FINBERT_AVAILABLE:
            finbert = analyzer.analyze_finbert(text)
            print(f"    FinBERT:  {finbert['compound']:+.3f} (pos={finbert['positive']:.2f}, neg={finbert['negative']:.2f})")
        
        if VADER_AVAILABLE:
            vader = analyzer.analyze_vader(text)
            print(f"    VADER:    {vader['compound']:+.3f} (pos={vader['positive']:.2f}, neg={vader['negative']:.2f})")
        
        if TEXTBLOB_AVAILABLE:
            textblob = analyzer.analyze_textblob(text)
            print(f"    TextBlob: {textblob['compound']:+.3f}")
    
    # Show statistics
    print("\n" + "="*70)
    print("ANALYSIS STATISTICS")
    print("="*70)
    stats = analyzer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
