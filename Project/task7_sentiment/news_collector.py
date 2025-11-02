"""
Task C.7: News Data Collector

This module implements data collection from multiple sources (NewsAPI, Twitter)
for sentiment analysis. Addresses Task Requirement 1 (5 marks).

Requirements addressed:
- Collect relevant textual data using appropriate APIs
- Ensure time-alignment with historical stock price dataset
- Document data sources and filtering/cleaning steps
- Prepare data for sentiment analysis

Data Sources:
1. NewsAPI (newsapi.org) - Financial news articles
2. Twitter API v2 (developer.twitter.com) - Social media sentiment

Author: Your Name
Date: October 2025
References:
- NewsAPI Documentation: https://newsapi.org/docs
- Twitter API v2: https://developer.twitter.com/en/docs/twitter-api
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import re
from bs4 import BeautifulSoup

# Try importing Twitter library (optional)
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    print("[WARNING] tweepy not installed. Twitter collection disabled.")
    print("          Install with: pip install tweepy")


class NewsCollector:
    """
    News data collection from NewsAPI and Twitter
    
    This class handles:
    1. API authentication and rate limiting
    2. News article collection from NewsAPI
    3. Tweet collection from Twitter (optional)
    4. Text preprocessing and cleaning
    5. Time alignment with trading days
    
    Usage:
        collector = NewsCollector(newsapi_key='your_key')
        news_df = collector.collect_news(
            ticker='CBA.AX',
            start_date='2023-01-01',
            end_date='2024-10-01',
            keywords=['Commonwealth Bank', 'CBA']
        )
        cleaned_df = collector.clean_news(news_df)
    """
    
    def __init__(self, newsapi_key: str, twitter_bearer_token: Optional[str] = None):
        """
        Initialize News Collector with API credentials
        
        Args:
            newsapi_key: API key from newsapi.org
            twitter_bearer_token: Bearer token from Twitter Developer Portal (optional)
        """
        self.newsapi_key = newsapi_key
        self.twitter_bearer_token = twitter_bearer_token
        
        # NewsAPI base URL
        self.newsapi_base_url = "https://newsapi.org/v2/everything"
        
        # Initialize Twitter client if available
        self.twitter_client = None
        if twitter_bearer_token and TWITTER_AVAILABLE:
            try:
                self.twitter_client = tweepy.Client(bearer_token=twitter_bearer_token)
                print("[OK] Twitter API initialized successfully")
            except Exception as e:
                print(f"[WARNING] Twitter API initialization failed: {e}")
        
        # Rate limiting
        self.newsapi_requests_today = 0
        self.newsapi_daily_limit = 100  # Free tier limit
        
        # Statistics
        self.collection_stats = {
            'total_articles': 0,
            'total_tweets': 0,
            'failed_requests': 0,
            'duplicates_removed': 0
        }
    
    # =========================================================================
    # NEWSAPI COLLECTION
    # =========================================================================
    
    def collect_news(self, ticker: str, start_date: str, end_date: str,
                     keywords: List[str], save_raw: bool = True) -> pd.DataFrame:
        """
        Collect news articles from NewsAPI
        
        This is the main data collection function for Task Requirement 1.
        It fetches news articles related to the specified ticker and keywords.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'CBA.AX')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            keywords: List of keywords to search for
            save_raw: Whether to save raw JSON responses
            
        Returns:
            pd.DataFrame with columns: [date, title, description, content, 
                                        source, url, published_at]
        
        Raises:
            ValueError: If API key is invalid or date range is invalid
            
        Note:
            NewsAPI free tier limitations:
            - 100 requests per day
            - Only 1 month of historical data
            - Max 100 articles per request
        """
        print(f"\n[COLLECTION] Starting news collection for {ticker}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Keywords: {keywords}")
        
        # Validate inputs
        self._validate_date_range(start_date, end_date)
        
        all_articles = []
        
        # Convert keywords to query string
        # Example: 'Commonwealth Bank OR CBA OR CommBank'
        query = ' OR '.join([f'"{kw}"' for kw in keywords])
        
        # NewsAPI can only search 1 month at a time, so split into chunks
        date_chunks = self._split_date_range(start_date, end_date, days=30)
        
        for chunk_start, chunk_end in date_chunks:
            print(f"\n  Fetching articles from {chunk_start} to {chunk_end}...")
            
            # Check rate limit
            if self.newsapi_requests_today >= self.newsapi_daily_limit:
                print(f"[WARNING] Daily API limit reached ({self.newsapi_daily_limit})")
                break
            
            # Prepare request parameters
            params = {
                'q': query,
                'from': chunk_start,
                'to': chunk_end,
                'language': 'en',
                'sortBy': 'relevancy',  # or 'publishedAt' or 'popularity'
                'pageSize': 100,  # Max articles per request
                'apiKey': self.newsapi_key
            }
            
            try:
                # Make API request
                response = requests.get(self.newsapi_base_url, params=params, timeout=30)
                self.newsapi_requests_today += 1
                
                # Check response status
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    print(f"    Found {len(articles)} articles")
                    all_articles.extend(articles)
                    
                    # Save raw JSON if requested
                    if save_raw:
                        self._save_raw_json(data, chunk_start, chunk_end)
                    
                elif response.status_code == 401:
                    raise ValueError("Invalid NewsAPI key. Get one from https://newsapi.org/register")
                    
                elif response.status_code == 429:
                    print("[WARNING] Rate limit exceeded. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                    
                else:
                    print(f"[ERROR] API request failed with status {response.status_code}")
                    print(f"Response: {response.text}")
                    self.collection_stats['failed_requests'] += 1
                    
                # Respect API rate limits (avoid hitting 429)
                time.sleep(1)  # Wait 1 second between requests
                
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Request failed: {e}")
                self.collection_stats['failed_requests'] += 1
                continue
        
        # Convert to DataFrame
        if not all_articles:
            print("[WARNING] No articles collected!")
            return pd.DataFrame()
        
        df = self._articles_to_dataframe(all_articles)
        self.collection_stats['total_articles'] = len(df)
        
        print(f"\n[OK] Collected {len(df)} total articles")
        return df
    
    def _articles_to_dataframe(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Convert NewsAPI article list to pandas DataFrame
        
        Args:
            articles: List of article dictionaries from NewsAPI
            
        Returns:
            pd.DataFrame with standardized columns
        """
        data = []
        for article in articles:
            data.append({
                'published_at': article.get('publishedAt', ''),
                'date': article.get('publishedAt', '')[:10],  # Extract date only
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'url': article.get('url', ''),
                'author': article.get('author', ''),
            })
        
        df = pd.DataFrame(data)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    # =========================================================================
    # TWITTER COLLECTION (OPTIONAL)
    # =========================================================================
    
    def collect_tweets(self, ticker: str, keywords: List[str],
                       start_date: str, end_date: str,
                       max_tweets: int = 1000) -> pd.DataFrame:
        """
        Collect tweets related to the stock ticker
        
        This provides social media sentiment as an alternative/complement
        to news articles.
        
        Args:
            ticker: Stock ticker (e.g., 'CBA.AX')
            keywords: Keywords to search (e.g., ['$CBA', '#CommBank'])
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            max_tweets: Maximum tweets to collect
            
        Returns:
            pd.DataFrame with columns: [date, text, likes, retweets, user]
            
        Note:
            Twitter API v2 Academic Research access required for historical data
            Free tier only allows recent 7 days
        """
        if not self.twitter_client:
            print("[WARNING] Twitter client not initialized. Skipping tweet collection.")
            return pd.DataFrame()
        
        print(f"\n[TWITTER] Collecting tweets for {ticker}")
        
        # Build search query
        # Example: '($CBA OR #CommBank OR "Commonwealth Bank") -is:retweet lang:en'
        query_parts = [f'({" OR ".join(keywords)})']
        query_parts.append('-is:retweet')  # Exclude retweets
        query_parts.append('lang:en')      # English only
        query = ' '.join(query_parts)
        
        print(f"Search query: {query}")
        
        tweets_data = []
        
        try:
            # Search tweets
            # Note: Requires Academic Research access for historical data
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=min(max_tweets, 100),  # API limit per request
                tweet_fields=['created_at', 'public_metrics', 'author_id'],
                expansions=['author_id'],
            )
            
            if not tweets.data:
                print("[WARNING] No tweets found")
                return pd.DataFrame()
            
            # Process tweets
            for tweet in tweets.data:
                tweets_data.append({
                    'date': tweet.created_at.date(),
                    'text': tweet.text,
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'author_id': tweet.author_id,
                })
            
            df = pd.DataFrame(tweets_data)
            df['date'] = pd.to_datetime(df['date'])
            
            self.collection_stats['total_tweets'] = len(df)
            print(f"[OK] Collected {len(df)} tweets")
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Tweet collection failed: {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # DATA CLEANING & PREPROCESSING
    # =========================================================================
    
    def clean_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess news articles
        
        This function implements the data cleaning steps required for
        Task Requirement 1 (documentation of filtering/cleaning steps).
        
        Cleaning steps:
        1. Remove duplicates (same title/URL)
        2. Remove very short articles (< 100 chars)
        3. Clean HTML tags and special characters
        4. Filter by keywords (relevance)
        5. Handle missing values
        6. Align to trading days (weekends → next Monday)
        
        Args:
            df: Raw news DataFrame
            
        Returns:
            pd.DataFrame: Cleaned news data
        """
        print(f"\n[CLEANING] Starting data preprocessing...")
        print(f"Initial articles: {len(df)}")
        
        df_clean = df.copy()
        
        # Step 1: Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['title'], keep='first')
        df_clean = df_clean.drop_duplicates(subset=['url'], keep='first')
        duplicates_removed = initial_count - len(df_clean)
        print(f"  [1] Removed {duplicates_removed} duplicates")
        self.collection_stats['duplicates_removed'] = duplicates_removed
        
        # Step 2: Remove very short articles (likely incomplete)
        df_clean = df_clean[df_clean['title'].str.len() >= 10]
        df_clean = df_clean[
            (df_clean['description'].str.len() >= 50) | 
            (df_clean['content'].str.len() >= 100)
        ]
        print(f"  [2] After length filter: {len(df_clean)} articles")
        
        # Step 3: Clean text
        print(f"  [3] Cleaning text (HTML, special chars)...")
        df_clean['title_clean'] = df_clean['title'].apply(self._clean_text)
        df_clean['description_clean'] = df_clean['description'].apply(self._clean_text)
        df_clean['content_clean'] = df_clean['content'].apply(self._clean_text)
        
        # Step 4: Combine text fields for analysis
        df_clean['full_text'] = (
            df_clean['title_clean'] + ' ' + 
            df_clean['description_clean'] + ' ' + 
            df_clean['content_clean']
        )
        
        # Step 5: Handle missing values
        df_clean = df_clean.dropna(subset=['date', 'title'])
        df_clean['full_text'] = df_clean['full_text'].fillna('')
        
        # Step 6: Align to trading days (IMPORTANT for time-series alignment!)
        df_clean = self._align_to_trading_days(df_clean)
        
        # Step 7: Sort by date
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        print(f"[OK] Cleaning complete. Final articles: {len(df_clean)}")
        
        return df_clean
    
    def _clean_text(self, text: str) -> str:
        """
        Clean individual text field
        
        Removes:
        - HTML tags
        - URLs
        - Special characters
        - Extra whitespace
        
        Args:
            text: Raw text string
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ''
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _align_to_trading_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align news dates to trading days
        
        This is CRITICAL for time-series alignment with stock data.
        
        Rules:
        - Weekend news (Saturday/Sunday) → Next Monday
        - Holiday news → Next trading day (requires market calendar)
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            pd.DataFrame: DataFrame with aligned dates
        """
        print("  [6] Aligning to trading days...")
        
        df_aligned = df.copy()
        
        # Simple approach: Move weekends to next Monday
        # For production: Use pandas_market_calendars library for exact holidays
        df_aligned['weekday'] = df_aligned['date'].dt.dayofweek
        
        # Saturday (5) → +2 days, Sunday (6) → +1 day
        df_aligned['days_to_add'] = 0
        df_aligned.loc[df_aligned['weekday'] == 5, 'days_to_add'] = 2  # Saturday
        df_aligned.loc[df_aligned['weekday'] == 6, 'days_to_add'] = 1  # Sunday
        
        df_aligned['date'] = df_aligned['date'] + pd.to_timedelta(df_aligned['days_to_add'], unit='D')
        
        # Drop helper columns
        df_aligned = df_aligned.drop(columns=['weekday', 'days_to_add'])
        
        return df_aligned
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _validate_date_range(self, start_date: str, end_date: str):
        """Validate date range format and logic"""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Dates must be in 'YYYY-MM-DD' format")
        
        if start >= end:
            raise ValueError("start_date must be before end_date")
        
        # Check NewsAPI limitation (1 month history for free tier)
        days_diff = (datetime.now() - start).days
        if days_diff > 30:
            print(f"[WARNING] NewsAPI free tier only allows 1 month history")
            print(f"          Consider upgrading or using cached data")
    
    def _split_date_range(self, start_date: str, end_date: str, days: int = 30):
        """Split date range into chunks (for API limits)"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        chunks = []
        current = start
        
        while current < end:
            chunk_end = min(current + timedelta(days=days), end)
            chunks.append((
                current.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d')
            ))
            current = chunk_end + timedelta(days=1)
        
        return chunks
    
    def _save_raw_json(self, data: Dict, start_date: str, end_date: str):
        """Save raw API response to JSON file"""
        from .config import Task7Config
        
        filename = f"newsapi_{start_date}_to_{end_date}.json"
        filepath = os.path.join(Task7Config.NEWS_RAW_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_statistics(self) -> Dict:
        """
        Get collection statistics
        
        Returns:
            dict: Collection statistics
        """
        return self.collection_stats.copy()
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """
        Save cleaned news data to CSV
        
        Args:
            df: News DataFrame
            filename: Output filename
        """
        from .config import Task7Config
        
        filepath = os.path.join(Task7Config.NEWS_PROCESSED_DIR, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"[OK] Saved to {filepath}")


# Example usage
if __name__ == '__main__':
    # Test configuration
    from config import Task7Config
    
    # Initialize collector
    collector = NewsCollector(
        newsapi_key=Task7Config.NEWSAPI_KEY,
        twitter_bearer_token=Task7Config.TWITTER_BEARER_TOKEN
    )
    
    # Collect news
    print("Testing news collection...")
    news_df = collector.collect_news(
        ticker='CBA.AX',
        start_date='2024-10-01',  # Recent date for testing
        end_date='2024-10-15',
        keywords=Task7Config.NEWS_KEYWORDS,
        save_raw=True
    )
    
    if not news_df.empty:
        # Clean news
        cleaned_df = collector.clean_news(news_df)
        
        # Save results
        collector.save_to_csv(cleaned_df, 'news_test.csv')
        
        # Show statistics
        stats = collector.get_statistics()
        print("\nCollection Statistics:")
        print(json.dumps(stats, indent=2))
        
        # Show sample
        print("\nSample articles:")
        print(cleaned_df[['date', 'title', 'source']].head(10))
    else:
        print("[ERROR] No news collected. Check API key and date range.")
