"""
Task C.7: Web-Based News Collection (Alternative to NewsAPI)

This module scrapes financial news from publicly available sources:
- Yahoo Finance News
- Reuters Business
- Google Finance
- Financial news RSS feeds

This approach is better for academic projects because:
1. No API key required
2. Free and unlimited access
3. Reproducible results
4. No rate limits

Author: Your Name
Date: November 2025
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import re
from urllib.parse import quote


class WebNewsCollector:
    """
    Collect financial news through web scraping
    
    This is the primary data collection method for Task C.7.
    It scrapes news from multiple public sources without requiring API keys.
    """
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize web scraper
        
        Args:
            delay: Delay between requests in seconds (be polite to servers)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.stats = {
            'total_articles': 0,
            'yahoo_finance': 0,
            'google_finance': 0,
            'rss_feeds': 0,
            'failed_requests': 0
        }
    
    # =========================================================================
    # MAIN COLLECTION FUNCTION
    # =========================================================================
    
    def collect_news(self, ticker: str, start_date: str, end_date: str,
                     company_name: str, max_articles: int = 200) -> pd.DataFrame:
        """
        Collect news articles from multiple web sources
        
        Args:
            ticker: Stock ticker (e.g., 'CBA.AX')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            company_name: Company name for search (e.g., 'Commonwealth Bank')
            max_articles: Maximum articles to collect
            
        Returns:
            DataFrame with columns: [date, title, description, content, source, url]
        """
        print(f"\n[WEB SCRAPING] Collecting news for {ticker}")
        print(f"Company: {company_name}")
        print(f"Date range: {start_date} to {end_date}")
        
        all_articles = []
        
        # Source 1: Yahoo Finance News
        print("\n[1/3] Scraping Yahoo Finance...")
        yahoo_articles = self._scrape_yahoo_finance(ticker, company_name, max_articles//3)
        all_articles.extend(yahoo_articles)
        self.stats['yahoo_finance'] = len(yahoo_articles)
        print(f"  Found {len(yahoo_articles)} articles from Yahoo Finance")
        
        time.sleep(self.delay)
        
        # Source 2: Google Finance News
        print("\n[2/3] Scraping Google Finance...")
        google_articles = self._scrape_google_finance(ticker, company_name, max_articles//3)
        all_articles.extend(google_articles)
        self.stats['google_finance'] = len(google_articles)
        print(f"  Found {len(google_articles)} articles from Google Finance")
        
        time.sleep(self.delay)
        
        # Source 3: RSS Feeds (Financial news)
        print("\n[3/3] Fetching RSS feeds...")
        rss_articles = self._fetch_rss_feeds(company_name, max_articles//3)
        all_articles.extend(rss_articles)
        self.stats['rss_feeds'] = len(rss_articles)
        print(f"  Found {len(rss_articles)} articles from RSS feeds")
        
        # Convert to DataFrame
        if len(all_articles) == 0:
            print("[WARNING] No articles collected!")
            return pd.DataFrame(columns=['date', 'title', 'description', 'content', 'source', 'url'])
        
        df = pd.DataFrame(all_articles)
        
        # Clean and filter
        df = self._clean_articles(df, start_date, end_date)
        
        self.stats['total_articles'] = len(df)
        print(f"\n[OK] Collected {len(df)} articles total")
        
        return df
    
    # =========================================================================
    # YAHOO FINANCE SCRAPER
    # =========================================================================
    
    def _scrape_yahoo_finance(self, ticker: str, company_name: str, 
                              max_articles: int) -> List[Dict]:
        """
        Scrape news from Yahoo Finance
        
        Yahoo Finance provides free access to financial news without login.
        """
        articles = []
        
        try:
            # Yahoo Finance news URL
            # Remove .AX suffix for Yahoo Finance search
            search_ticker = ticker.replace('.AX', '')
            url = f"https://finance.yahoo.com/quote/{search_ticker}/news"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"  [WARNING] Yahoo Finance returned status {response.status_code}")
                return articles
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles (Yahoo's HTML structure)
            news_items = soup.find_all('li', class_=re.compile('js-stream-content'))
            
            if not news_items:
                # Try alternative structure
                news_items = soup.find_all('div', class_=re.compile('Ov\\(h\\)'))
            
            for item in news_items[:max_articles]:
                try:
                    # Extract article details
                    title_elem = item.find('h3') or item.find('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    
                    # Get link
                    link = title_elem.get('href', '') if title_elem.name == 'a' else ''
                    if link and not link.startswith('http'):
                        link = 'https://finance.yahoo.com' + link
                    
                    # Get description
                    desc_elem = item.find('p')
                    description = desc_elem.get_text(strip=True) if desc_elem else ''
                    
                    # Get date (if available)
                    date_elem = item.find('time')
                    if date_elem:
                        date_str = date_elem.get('datetime', '')
                        try:
                            article_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            date = article_date.strftime('%Y-%m-%d')
                        except:
                            date = datetime.now().strftime('%Y-%m-%d')
                    else:
                        date = datetime.now().strftime('%Y-%m-%d')
                    
                    articles.append({
                        'date': date,
                        'title': title,
                        'description': description,
                        'content': description,  # Full content requires individual page scraping
                        'source': 'Yahoo Finance',
                        'url': link
                    })
                    
                except Exception as e:
                    continue
            
        except Exception as e:
            print(f"  [ERROR] Yahoo Finance scraping failed: {e}")
            self.stats['failed_requests'] += 1
        
        return articles
    
    # =========================================================================
    # GOOGLE FINANCE SCRAPER
    # =========================================================================
    
    def _scrape_google_finance(self, ticker: str, company_name: str,
                               max_articles: int) -> List[Dict]:
        """
        Scrape news from Google Finance
        
        Google Finance aggregates news from multiple sources.
        """
        articles = []
        
        try:
            # Google Finance URL
            # For Australian stocks, use ASX:CBA format
            if '.AX' in ticker:
                search_ticker = f"ASX:{ticker.replace('.AX', '')}"
            else:
                search_ticker = ticker
            
            url = f"https://www.google.com/finance/quote/{search_ticker}"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"  [WARNING] Google Finance returned status {response.status_code}")
                return articles
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news section
            news_section = soup.find('div', {'class': re.compile('yY3Lee')})
            
            if news_section:
                news_items = news_section.find_all('div', {'class': re.compile('z4rs2b')})
                
                for item in news_items[:max_articles]:
                    try:
                        # Extract title
                        title_elem = item.find('div', {'class': re.compile('Yfwt5')})
                        if not title_elem:
                            continue
                        
                        title = title_elem.get_text(strip=True)
                        
                        # Get source and date
                        source_elem = item.find('div', {'class': re.compile('sfyJob')})
                        source_text = source_elem.get_text(strip=True) if source_elem else ''
                        
                        # Parse "Source • 2 hours ago" format
                        parts = source_text.split('•')
                        source = parts[0].strip() if len(parts) > 0 else 'Google Finance'
                        
                        # Estimate date from relative time
                        date = datetime.now().strftime('%Y-%m-%d')
                        
                        # Get link
                        link_elem = item.find('a')
                        link = link_elem.get('href', '') if link_elem else ''
                        if link and not link.startswith('http'):
                            link = 'https://www.google.com' + link
                        
                        articles.append({
                            'date': date,
                            'title': title,
                            'description': '',
                            'content': title,
                            'source': f'Google Finance ({source})',
                            'url': link
                        })
                        
                    except Exception as e:
                        continue
            
        except Exception as e:
            print(f"  [ERROR] Google Finance scraping failed: {e}")
            self.stats['failed_requests'] += 1
        
        return articles
    
    # =========================================================================
    # RSS FEED COLLECTOR
    # =========================================================================
    
    def _fetch_rss_feeds(self, company_name: str, max_articles: int) -> List[Dict]:
        """
        Fetch news from financial RSS feeds
        
        RSS feeds are a reliable way to get structured news data.
        """
        articles = []
        
        # List of financial RSS feeds
        rss_feeds = [
            'https://www.ft.com/rss/companies/banks',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.reuters.com/rssFeed/businessNews',
        ]
        
        for feed_url in rss_feeds:
            try:
                response = self.session.get(feed_url, timeout=10)
                
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.content, 'xml')
                
                items = soup.find_all('item')
                
                for item in items:
                    try:
                        title = item.find('title').get_text(strip=True)
                        
                        # Filter by company name
                        if company_name.lower() not in title.lower():
                            continue
                        
                        description = item.find('description')
                        description = description.get_text(strip=True) if description else ''
                        
                        link = item.find('link')
                        link = link.get_text(strip=True) if link else ''
                        
                        pub_date = item.find('pubDate')
                        if pub_date:
                            try:
                                date_obj = datetime.strptime(
                                    pub_date.get_text(strip=True),
                                    '%a, %d %b %Y %H:%M:%S %z'
                                )
                                date = date_obj.strftime('%Y-%m-%d')
                            except:
                                date = datetime.now().strftime('%Y-%m-%d')
                        else:
                            date = datetime.now().strftime('%Y-%m-%d')
                        
                        articles.append({
                            'date': date,
                            'title': title,
                            'description': description,
                            'content': description,
                            'source': 'RSS Feed',
                            'url': link
                        })
                        
                        if len(articles) >= max_articles:
                            break
                        
                    except Exception as e:
                        continue
                
                if len(articles) >= max_articles:
                    break
                
                time.sleep(self.delay)
                
            except Exception as e:
                continue
        
        return articles
    
    # =========================================================================
    # DATA CLEANING
    # =========================================================================
    
    def _clean_articles(self, df: pd.DataFrame, start_date: str, 
                        end_date: str) -> pd.DataFrame:
        """
        Clean and filter collected articles
        
        Steps:
        1. Remove duplicates
        2. Filter by date range
        3. Remove empty titles
        4. Sort by date
        """
        if len(df) == 0:
            return df
        
        # Remove duplicates based on title
        original_count = len(df)
        df = df.drop_duplicates(subset=['title'], keep='first')
        print(f"  Removed {original_count - len(df)} duplicate articles")
        
        # Remove empty titles
        df = df[df['title'].str.strip() != '']
        
        # Filter by date range (best effort - some dates might be current)
        try:
            df['date'] = pd.to_datetime(df['date'])
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            # Keep articles within range or with current date (newly published)
            df = df[(df['date'] >= start) & (df['date'] <= end) | 
                    (df['date'] >= pd.Timestamp.now() - pd.Timedelta(days=7))]
            
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        except:
            pass
        
        # Sort by date
        df = df.sort_values('date')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Return collection statistics"""
        return self.stats.copy()
    
    def save_articles(self, df: pd.DataFrame, output_path: str):
        """
        Save collected articles to CSV
        
        Args:
            df: DataFrame of articles
            output_path: Path to save CSV file
        """
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"[OK] Saved {len(df)} articles to {output_path}")


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == '__main__':
    """
    Demo: Collect news for CBA.AX
    """
    collector = WebNewsCollector(delay=1.0)
    
    # Collect recent news (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    news_df = collector.collect_news(
        ticker='CBA.AX',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        company_name='Commonwealth Bank',
        max_articles=50
    )
    
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80)
    print(f"Total articles: {len(news_df)}")
    print(f"\nStats: {collector.get_stats()}")
    
    if len(news_df) > 0:
        print("\nSample articles:")
        print(news_df[['date', 'title', 'source']].head(5))
        
        # Save
        collector.save_articles(news_df, 'sample_news.csv')
