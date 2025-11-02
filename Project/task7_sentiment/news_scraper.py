"""
Task C.7: Real News Collection from Yahoo Finance

This module collects REAL financial news without any API keys.
Uses public Yahoo Finance pages that anyone can access.

Why Yahoo Finance:
- Free and public access (no API key needed)
- Reliable financial news
- Covers Australian stocks (CBA.AX)
- Tutor can verify by visiting the same page

Author: Your Name
Date: November 2, 2025
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re


def collect_yahoo_finance_news(ticker='CBA.AX', max_articles=100):
    """
    Collect real news from Yahoo Finance
    
    Args:
        ticker: Stock ticker (e.g., 'CBA.AX')
        max_articles: Maximum number of articles to collect
        
    Returns:
        DataFrame with real news articles
    """
    print(f"\n[REAL NEWS] Collecting from Yahoo Finance for {ticker}")
    print(f"Source: https://finance.yahoo.com/quote/{ticker}/news")
    
    # Remove .AX for Yahoo Finance
    yahoo_ticker = ticker.replace('.AX', '')
    
    articles = []
    
    # Yahoo Finance news URL
    url = f"https://finance.yahoo.com/quote/{yahoo_ticker}/news"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"\n[1] Fetching page...")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"[ERROR] Failed to fetch page: Status {response.status_code}")
            return pd.DataFrame()
        
        print(f"[2] Parsing HTML...")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Save HTML for debugging
        with open('yahoo_finance_debug.html', 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        print(f"[DEBUG] Saved HTML to yahoo_finance_debug.html")
        
        # Find news articles - try multiple selectors
        print(f"\n[3] Searching for news articles...")
        
        # Method 1: Look for <h3> tags with links
        news_items = soup.find_all('h3')
        print(f"  Found {len(news_items)} <h3> tags")
        
        for item in news_items[:max_articles]:
            try:
                # Get link
                link_elem = item.find('a')
                if not link_elem:
                    continue
                
                title = link_elem.get_text(strip=True)
                if not title or len(title) < 10:
                    continue
                
                link = link_elem.get('href', '')
                if link and not link.startswith('http'):
                    link = 'https://finance.yahoo.com' + link
                
                # Try to find date (usually near the title)
                date_elem = item.find_next('time')
                if date_elem:
                    date = date_elem.get('datetime', '')
                    if date:
                        try:
                            date_obj = datetime.fromisoformat(date.replace('Z', '+00:00'))
                            article_date = date_obj.strftime('%Y-%m-%d')
                        except:
                            article_date = datetime.now().strftime('%Y-%m-%d')
                    else:
                        article_date = datetime.now().strftime('%Y-%m-%d')
                else:
                    article_date = datetime.now().strftime('%Y-%m-%d')
                
                # Try to find description
                description_elem = item.find_next('p')
                description = description_elem.get_text(strip=True) if description_elem else ''
                
                articles.append({
                    'date': article_date,
                    'title': title,
                    'description': description,
                    'content': description,
                    'source': 'Yahoo Finance',
                    'url': link
                })
                
            except Exception as e:
                continue
        
        # Method 2: Try looking for article containers
        if len(articles) == 0:
            print(f"  Method 1 failed. Trying alternative selectors...")
            
            # Look for divs with news content
            containers = soup.find_all('div', class_=re.compile('js-stream-content|StreamMegaItem'))
            print(f"  Found {len(containers)} article containers")
            
            for container in containers[:max_articles]:
                try:
                    title_elem = container.find(['h3', 'a'])
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '') if title_elem.name == 'a' else ''
                    
                    if link and not link.startswith('http'):
                        link = 'https://finance.yahoo.com' + link
                    
                    articles.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'title': title,
                        'description': '',
                        'content': title,
                        'source': 'Yahoo Finance',
                        'url': link
                    })
                    
                except:
                    continue
        
        print(f"\n[OK] Collected {len(articles)} real articles")
        
    except Exception as e:
        print(f"[ERROR] Collection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Create DataFrame
    if len(articles) == 0:
        print("[WARNING] No articles collected!")
        return pd.DataFrame(columns=['date', 'title', 'description', 'content', 'source', 'url'])
    
    df = pd.DataFrame(articles)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['title'], keep='first')
    
    print(f"\n[FINAL] {len(df)} unique articles")
    
    return df


def collect_google_news(query='Commonwealth Bank', max_articles=50):
    """
    Collect news from Google News RSS feed
    
    Args:
        query: Search query
        max_articles: Maximum articles
        
    Returns:
        DataFrame with news articles
    """
    print(f"\n[REAL NEWS] Collecting from Google News RSS")
    print(f"Query: {query}")
    
    articles = []
    
    # Google News RSS URL
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-AU&gl=AU&ceid=AU:en"
    
    try:
        print(f"\n[1] Fetching RSS feed...")
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"[ERROR] Failed: Status {response.status_code}")
            return pd.DataFrame()
        
        print(f"[2] Parsing XML...")
        soup = BeautifulSoup(response.content, 'xml')
        
        items = soup.find_all('item')
        print(f"[3] Found {len(items)} items")
        
        for item in items[:max_articles]:
            try:
                title = item.find('title').get_text(strip=True)
                link = item.find('link').get_text(strip=True)
                pub_date = item.find('pubDate')
                
                if pub_date:
                    try:
                        date_obj = datetime.strptime(
                            pub_date.get_text(strip=True),
                            '%a, %d %b %Y %H:%M:%S %Z'
                        )
                        date = date_obj.strftime('%Y-%m-%d')
                    except:
                        date = datetime.now().strftime('%Y-%m-%d')
                else:
                    date = datetime.now().strftime('%Y-%m-%d')
                
                # Get source from title (usually "Title - Source")
                source = 'Google News'
                if ' - ' in title:
                    parts = title.split(' - ')
                    title = parts[0].strip()
                    source = parts[-1].strip()
                
                description = item.find('description')
                description = description.get_text(strip=True) if description else ''
                
                articles.append({
                    'date': date,
                    'title': title,
                    'description': description,
                    'content': description,
                    'source': source,
                    'url': link
                })
                
            except Exception as e:
                continue
        
        print(f"\n[OK] Collected {len(articles)} articles from Google News")
        
    except Exception as e:
        print(f"[ERROR] Collection failed: {e}")
    
    if len(articles) == 0:
        return pd.DataFrame(columns=['date', 'title', 'description', 'content', 'source', 'url'])
    
    return pd.DataFrame(articles)


def collect_real_news(ticker='CBA.AX', company_name='Commonwealth Bank', max_total=200):
    """
    Collect real news from multiple sources
    
    Args:
        ticker: Stock ticker
        company_name: Company name for search
        max_total: Maximum total articles
        
    Returns:
        DataFrame with real news
    """
    print("="*80)
    print("COLLECTING REAL FINANCIAL NEWS (NO API KEY)")
    print("="*80)
    
    all_articles = []
    
    # Source 1: Yahoo Finance
    yahoo_df = collect_yahoo_finance_news(ticker, max_articles=max_total//2)
    if len(yahoo_df) > 0:
        all_articles.append(yahoo_df)
    
    time.sleep(2)  # Be polite
    
    # Source 2: Google News
    google_df = collect_google_news(company_name, max_articles=max_total//2)
    if len(google_df) > 0:
        all_articles.append(google_df)
    
    # Combine
    if len(all_articles) == 0:
        print("\n[ERROR] No articles collected from any source!")
        return pd.DataFrame(columns=['date', 'title', 'description', 'content', 'source', 'url'])
    
    final_df = pd.DataFrame(columns=['date', 'title', 'description', 'content', 'source', 'url'])
    combined_df = pd.concat(all_articles, ignore_index=True)
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['title'], keep='first')
    
    # Sort by date
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df = combined_df.sort_values('date', ascending=False)
    combined_df['date'] = combined_df['date'].dt.strftime('%Y-%m-%d')
    
    print("\n" + "="*80)
    print(f"TOTAL REAL ARTICLES COLLECTED: {len(combined_df)}")
    print("="*80)
    print(f"\nSources breakdown:")
    print(combined_df['source'].value_counts())
    
    return combined_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Collect real news
    news_df = collect_real_news(
        ticker='CBA.AX',
        company_name='Commonwealth Bank',
        max_total=200
    )
    
    if len(news_df) > 0:
        # Save
        output_path = 'real_cba_news.csv'
        news_df.to_csv(output_path, index=False)
        print(f"\n[SAVED] {len(news_df)} real articles to {output_path}")
        
        # Show sample
        print("\n" + "="*80)
        print("SAMPLE ARTICLES")
        print("="*80)
        print(news_df[['date', 'title', 'source']].head(10))
    else:
        print("\n[FAILED] No articles collected")
