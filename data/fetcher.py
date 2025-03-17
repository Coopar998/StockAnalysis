"""
Stock Price Prediction System - Data Fetcher
-------------------------------------------
This file handles fetching stock data from Yahoo Finance API.
It provides functions to retrieve stock symbols and historical price data.
"""

import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import hashlib
import pickle

# Define cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_sp500_tickers():
    """Get tickers for S&P 500 companies"""
    cache_file = os.path.join(CACHE_DIR, 'sp500_tickers.json')
    
    # Check cache first
    if os.path.exists(cache_file):
        # Check if cache is less than 7 days old
        if datetime.now().timestamp() - os.path.getmtime(cache_file) < 7 * 24 * 3600:
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cached S&P 500 tickers: {e}")
    
    try:
        # Use Wikipedia table of S&P 500 companies
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(sp500_url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Clean tickers
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(tickers, f)
        
        print(f"Retrieved {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        print(f"Error retrieving S&P 500 tickers: {e}")
        # Return a default list of large cap stocks if the web scraping fails
        default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "PG"]
        return default_tickers

def get_top_market_cap_stocks(n=100):
    """Get the top n stocks by market capitalization"""
    cache_file = os.path.join(CACHE_DIR, f'top_{n}_market_cap.json')
    
    # Check cache first
    if os.path.exists(cache_file):
        # Check if cache is less than 1 day old
        if datetime.now().timestamp() - os.path.getmtime(cache_file) < 24 * 3600:
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cached top market cap stocks: {e}")
    
    try:
        # First get S&P 500 tickers
        all_tickers = get_sp500_tickers()
        
        # Get market cap for each ticker
        market_caps = {}
        for ticker in all_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if 'marketCap' in info and info['marketCap'] is not None:
                    market_caps[ticker] = info['marketCap']
                time.sleep(0.2)  # Short delay to avoid API rate limits
            except Exception as e:
                print(f"Error getting market cap for {ticker}: {e}")
                continue
        
        # Sort by market cap and get top n
        sorted_tickers = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
        top_tickers = [ticker for ticker, _ in sorted_tickers[:n]]
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(top_tickers, f)
        
        print(f"Selected top {len(top_tickers)} stocks by market cap")
        return top_tickers
    except Exception as e:
        print(f"Error getting top market cap stocks: {e}")
        # Return a default list if the process fails
        default_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V", 
                         "PG", "AVGO", "MA", "HD", "CVX", "MRK", "LLY", "PEP", "COST", "ABBV",
                         "KO", "WMT", "PFE", "CSCO", "TMO", "MCD", "ACN", "ABT", "CRM", "DHR",
                         "BAC", "ADBE", "CMCSA", "INTC", "VZ", "NKE", "DIS", "PM", "NEE", "UNH"]
        return default_tickers[:n]

def download_stock_data(ticker, start_date, end_date, cache=True):
    """
    Download historical stock data from Yahoo Finance with caching
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data retrieval (YYYY-MM-DD)
        end_date: End date for data retrieval (YYYY-MM-DD)
        cache: Whether to use caching (default: True)
    
    Returns:
        Pandas DataFrame with stock data or None if download fails
    """
    if cache:
        # Create a cache key based on the parameters
        cache_key = f"{ticker}_{start_date}_{end_date}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f'stock_data_{cache_hash}.pkl')
        
        # Check if cached data exists and is recent (less than 1 day old)
        if os.path.exists(cache_file):
            if datetime.now().timestamp() - os.path.getmtime(cache_file) < 24 * 3600:
                try:
                    with open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    print(f"Loaded cached data for {ticker} from {start_date} to {end_date}")
                    return df
                except Exception as e:
                    print(f"Error loading cached data: {e}")
    
    # Download data from Yahoo Finance with retries
    print(f"Downloading data for {ticker} from {start_date} to {end_date}")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                print(f"No data available for {ticker}")
                return None
                
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
            
            # Cache the result if caching is enabled
            if cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
            
            print(f"Downloaded {len(df)} rows of data for {ticker}")
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt+1} failed for {ticker}. Retrying...")
                time.sleep(2)  # Short delay before retry
            else:
                print(f"Failed to download data for {ticker} after {max_retries} attempts: {e}")
                return None