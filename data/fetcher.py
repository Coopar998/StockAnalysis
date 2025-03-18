"""
Stock Price Prediction System - Data Fetcher
-------------------------------------------
This file handles fetching stock data from Yahoo Finance API.
It provides functions to retrieve stock symbols and historical price data.
Optimized for batch processing and efficient caching.
Added rate limiting protection.
"""

import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import hashlib
import pickle
import concurrent.futures
import random

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

def get_market_cap_with_retry(ticker, max_retries=3, base_delay=2.0):
    """Get market cap for a single ticker with retry and backoff"""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get('marketCap')
            
            if market_cap is not None:
                return market_cap
            else:
                # If no market cap found but no error, return None after small delay
                time.sleep(base_delay)
                return None
                
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if rate limited
            if "rate" in error_str and "limit" in error_str:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                print(f"Rate limited for {ticker}. Retrying after {delay:.2f}s (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                # Other error, wait a smaller amount
                print(f"Error getting market cap for {ticker}: {e}")
                time.sleep(base_delay)
                
    # Max retries reached
    print(f"Failed to get market cap for {ticker} after {max_retries} attempts")
    return None

def get_top_market_cap_stocks(n=100):
    """Get the top n stocks by market capitalization"""
    cache_file = os.path.join(CACHE_DIR, f'top_{n}_market_cap.json')
    
    # Check cache first
    if os.path.exists(cache_file):
        # Check if cache is less than 1 day old
        if datetime.now().timestamp() - os.path.getmtime(cache_file) < 24 * 3600:
            try:
                with open(cache_file, 'r') as f:
                    cached_tickers = json.load(f)
                    print(f"Loaded {len(cached_tickers)} top market cap tickers from cache")
                    return cached_tickers
            except Exception as e:
                print(f"Error loading cached top market cap stocks: {e}")
    
    try:
        # First get S&P 500 tickers
        all_tickers = get_sp500_tickers()
        
        # Get market cap for each ticker in smaller batches with delays to avoid rate limiting
        batch_size = 5  # Reduced batch size
        max_workers = 2  # Reduced concurrent workers
        market_caps = {}
        
        print(f"Fetching market cap data for {len(all_tickers)} tickers in batches of {batch_size}")
        
        for i in range(0, len(all_tickers), batch_size):
            batch = all_tickers[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(all_tickers) + batch_size - 1)//batch_size}: {batch}")
            
            # Use serial processing if batch size is small enough
            if batch_size <= 5:
                for ticker in batch:
                    market_cap = get_market_cap_with_retry(ticker)
                    if market_cap is not None:
                        market_caps[ticker] = market_cap
                    # Add delay between each ticker to avoid rate limiting
                    time.sleep(1.5)
            else:
                # Use limited parallel processing for larger batches
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_ticker = {executor.submit(get_market_cap_with_retry, ticker): ticker for ticker in batch}
                    
                    for future in concurrent.futures.as_completed(future_to_ticker):
                        ticker = future_to_ticker[future]
                        try:
                            market_cap = future.result()
                            if market_cap is not None:
                                market_caps[ticker] = market_cap
                        except Exception as e:
                            print(f"Error processing {ticker}: {e}")
            
            # Longer delay between batches
            batch_delay = 3.0
            print(f"Batch complete. Waiting {batch_delay}s before next batch...")
            time.sleep(batch_delay)
        
        # Sort by market cap and get top n
        sorted_tickers = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
        top_tickers = [ticker for ticker, _ in sorted_tickers[:n]]
        
        # If we don't have enough stocks with market cap data, supplement with default list
        if len(top_tickers) < n:
            default_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V", 
                             "PG", "AVGO", "MA", "HD", "CVX", "MRK", "LLY", "PEP", "COST", "ABBV"]
            
            # Add defaults that aren't already in our list
            for ticker in default_tickers:
                if ticker not in top_tickers:
                    top_tickers.append(ticker)
                    if len(top_tickers) >= n:
                        break
        
        # Trim to exact size
        top_tickers = top_tickers[:n]
        
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
            error_str = str(e).lower()
            
            # Check if rate limited
            if "rate" in error_str and "limit" in error_str:
                delay = 5.0 * (2 ** attempt) + random.uniform(0.1, 1.0)
                print(f"Rate limited when downloading {ticker}. Retrying after {delay:.2f}s (Attempt {attempt+1}/{max_retries})")
            else:
                delay = 2.0
                print(f"Attempt {attempt+1} failed for {ticker}. Retrying after {delay:.2f}s: {e}")
                
            time.sleep(delay)  # Longer delay before retry for rate limiting
            
    print(f"Failed to download data for {ticker} after {max_retries} attempts")
    return None

def download_multiple_stock_data(tickers, start_date, end_date, cache=True):
    """
    Download data for multiple stocks efficiently
    
    Args:
        tickers: List of ticker symbols
        start_date, end_date: Date range
        cache: Whether to use caching
        
    Returns:
        Dictionary mapping tickers to DataFrames
    """
    if not tickers:
        return {}
    
    result = {}
    cache_hits = 0
    
    # First try to load from cache
    if cache:
        for ticker in tickers:
            cache_key = f"{ticker}_{start_date}_{end_date}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = os.path.join(CACHE_DIR, f'stock_data_{cache_hash}.pkl')
            
            if os.path.exists(cache_file) and datetime.now().timestamp() - os.path.getmtime(cache_file) < 24 * 3600:
                try:
                    with open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    result[ticker] = df
                    cache_hits += 1
                except Exception as e:
                    print(f"Error loading cached data for {ticker}: {e}")
    
    # Get the remaining tickers that weren't in cache
    remaining_tickers = [t for t in tickers if t not in result]
    
    if not remaining_tickers:
        print(f"Loaded all {len(tickers)} tickers from cache")
        return result
    
    print(f"Loaded {cache_hits} tickers from cache, downloading {len(remaining_tickers)} remaining tickers")
    
    # Process remaining tickers in smaller batches to avoid rate limiting
    # Yahoo Finance performs better with smaller batches
    batch_size = 5  # Smaller batch size to avoid rate limiting
    
    for i in range(0, len(remaining_tickers), batch_size):
        batch = remaining_tickers[i:i+batch_size]
        print(f"Downloading batch {i//batch_size + 1}/{(len(remaining_tickers) + batch_size - 1)//batch_size} with {len(batch)} tickers...")
        
        try:
            # Download batch
            data = yf.download(batch, start=start_date, end=end_date, group_by='ticker', progress=False)
            
            # Check if we got a single ticker result (not grouped)
            if len(batch) == 1 and not isinstance(data.columns, pd.MultiIndex):
                ticker = batch[0]
                # Store the single ticker data
                if not data.empty:
                    if cache:
                        cache_key = f"{ticker}_{start_date}_{end_date}"
                        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
                        cache_file = os.path.join(CACHE_DIR, f'stock_data_{cache_hash}.pkl')
                        with open(cache_file, 'wb') as f:
                            pickle.dump(data, f)
                    result[ticker] = data
            else:
                # Process each ticker in the batch
                for ticker in batch:
                    if ticker in data.columns.levels[0]:
                        # Extract ticker data
                        ticker_data = data[ticker].copy()
                        
                        # Flatten column names if needed
                        if isinstance(ticker_data.columns, pd.MultiIndex):
                            ticker_data.columns = [col[1] if isinstance(col, tuple) else col for col in ticker_data.columns]
                        
                        # Skip if empty
                        if ticker_data.empty:
                            print(f"No data available for {ticker}")
                            continue
                        
                        # Cache the data
                        if cache:
                            cache_key = f"{ticker}_{start_date}_{end_date}"
                            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
                            cache_file = os.path.join(CACHE_DIR, f'stock_data_{cache_hash}.pkl')
                            with open(cache_file, 'wb') as f:
                                pickle.dump(ticker_data, f)
                        
                        # Add to result
                        result[ticker] = ticker_data
            
            # Avoid hitting API limits - longer delay between batches
            time.sleep(3.0)
            
        except Exception as e:
            print(f"Error downloading batch data: {e}")
            # If batch download fails, try individual downloads with delay between each
            for ticker in batch:
                if ticker not in result:
                    try:
                        ticker_data = download_stock_data(ticker, start_date, end_date, cache)
                        if ticker_data is not None:
                            result[ticker] = ticker_data
                        # Add delay between individual downloads
                        time.sleep(1.5)
                    except Exception as individual_e:
                        print(f"Error downloading individual data for {ticker}: {individual_e}")
    
    print(f"Successfully downloaded data for {len(result)} out of {len(tickers)} tickers")
    return result

def clear_stock_data_cache(max_age_days=7):
    """
    Clear old cache files to free up disk space
    
    Args:
        max_age_days: Maximum age of cache files in days
        
    Returns:
        Number of files removed
    """
    max_age_seconds = max_age_days * 24 * 3600
    removed = 0
    
    for filename in os.listdir(CACHE_DIR):
        if filename.startswith('stock_data_') and filename.endswith('.pkl'):
            file_path = os.path.join(CACHE_DIR, filename)
            file_age = datetime.now().timestamp() - os.path.getmtime(file_path)
            
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    removed += 1
                except Exception as e:
                    print(f"Error removing cache file {file_path}: {e}")
    
    print(f"Removed {removed} old cache files")
    return removed