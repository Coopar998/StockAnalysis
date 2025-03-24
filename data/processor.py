"""
Stock Price Prediction System - Data Processor
---------------------------------------------
This file handles data preprocessing and sequence creation for model input.
Optimized for speed with selective feature processing and database integration.
"""

import numpy as np
import pandas as pd
import logging
from data.fetcher import download_stock_data
from datetime import datetime
from data.features.basic_features import add_basic_features
from data.features.technical_indicators import add_technical_indicators
from utils.database import StockDatabase
from utils.config_reader import get_config

# Initialize database
db = StockDatabase()

# Define essential features for faster processing
ESSENTIAL_FEATURES = [
    'Close', 'Open', 'High', 'Low', 'Volume',  # Base price data
    'Return', 'Close_Norm',  # Basic features
    'SMA_20', 'SMA_50',  # Moving averages
    'RSI_14',  # RSI
    'MACD', 'MACD_Signal',  # MACD
    'BB_Width', 'BB_Position',  # Bollinger Bands
    'ATR_Norm',  # Volatility
    'OBV',  # Volume
    'ROC_10',  # Momentum
    'Close_z_score'  # Mean reversion
]

def optimize_dataframe_memory(df):
    """
    Optimize DataFrame memory usage by converting to more efficient types
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
        
    for col in df.select_dtypes(include=['int64']).columns:
        # Convert int64 to more memory-efficient integer types when possible
        min_val, max_val = df[col].min(), df[col].max()
        if min_val >= 0:
            if max_val < 256:
                df[col] = df[col].astype('uint8')
            elif max_val < 65536:
                df[col] = df[col].astype('uint16')
        else:
            if min_val > -128 and max_val < 128:
                df[col] = df[col].astype('int8')
            elif min_val > -32768 and max_val < 32768:
                df[col] = df[col].astype('int16')
                
    return df

def prepare_data(ticker, start_date, end_date, min_data_points=None, lightweight_mode=None, verbose=False, force_download=False, max_age_days=1):
    """
    Download and prepare data for stock prediction, using database when available
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data retrieval (YYYY-MM-DD)
        end_date: End date for data retrieval (YYYY-MM-DD)
        min_data_points: Minimum number of data points required
        lightweight_mode: If True, use faster processing with fewer indicators
        verbose: Whether to print detailed information
        force_download: Whether to force download from API even if data exists in database
        max_age_days: Maximum age in days before data is considered stale
        
    Returns:
        Tuple of (DataFrame, message) where DataFrame contains processed data
        or None if processing fails, and message contains success/error details
    """
    logger = logging.getLogger('stock_prediction')
    config = get_config()
    
    # Use configured values if not explicitly provided
    if min_data_points is None:
        min_data_points = config.get('data_processing', 'min_data_points', default=250)
        
    if lightweight_mode is None:
        lightweight_mode = config.get('data_processing', 'lightweight_mode', default=True)
    
    try:
        # Check if we need to update data
        need_update = db.data_needs_update(ticker, max_age_days) if not force_download else True
        
        if need_update:
            if verbose:
                log_msg = "Downloading fresh data" if force_download else "Data needs update"
                logger.info(f"{log_msg} for {ticker}")
        else:
            if verbose:
                logger.info(f"Using recent data from database for {ticker}")
            
            # Get data from database with indicators included
            df = db.get_stock_data(ticker, start_date, end_date, include_indicators=True)
            
            if df is not None and not df.empty and len(df) >= min_data_points:
                # Verify all required columns are present
                if 'Close' in df.columns:
                    # Create target variable if needed
                    if 'Target' not in df.columns:
                        df['Target'] = df['Close'].shift(-1)
                        df = df.dropna()
                        
                    # Optimize memory usage
                    df = optimize_dataframe_memory(df)
                    
                    return df, "Success (from database)"
        
        # If we get here, we need to download and process data
        # Download stock data
        df = download_stock_data(ticker, start_date, end_date, force_download=force_download, max_age_days=max_age_days)
        
        if df is None:
            return None, "No data available"
            
        if len(df) < min_data_points:
            if verbose:
                logger.warning(f"Insufficient data for {ticker}. Only {len(df)} data points available, minimum required is {min_data_points}")
            return None, f"Insufficient data: only {len(df)} days available, need {min_data_points}"
        
        # Ensure required columns exist
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            if col not in df.columns:
                if f"Adj {col}" in df.columns:
                    # Silently use adjusted column
                    df[col] = df[f"Adj {col}"]
                elif f"{col}_" in [c[:len(col)+1] for c in df.columns]:
                    matching_cols = [c for c in df.columns if c.startswith(f"{col}_")]
                    # Silently use matching column
                    df[col] = df[matching_cols[0]]
                else:
                    if verbose:
                        logger.warning(f"Warning: '{col}' column not found for {ticker}!")
        
        # Add basic features
        add_basic_features(df, verbose=verbose)
        
        # Add technical indicators with lightweight mode option
        add_technical_indicators(df, lightweight_mode, verbose=verbose)
        
        # Create target variable (next day's price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with NaN values
        df = df.dropna()
        if verbose:
            logger.info(f"Final shape after cleaning for {ticker}: {df.shape}")
        
        # Optimize memory usage
        df = optimize_dataframe_memory(df)
        
        # Check one more time if we have enough data after all processing
        if len(df) < min_data_points:
            if verbose:
                logger.warning(f"Insufficient data for {ticker} after all processing: only {len(df)} rows remain, need {min_data_points}")
            return None, f"Insufficient data after processing: only {len(df)} rows remain"
        
        return df, "Success"
    
    except Exception as e:
        error_msg = str(e)
        if verbose:
            logger.error(f"Error processing {ticker}: {error_msg}")
        return None, f"Error: {error_msg}"

def create_sequences(data, seq_length=None, essential_only=False, verbose=False):
    """
    Create sequences for the LSTM model, with stride to reduce data leakage
    
    Args:
        data: DataFrame with processed stock data
        seq_length: Number of time steps in each sequence
        essential_only: If True, use only essential features for faster processing
        verbose: Whether to print detailed information
        
    Returns:
        X: Features array with shape (n_samples, seq_length, n_features)
        y: Target array with shape (n_samples,)
        dates: List of dates corresponding to each target
        features: List of feature names
        raw_prices: Array of raw Close prices
    """
    logger = logging.getLogger('stock_prediction')
    config = get_config()
    
    # Use configured values if not explicitly provided
    if seq_length is None:
        seq_length = config.get('data_processing', 'sequence_length', default=20)
    
    # Get stride from config (controls overlap between sequences)
    stride = config.get('data_processing', 'sequence_stride', default=5)
    
    # Select only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Exclude the target from features
    if essential_only and ESSENTIAL_FEATURES:
        # Use only essential features plus any that match the pattern
        features = [col for col in numeric_cols if (col in ESSENTIAL_FEATURES or 
                                                  any(col.startswith(prefix) for prefix in 
                                                     ['SMA_', 'EMA_', 'ROC_', 'RSI_'])) and 
                    col != 'Target']
    else:
        # Use all features
        features = [col for col in numeric_cols if col != 'Target']
    
    if verbose:
        logger.info(f"Using {len(features)} features" + (" (essential only)" if essential_only else ""))
    
    # Use NumPy vectorized operations for faster sequence creation
    values = data[features].values
    target = data['Target'].values
    
    # Use stride to control the overlap between sequences
    # Stride=1 means every point is used as a sequence start (maximum overlap)
    # Stride=seq_length means no overlap between sequences
    # Higher stride values reduce data leakage but also reduce the number of training examples
    n_samples = max(1, (len(data) - seq_length) // stride)
    
    if verbose:
        logger.info(f"Creating {n_samples} sequences with stride {stride}")
    
    # Pre-allocate arrays for better performance
    X = np.zeros((n_samples, seq_length, len(features)), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    raw_prices = np.zeros(n_samples, dtype=np.float32)
    
    # Fill arrays using indexing and stride
    for i in range(n_samples):
        idx = i * stride  # Use stride to reduce overlap between sequences
        X[i] = values[idx:idx+seq_length]
        y[i] = target[idx+seq_length-1]  # Target corresponds to last point in sequence
        raw_prices[i] = data['Close'].iloc[idx+seq_length-1]
    
    # Get dates for each target
    dates = data.index[np.array([i * stride + seq_length - 1 for i in range(n_samples)])].tolist()
    
    return X, y, dates, features, raw_prices

def combine_data_from_tickers(ticker_data_map, seq_length=None, essential_only=True, verbose=False):
    """
    Combine data from multiple tickers for training a single model.
    
    Args:
        ticker_data_map: Dictionary mapping ticker symbols to their prepared dataframes
        seq_length: Length of sequences for LSTM
        essential_only: If True, use only essential features for faster processing
        verbose: Whether to print detailed information
        
    Returns:
        Combined X and y arrays for training, and ticker features information
    """
    logger = logging.getLogger('stock_prediction')
    config = get_config()
    
    # Use configured values if not explicitly provided
    if seq_length is None:
        seq_length = config.get('data_processing', 'sequence_length', default=20)
    
    all_X = []
    all_y = []
    ticker_features = {}
    
    for ticker, data in ticker_data_map.items():
        if data is None or len(data) < seq_length + 50:  # Ensure enough data
            continue
            
        # Create sequences
        X, y, _, features, _ = create_sequences(data, seq_length, essential_only, verbose)
        
        if len(X) > 0:
            # Store 80% for training
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            
            # Store features for reference
            ticker_features[ticker] = features
            
            # Add normalized data
            all_X.append(X_train)
            all_y.append(y_train)
            
            if verbose:
                logger.info(f"Added {len(X_train)} sequences from {ticker}")
    
    if not all_X:
        return None, None, None
        
    # Combine all data
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    
    if verbose:
        logger.info(f"Combined dataset: {X_combined.shape} sequences from {len(ticker_data_map)} tickers")
    
    return X_combined, y_combined, ticker_features

def get_data_freshness_report():
    """
    Generate a report on data freshness for all stocks in the database
    
    Returns:
        DataFrame with freshness information
    """
    # Get all available tickers
    tickers = db.get_available_tickers()
    
    freshness_data = []
    for ticker in tickers:
        last_update = db.get_last_update_date(ticker)
        start_date, end_date = db.get_data_date_range(ticker)
        
        if last_update:
            days_old = (datetime.now() - last_update).days
            needs_update = days_old >= 1
            
            freshness_data.append({
                'ticker': ticker,
                'last_update': last_update,
                'days_since_update': days_old,
                'needs_update': needs_update,
                'start_date': start_date,
                'end_date': end_date
            })
    
    if freshness_data:
        return pd.DataFrame(freshness_data)
    
    return pd.DataFrame(columns=['ticker', 'last_update', 'days_since_update', 'needs_update', 'start_date', 'end_date'])