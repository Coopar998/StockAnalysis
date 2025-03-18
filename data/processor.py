"""
Stock Price Prediction System - Data Processor
---------------------------------------------
This file handles data preprocessing and sequence creation for model input.
Optimized for speed with selective feature processing.
"""

import numpy as np
import pandas as pd
from data.fetcher import download_stock_data
from data.features.basic_features import add_basic_features
from data.features.technical_indicators import add_technical_indicators

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

def prepare_data(ticker, start_date, end_date, min_data_points=250, lightweight_mode=False):
    """
    Download and prepare data for stock prediction
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data retrieval (YYYY-MM-DD)
        end_date: End date for data retrieval (YYYY-MM-DD)
        min_data_points: Minimum number of data points required
        lightweight_mode: If True, use faster processing with fewer indicators
        
    Returns:
        Tuple of (DataFrame, message) where DataFrame contains processed data
        or None if processing fails, and message contains success/error details
    """
    try:
        # Download stock data
        df = download_stock_data(ticker, start_date, end_date)
        
        if df is None:
            return None, "No data available"
            
        if len(df) < min_data_points:
            print(f"Insufficient data for {ticker}. Only {len(df)} data points available, minimum required is {min_data_points}")
            return None, f"Insufficient data: only {len(df)} days available, need {min_data_points}"
        
        # Ensure required columns exist
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            if col not in df.columns:
                if f"Adj {col}" in df.columns:
                    print(f"Using 'Adj {col}' instead of '{col}'")
                    df[col] = df[f"Adj {col}"]
                elif f"{col}_" in [c[:len(col)+1] for c in df.columns]:
                    matching_cols = [c for c in df.columns if c.startswith(f"{col}_")]
                    print(f"Using '{matching_cols[0]}' instead of '{col}'")
                    df[col] = df[matching_cols[0]]
                else:
                    print(f"Warning: '{col}' column not found!")
        
        # Add basic features
        add_basic_features(df)
        
        # Add technical indicators with lightweight mode option
        add_technical_indicators(df, lightweight_mode)
        
        # Create target variable (next day's price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with NaN values
        df = df.dropna()
        print(f"Final shape after cleaning: {df.shape}")
        
        # Optimize memory usage
        df = optimize_dataframe_memory(df)
        
        # Check one more time if we have enough data after all processing
        if len(df) < min_data_points:
            print(f"Insufficient data after all processing: only {len(df)} rows remain, need {min_data_points}")
            return None, f"Insufficient data after processing: only {len(df)} rows remain"
        
        return df, "Success"
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing {ticker}: {error_msg}")
        return None, f"Error: {error_msg}"

def create_sequences(data, seq_length=20, essential_only=False):
    """
    Create sequences for the LSTM model
    
    Args:
        data: DataFrame with processed stock data
        seq_length: Number of time steps in each sequence
        essential_only: If True, use only essential features for faster processing
        
    Returns:
        X: Features array with shape (n_samples, seq_length, n_features)
        y: Target array with shape (n_samples,)
        dates: List of dates corresponding to each target
        features: List of feature names
        raw_prices: Array of raw Close prices
    """
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
    
    print(f"Using {len(features)} features" + (" (essential only)" if essential_only else ""))
    
    # Use NumPy vectorized operations for faster sequence creation
    values = data[features].values
    target = data['Target'].values
    
    # Pre-allocate arrays for better performance
    n_samples = len(data) - seq_length
    X = np.zeros((n_samples, seq_length, len(features)), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    raw_prices = np.zeros(n_samples, dtype=np.float32)
    
    # Fill arrays using indexing
    for i in range(n_samples):
        X[i] = values[i:i+seq_length]
        y[i] = target[i+seq_length]
        raw_prices[i] = data['Close'].iloc[i+seq_length]
    
    # Get dates for each target
    dates = data.index[seq_length:].tolist()
    
    return X, y, dates, features, raw_prices

def combine_data_from_tickers(ticker_data_map, seq_length=20, essential_only=True):
    """
    Combine data from multiple tickers for training a single model.
    
    Args:
        ticker_data_map: Dictionary mapping ticker symbols to their prepared dataframes
        seq_length: Length of sequences for LSTM
        essential_only: If True, use only essential features for faster processing
        
    Returns:
        Combined X and y arrays for training, and ticker features information
    """
    all_X = []
    all_y = []
    ticker_features = {}
    
    for ticker, data in ticker_data_map.items():
        if data is None or len(data) < seq_length + 50:  # Ensure enough data
            continue
            
        # Create sequences
        X, y, _, features, _ = create_sequences(data, seq_length, essential_only)
        
        if len(X) > 0:
            # Store 80% for training
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            
            # Store features for reference
            ticker_features[ticker] = features
            
            # Add normalized data
            all_X.append(X_train)
            all_y.append(y_train)
            
            print(f"Added {len(X_train)} sequences from {ticker}")
    
    if not all_X:
        return None, None, None
        
    # Combine all data
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    
    print(f"Combined dataset: {X_combined.shape} sequences from {len(ticker_data_map)} tickers")
    
    return X_combined, y_combined, ticker_features