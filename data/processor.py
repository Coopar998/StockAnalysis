"""
Stock Price Prediction System - Data Processor
---------------------------------------------
This file handles data preprocessing and sequence creation for model input.
"""

import numpy as np
import pandas as pd
from data.fetcher import download_stock_data
from data.features.basic_features import add_basic_features
from data.features.technical_indicators import add_technical_indicators

def prepare_data(ticker, start_date, end_date, min_data_points=250):
    """
    Download and prepare data for stock prediction
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data retrieval (YYYY-MM-DD)
        end_date: End date for data retrieval (YYYY-MM-DD)
        min_data_points: Minimum number of data points required
        
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
        
        # Add technical indicators
        add_technical_indicators(df)
        
        # Create target variable (next day's price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with NaN values
        df = df.dropna()
        print(f"Final shape after cleaning: {df.shape}")
        
        # Check one more time if we have enough data after all processing
        if len(df) < min_data_points:
            print(f"Insufficient data after all processing: only {len(df)} rows remain, need {min_data_points}")
            return None, f"Insufficient data after processing: only {len(df)} rows remain"
        
        return df, "Success"
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing {ticker}: {error_msg}")
        return None, f"Error: {error_msg}"

def create_sequences(data, seq_length=20):
    """
    Create sequences for the LSTM model
    
    Args:
        data: DataFrame with processed stock data
        seq_length: Number of time steps in each sequence
        
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
    features = [col for col in numeric_cols if col != 'Target']
    print(f"Using {len(features)} features")
    
    X, y = [], []
    dates = []
    raw_prices = []  # Store raw Close prices for reference
    
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:(i+seq_length)].values)
        y.append(data['Target'].iloc[i+seq_length])
        dates.append(data.index[i+seq_length])
        raw_prices.append(data['Close'].iloc[i+seq_length])
    
    return np.array(X), np.array(y), dates, features, np.array(raw_prices)

def combine_data_from_tickers(ticker_data_map, seq_length=20):
    """
    Combine data from multiple tickers for training a single model.
    
    Args:
        ticker_data_map: Dictionary mapping ticker symbols to their prepared dataframes
        seq_length: Length of sequences for LSTM
        
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
        X, y, _, features, _ = create_sequences(data, seq_length)
        
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