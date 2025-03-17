"""
Stock Price Prediction System - Basic Features
---------------------------------------------
This file handles the creation of basic price and volume-based features.
"""

def add_basic_features(df):
    """
    Add basic price and return features to dataframe
    
    Args:
        df: DataFrame with stock data (must contain at least Close column)
        
    Returns:
        None (modifies DataFrame in-place)
    """
    if 'High' in df.columns and 'Low' in df.columns:
        df['Range'] = df['High'] - df['Low']
    else:
        print("Skipping Range calculation due to missing columns")
    
    if 'Close' in df.columns:
        # Previous close and daily return
        df['PrevClose'] = df['Close'].shift(1)
        df['Return'] = df['Close'].pct_change() * 100
        
        # Add normalized price features to help with scaling
        df['Close_Norm'] = df['Close'] / df['Close'].rolling(window=20).mean()
        df['Price_Relative_50d'] = df['Close'] / df['Close'].rolling(window=50).mean()
        df['Price_Relative_200d'] = df['Close'] / df['Close'].rolling(window=200).mean()
    else:
        print("Skipping PrevClose and Return calculation due to missing Close column")
    
    if 'Volume' in df.columns:
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change() * 100
        df['Volume_Relative_20d'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Combine price and volume
        if 'Close' in df.columns:
            df['Price_Volume_Change'] = df['Return'] * df['Volume_Change']
    else:
        print("Skipping Volume features due to missing Volume column")
    
    # Add day of week feature (0=Monday, 4=Friday)
    if hasattr(df.index, 'dayofweek'):
        df['DayOfWeek'] = df.index.dayofweek
    
    # Add month feature (1-12)
    if hasattr(df.index, 'month'):
        df['Month'] = df.index.month