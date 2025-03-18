"""
Stock Price Prediction System - Technical Indicators
--------------------------------------------------
This file handles the creation of technical indicators using TA-Lib.
Optimized for faster processing with lightweight mode.
"""

import numpy as np
import talib

def add_technical_indicators(df, lightweight_mode=False):
    """
    Add technical indicators to dataframe using TA-Lib
    
    Args:
        df: DataFrame with stock data (must contain OHLCV columns)
        lightweight_mode: If True, only add essential indicators for faster processing
        
    Returns:
        None (modifies DataFrame in-place)
    """
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        print("Warning: Missing some OHLCV columns. Some indicators may not be calculated.")
    
    # Prepare numpy arrays for TA-Lib
    if 'Open' in df.columns:
        open_arr = df['Open'].astype(float).values
    else:
        open_arr = None
        
    if 'High' in df.columns:
        high_arr = df['High'].astype(float).values
    else:
        high_arr = None
        
    if 'Low' in df.columns:
        low_arr = df['Low'].astype(float).values
    else:
        low_arr = None
        
    if 'Close' in df.columns:
        close_arr = df['Close'].astype(float).values
        print(f"Shape of Close array: {close_arr.shape}")
    else:
        close_arr = None
        print("No Close column found!")
        
    if 'Volume' in df.columns:
        volume_arr = df['Volume'].astype(float).values
    else:
        volume_arr = None
    
    if close_arr is not None:
        if lightweight_mode:
            # Lightweight mode: Only calculate essential indicators for faster processing
            add_essential_indicators(df, close_arr, high_arr, low_arr, volume_arr)
        else:
            # Full mode: Calculate all indicators
            add_moving_averages(df, close_arr)
            add_macd(df, close_arr)
            add_rsi(df, close_arr)
            add_bollinger_bands(df, close_arr)
            
            if high_arr is not None and low_arr is not None:
                add_stochastic(df, high_arr, low_arr, close_arr)
                add_atr(df, high_arr, low_arr, close_arr)
                
            if volume_arr is not None:
                # Add volume-based indicators
                add_volume_indicators(df, close_arr, volume_arr)
                
            # Add advanced indicators for improved predictions
            add_advanced_indicators(df)

def add_essential_indicators(df, close_arr, high_arr=None, low_arr=None, volume_arr=None):
    """Add only the most essential indicators for fast processing"""
    try:
        # Most important moving averages
        df['SMA_20'] = talib.SMA(close_arr, timeperiod=20)
        df['SMA_50'] = talib.SMA(close_arr, timeperiod=50)
        
        # MACD (essential)
        macd, signal, hist = talib.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        
        # RSI (essential)
        df['RSI_14'] = talib.RSI(close_arr, timeperiod=14)
        
        # Essential Bollinger Band metrics
        upper, middle, lower = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_Upper'] = upper
        df['BB_Lower'] = lower
        df['BB_Width'] = (upper - lower) / middle
        
        # Add ATR if we have the data
        if high_arr is not None and low_arr is not None:
            df['ATR_14'] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
            df['ATR_Norm'] = df['ATR_14'] / df['Close'] * 100
            
        # Add OBV if we have volume data
        if volume_arr is not None:
            df['OBV'] = talib.OBV(close_arr, volume_arr)
            
        # Add basic momentum
        df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
        
        # Z-score for mean reversion (essential)
        df['Close_MA20'] = df['Close'].rolling(window=20).mean()  # Already calculated as SMA_20, but keeping for clarity
        df['Close_MA20_std'] = df['Close'].rolling(window=20).std()
        df['Close_z_score'] = (df['Close'] - df['Close_MA20']) / df['Close_MA20_std']
            
        print("Added essential technical indicators")
    except Exception as e:
        print(f"Error adding essential indicators: {e}")
    
def add_moving_averages(df, close_arr):
    """Add moving averages to dataframe"""
    try:
        df['SMA_10'] = talib.SMA(close_arr, timeperiod=10)
        df['SMA_50'] = talib.SMA(close_arr, timeperiod=50)
        df['SMA_200'] = talib.SMA(close_arr, timeperiod=200)
        df['EMA_12'] = talib.EMA(close_arr, timeperiod=12)
        df['EMA_26'] = talib.EMA(close_arr, timeperiod=26)
        
        # Add crossover signals
        df['SMA_20'] = talib.SMA(close_arr, timeperiod=20)
        df['SMA_50_200_Ratio'] = df['SMA_50'] / df['SMA_200']
        df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
        df['Death_Cross'] = (df['SMA_50'] < df['SMA_200']).astype(int)
        df['EMA_Cross'] = (df['EMA_12'] > df['EMA_26']).astype(int)
        
        print("Added moving averages")
    except Exception as e:
        print(f"Error adding moving averages: {e}")

def add_macd(df, close_arr):
    """Add MACD indicator to dataframe"""
    try:
        macd, signal, hist = talib.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        df['MACD_Cross'] = ((df['MACD'] > df['MACD_Signal']).astype(int) - 
                            (df['MACD'] < df['MACD_Signal']).astype(int))
        print("Added MACD")
    except Exception as e:
        print(f"Error adding MACD: {e}")

def add_rsi(df, close_arr):
    """Add RSI indicator to dataframe"""
    try:
        df['RSI_14'] = talib.RSI(close_arr, timeperiod=14)
        
        # Add RSI-based signals
        df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
        df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)
        
        # Add RSI divergence (simplified)
        df['RSI_5d_chg'] = df['RSI_14'].diff(5)
        df['Close_5d_chg'] = df['Close'].diff(5)
        df['RSI_Divergence'] = ((df['RSI_5d_chg'] < 0) & (df['Close_5d_chg'] > 0) | 
                              (df['RSI_5d_chg'] > 0) & (df['Close_5d_chg'] < 0)).astype(int)
        
        print("Added RSI")
    except Exception as e:
        print(f"Error adding RSI: {e}")

def add_bollinger_bands(df, close_arr):
    """Add Bollinger Bands to dataframe"""
    try:
        upper, middle, lower = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        # Add BB width and relative position
        df['BB_Width'] = (upper - lower) / middle
        df['BB_Position'] = (df['Close'] - lower) / (upper - lower)
        
        # Add BB signals
        df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(window=50).mean() * 0.8).astype(int)
        df['BB_Upper_Break'] = (df['Close'] > df['BB_Upper']).astype(int)
        df['BB_Lower_Break'] = (df['Close'] < df['BB_Lower']).astype(int)
        
        print("Added Bollinger Bands")
    except Exception as e:
        print(f"Error adding Bollinger Bands: {e}")

def add_stochastic(df, high_arr, low_arr, close_arr):
    """Add Stochastic Oscillator to dataframe"""
    try:
        slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr, 
                                   fastk_period=14, slowk_period=3, 
                                   slowk_matype=0, slowd_period=3, 
                                   slowd_matype=0)
        df['STOCH_K'] = slowk
        df['STOCH_D'] = slowd
        
        # Add Stochastic signals
        df['STOCH_Overbought'] = ((df['STOCH_K'] > 80) & (df['STOCH_D'] > 80)).astype(int)
        df['STOCH_Oversold'] = ((df['STOCH_K'] < 20) & (df['STOCH_D'] < 20)).astype(int)
        df['STOCH_Cross'] = ((df['STOCH_K'] > df['STOCH_D']).astype(int) - 
                            (df['STOCH_K'] < df['STOCH_D']).astype(int))
        
        print("Added Stochastic Oscillator")
    except Exception as e:
        print(f"Error adding Stochastic Oscillator: {e}")

def add_atr(df, high_arr, low_arr, close_arr):
    """Add Average True Range to dataframe"""
    try:
        df['ATR_14'] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
        
        # Normalize ATR as a percentage of price
        df['ATR_Norm'] = df['ATR_14'] / df['Close'] * 100
        
        # Add ATR signals
        df['ATR_High'] = (df['ATR_Norm'] > df['ATR_Norm'].rolling(window=20).mean() * 1.5).astype(int)
        df['ATR_Low'] = (df['ATR_Norm'] < df['ATR_Norm'].rolling(window=20).mean() * 0.5).astype(int)
        
        print("Added ATR")
    except Exception as e:
        print(f"Error adding ATR: {e}")

def add_volume_indicators(df, close_arr, volume_arr):
    """Add volume-based indicators to dataframe"""
    try:
        # On-Balance Volume (OBV)
        df['OBV'] = talib.OBV(close_arr, volume_arr)
        df['OBV_SMA'] = talib.SMA(df['OBV'].values, timeperiod=20)
        df['OBV_Signal'] = (df['OBV'] > df['OBV_SMA']).astype(int)
        
        # Chaikin A/D Line
        if all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
            df['AD'] = talib.AD(df['High'].values, df['Low'].values, close_arr, volume_arr)
            df['AD_SMA'] = talib.SMA(df['AD'].values, timeperiod=20)
            df['AD_Signal'] = (df['AD'] > df['AD_SMA']).astype(int)
        
        # Money Flow Index
        if all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
            df['MFI'] = talib.MFI(df['High'].values, df['Low'].values, close_arr, volume_arr, timeperiod=14)
            df['MFI_Overbought'] = (df['MFI'] > 80).astype(int)
            df['MFI_Oversold'] = (df['MFI'] < 20).astype(int)
        
        print("Added volume indicators")
    except Exception as e:
        print(f"Error adding volume indicators: {e}")

def add_advanced_indicators(df):
    """
    Add advanced technical indicators for improved prediction
    
    Args:
        df: DataFrame with stock data (must contain OHLCV columns)
        
    Returns:
        None (modifies DataFrame in-place)
    """
    # Ensure we have the required data
    if not all(col in df.columns for col in ['Close', 'High', 'Low']):
        print("Warning: Missing required columns for advanced indicators")
        return
    
    # 1. Calculate price momentum indicators
    try:
        # Rate of Change (ROC) - momentum indicator
        df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
        df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
        df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
        
        # Add momentum crossover signals
        df['ROC_5_10_Cross'] = ((df['ROC_5'] > df['ROC_10']).astype(int) - 
                              (df['ROC_5'] < df['ROC_10']).astype(int))
        
        print("Added momentum indicators")
    except Exception as e:
        print(f"Error adding momentum indicators: {e}")
    
    # 2. Add mean reversion indicators
    try:
        # Z-score based on 20-day moving average
        df['Close_MA20'] = df['Close'].rolling(window=20).mean()
        df['Close_MA20_std'] = df['Close'].rolling(window=20).std()
        df['Close_z_score'] = (df['Close'] - df['Close_MA20']) / df['Close_MA20_std']
        
        # Mean reversion signals
        df['Overbought'] = (df['Close_z_score'] > 2).astype(int)
        df['Oversold'] = (df['Close_z_score'] < -2).astype(int)
        
        print("Added mean reversion indicators")
    except Exception as e:
        print(f"Error adding mean reversion indicators: {e}")
    
    # 3. Add volatility indicators
    try:
        # Historical volatility (standard deviation of returns)
        df['Volatility_10d'] = df['Close'].pct_change().rolling(window=10).std() * np.sqrt(252) * 100
        df['Volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
        
        # Volatility regime (high/low)
        df['Volatility_Regime'] = (df['Volatility_20d'] > df['Volatility_20d'].rolling(window=50).mean()).astype(int)
        
        print("Added volatility indicators")
    except Exception as e:
        print(f"Error adding volatility indicators: {e}")
    
    # 4. Add time-based indicators
    if hasattr(df.index, 'dayofweek'):
        # Day of week (0=Monday, 4=Friday)
        df['DayOfWeek'] = df.index.dayofweek
        
        # Month (1-12)
        if hasattr(df.index, 'month'):
            df['Month'] = df.index.month
            
            # Quarter (1-4)
            if hasattr(df.index, 'quarter'):
                df['Quarter'] = df.index.quarter
        
        print("Added time-based indicators")
    
    # 5. Add advanced pattern recognition
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        try:
            # Detect common candlestick patterns
            # Doji (open and close are almost the same)
            doji_threshold = 0.1  # % difference threshold
            df['Doji'] = (abs(df['Open'] - df['Close']) / df['Close'] < doji_threshold).astype(int)
            
            # Hammer (small body at the top, long lower shadow)
            df['Body'] = abs(df['Close'] - df['Open'])
            df['UpperShadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['LowerShadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            
            df['Hammer'] = ((df['Body'] < df['High'] - df['Low'] * 0.3) & 
                            (df['LowerShadow'] > df['Body'] * 2) &
                            (df['UpperShadow'] < df['Body'] * 0.5)).astype(int)
            
            # Engulfing patterns
            df['PrevOpen'] = df['Open'].shift(1)
            df['PrevClose'] = df['Close'].shift(1)
            
            # Bullish engulfing
            df['BullishEngulfing'] = ((df['Close'] > df['Open']) &
                                   (df['Open'] < df['PrevClose']) &
                                   (df['Close'] > df['PrevOpen']) &
                                   (df['PrevClose'] < df['PrevOpen'])).astype(int)
            
            # Bearish engulfing
            df['BearishEngulfing'] = ((df['Close'] < df['Open']) &
                                   (df['Open'] > df['PrevClose']) &
                                   (df['Close'] < df['PrevOpen']) &
                                   (df['PrevClose'] > df['PrevOpen'])).astype(int)
            
            print("Added pattern recognition indicators")
        except Exception as e:
            print(f"Error adding pattern recognition: {e}")
    
    # 6. Add support/resistance level indicators
    try:
        lookback = 50  # Period to consider for support/resistance
        
        # Find potential support/resistance levels using recent highs and lows
        if len(df) >= lookback:
            # Recent high as resistance
            recent_high = df['High'].rolling(window=lookback).max()
            # Recent low as support
            recent_low = df['Low'].rolling(window=lookback).min()
            
            # Distance from current price to support/resistance (percentage)
            df['Distance_To_Resistance'] = (recent_high - df['Close']) / df['Close'] * 100
            df['Distance_To_Support'] = (df['Close'] - recent_low) / df['Close'] * 100
            
            # Approaching support/resistance
            df['Near_Resistance'] = (df['Distance_To_Resistance'] < 2).astype(int)
            df['Near_Support'] = (df['Distance_To_Support'] < 2).astype(int)
            
            # Breaking support/resistance
            df['Break_Resistance'] = ((df['Close'] > recent_high) & 
                                   (df['Close'].shift(1) <= recent_high.shift(1))).astype(int)
            
            df['Break_Support'] = ((df['Close'] < recent_low) & 
                                (df['Close'].shift(1) >= recent_low.shift(1))).astype(int)
            
            print("Added support/resistance indicators")
        else:
            print("Not enough data for support/resistance calculation")
    except Exception as e:
        print(f"Error calculating support/resistance: {e}")
    
    # 7. Calculate combined signal strength
    try:
        # Select columns that could be bullish signals
        bullish_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                     ['bullish', 'oversold', 'break_resistance', 'hammer'])]
        
        # Select columns that could be bearish signals
        bearish_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                      ['bearish', 'overbought', 'break_support'])]
        
        # Calculate signal strength (if any signal columns exist)
        if bullish_cols:
            df['Bullish_Strength'] = df[bullish_cols].sum(axis=1)
        
        if bearish_cols:
            df['Bearish_Strength'] = df[bearish_cols].sum(axis=1)
            
        if bullish_cols and bearish_cols:
            df['Signal_Strength'] = df['Bullish_Strength'] - df['Bearish_Strength']
            
        print("Added signal strength indicators")
    except Exception as e:
        print(f"Error calculating signal strength: {e}")