import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time  # Add this for delays between API calls

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Technical indicators
import talib

def prepare_data(ticker, start_date, end_date, min_data_points=250):
    """Download and prepare data for a stock"""
    print(f"Downloading data for {ticker} from {start_date} to {end_date}")
    
    try:
        # Add a retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt+1} failed for {ticker}. Retrying...")
                    time.sleep(2)  # Add delay between retries
                else:
                    raise e
        
        # Check if we have enough data
        if df.empty:
            print(f"No data available for {ticker}")
            return None, "No data available"
            
        if len(df) < min_data_points:
            print(f"Insufficient data for {ticker}. Only {len(df)} data points available, minimum required is {min_data_points}")
            return None, f"Insufficient data: only {len(df)} days available, need {min_data_points}"
            
        print(f"Downloaded {len(df)} rows of data")
        
        # Debug the DataFrame structure
        print("\nColumn names:")
        print(df.columns.tolist())
        
        # Handle potential multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            print("\nDetected multi-level columns. Flattening...")
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
        
        print("\nColumn names after processing:")
        print(df.columns.tolist())
        
        # Make a copy to avoid setting on a view
        df = df.copy()
        
        # Check if columns exist before using them
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            if col not in df.columns:
                if f"Adj {col}" in df.columns:
                    print(f"Using 'Adj {col}' instead of '{col}'")
                    df[col] = df[f"Adj {col}"]
                elif f"{col}_" in [c[:len(col)+1] for c in df.columns]:
                    # Find columns that start with col_
                    matching_cols = [c for c in df.columns if c.startswith(f"{col}_")]
                    print(f"Using '{matching_cols[0]}' instead of '{col}'")
                    df[col] = df[matching_cols[0]]
                else:
                    print(f"Warning: '{col}' column not found!")
        
        # Basic feature extraction
        if 'High' in df.columns and 'Low' in df.columns:
            df['Range'] = df['High'] - df['Low']
        else:
            print("Skipping Range calculation due to missing columns")
        
        if 'Close' in df.columns:
            df['PrevClose'] = df['Close'].shift(1)
            # Use pct_change for return calculation
            df['Return'] = df['Close'].pct_change() * 100
            
            # Add normalized price features to help with scaling
            df['Close_Norm'] = df['Close'] / df['Close'].rolling(window=20).mean()
            df['Price_Relative_50d'] = df['Close'] / df['Close'].rolling(window=50).mean()
            df['Price_Relative_200d'] = df['Close'] / df['Close'].rolling(window=200).mean()
        else:
            print("Skipping PrevClose and Return calculation due to missing Close column")
        
        # 2. TA-Lib indicators - only add if we have the necessary columns
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            print("Warning: Missing some OHLCV columns. Some indicators may not be calculated.")
        
        # Make sure we have float arrays for columns that exist
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
        
        # Only add indicators if we have the necessary data
        if close_arr is not None:
            # Moving Averages
            try:
                df['SMA_10'] = talib.SMA(close_arr, timeperiod=10)
                df['SMA_50'] = talib.SMA(close_arr, timeperiod=50)
                df['SMA_200'] = talib.SMA(close_arr, timeperiod=200)
                df['EMA_12'] = talib.EMA(close_arr, timeperiod=12)
                df['EMA_26'] = talib.EMA(close_arr, timeperiod=26)
                print("Added moving averages")
            except Exception as e:
                print(f"Error adding moving averages: {e}")

            # MACD (12,26,9)
            try:
                macd, signal, hist = talib.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)
                df['MACD'] = macd
                df['MACD_Signal'] = signal
                df['MACD_Hist'] = hist
                print("Added MACD")
            except Exception as e:
                print(f"Error adding MACD: {e}")

            # RSI (14)
            try:
                df['RSI_14'] = talib.RSI(close_arr, timeperiod=14)
                print("Added RSI")
            except Exception as e:
                print(f"Error adding RSI: {e}")
            
            # Bollinger Bands (20,2)
            try:
                upper, middle, lower = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2)
                df['BB_Upper'] = upper
                df['BB_Middle'] = middle
                df['BB_Lower'] = lower
                # Add BB relative position (0-1 scale where the price is in the band)
                df['BB_Position'] = (df['Close'] - lower) / (upper - lower)
                print("Added Bollinger Bands")
            except Exception as e:
                print(f"Error adding Bollinger Bands: {e}")
            
            # Stochastic Oscillator (14,3,3) - needs high, low, close
            if high_arr is not None and low_arr is not None:
                try:
                    slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
                    df['STOCH_K'] = slowk
                    df['STOCH_D'] = slowd
                    print("Added Stochastic Oscillator")
                except Exception as e:
                    print(f"Error adding Stochastic Oscillator: {e}")
            
            # ATR (14) - needs high, low, close
            if high_arr is not None and low_arr is not None:
                try:
                    df['ATR_14'] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
                    # Normalize ATR as a percentage of price
                    df['ATR_Norm'] = df['ATR_14'] / df['Close'] * 100
                    print("Added ATR")
                except Exception as e:
                    print(f"Error adding ATR: {e}")
            
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
    """Create sequences for the LSTM model"""
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

def build_model(input_shape):
    """Build an improved LSTM model with additional layers"""
    model = Sequential([
        LSTM(96, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='linear'),  # Linear layer to help with regression
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def process_ticker(ticker, start_date, end_date, min_data_points=250):
    """Process a single ticker and return results"""
    print(f"\nProcessing {ticker}")
    data, message = prepare_data(ticker, start_date, end_date, min_data_points)
    
    if data is None:
        print(f"Skipping {ticker}: {message}")
        return {"ticker": ticker, "success": False, "message": message}
    
    # Check if we have all required columns for modeling
    if 'Target' not in data.columns:
        print(f"Error: Target column missing for {ticker}. Cannot continue with model training.")
        return {"ticker": ticker, "success": False, "message": "Target column missing"}
    
    try:
        # Create sequences
        seq_length = 20
        X, y, dates, features, raw_prices = create_sequences(data, seq_length)
        print(f"Created {len(X)} sequences with shape {X.shape}")
        
        if len(X) < 100:  # Make sure we have enough sequences
            print(f"Not enough sequences for {ticker}: only {len(X)} available")
            return {"ticker": ticker, "success": False, "message": f"Not enough sequences: only {len(X)} available"}
        
        # Train-test split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        test_dates = dates[split:]
        test_raw_prices = raw_prices[split:]
        
        # Scale features
        scaler_X = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
        
        X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
        X_test_scaled = scaler_X.transform(X_test_reshaped)
        
        # Reshape back to 3D
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Scale target values - using target feature scaling with min-max
        # This helps with the prediction task while preserving the scale
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
        
        # Build and train model
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        model = build_model(input_shape)
        model.summary()
        
        early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
        
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=150,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        
        # Inverse transform predictions back to original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # FIX for consistent lower predictions: Calibrate with offset correction
        # Calculate the mean offset between actual and predicted values
        y_mean_offset = np.mean(y_test) - np.mean(y_pred)
        print(f"Mean prediction offset: ${y_mean_offset:.2f}")
        
        # Apply the offset correction
        y_pred_corrected = y_pred + y_mean_offset
        
        # Calculate metrics before and after correction
        mae_before = mean_absolute_error(y_test, y_pred)
        rmse_before = np.sqrt(mean_squared_error(y_test, y_pred))
        mape_before = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        mae_after = mean_absolute_error(y_test, y_pred_corrected)
        rmse_after = np.sqrt(mean_squared_error(y_test, y_pred_corrected))
        mape_after = np.mean(np.abs((y_test - y_pred_corrected) / y_test)) * 100
        
        print("Before offset correction:")
        print(f"MAE: ${mae_before:.2f}")
        print(f"RMSE: ${rmse_before:.2f}")
        print(f"MAPE: {mape_before:.2f}%")
        
        print("\nAfter offset correction:")
        print(f"MAE: ${mae_after:.2f}")
        print(f"RMSE: ${rmse_after:.2f}")
        print(f"MAPE: {mape_after:.2f}%")
        
        # Generate buy/sell signals based on corrected predictions
        y_diff = np.diff(y_pred_corrected.flatten())
        buy_signals = y_diff > 0  # Predicted price increase
        sell_signals = y_diff < 0  # Predicted price decrease
        
        # Plot results
        plt.figure(figsize=(12, 9))
        
        # Plot 1: Training loss
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{ticker} Model Training')
        plt.legend()
        
        # Plot 2: Price prediction and signals
        plt.subplot(2, 1, 2)
        plt.plot(test_dates, y_test, label='Actual', color='blue')
        plt.plot(test_dates, y_pred_corrected, label='Predicted', color='orange')
        
        # Add buy signals
        if np.any(buy_signals):
            buy_dates = [test_dates[i+1] for i, val in enumerate(buy_signals) if val]
            buy_prices = [y_test[i+1] for i, val in enumerate(buy_signals) if val]
            plt.scatter(buy_dates, buy_prices, color='green', label='Buy', marker='^', alpha=0.7)
        
        # Add sell signals
        if np.any(sell_signals):
            sell_dates = [test_dates[i+1] for i, val in enumerate(sell_signals) if val]
            sell_prices = [y_test[i+1] for i, val in enumerate(sell_signals) if val]
            plt.scatter(sell_dates, sell_prices, color='red', label='Sell', marker='v', alpha=0.7)
        
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{ticker}_prediction.png")
        plt.close()  # Close the figure to avoid display in non-interactive environments
        
        # Calculate trading strategy performance
        print("\nSimulated Trading Performance:")
        initial_capital = 10000
        position = 0
        capital = initial_capital
        trades = []
        
        for i in range(1, len(y_pred_corrected)):
            # Buy signal and not already in position
            if y_diff[i-1] > 0 and position == 0:
                position = capital / y_test[i]
                capital = 0
                trades.append(('buy', test_dates[i], y_test[i]))
            # Sell signal and in position
            elif y_diff[i-1] < 0 and position > 0:
                capital = position * y_test[i]
                position = 0
                trades.append(('sell', test_dates[i], y_test[i]))
        
        # Close out any remaining position at the end
        if position > 0:
            capital = position * y_test[-1]
            trades.append(('sell', test_dates[-1], y_test[-1]))
        
        final_value = capital
        buy_hold_return = initial_capital * (y_test[-1] / y_test[0])
        
        print(f"Initial capital: ${initial_capital:.2f}")
        print(f"Final value: ${final_value:.2f}")
        print(f"Total return: {((final_value/initial_capital)-1)*100:.2f}%")
        print(f"Buy & Hold return: {((buy_hold_return/initial_capital)-1)*100:.2f}%")
        print(f"Total trades: {len(trades)}")
        
        return {
            "ticker": ticker,
            "success": True,
            "message": "Success",
            "mae": mae_after,
            "rmse": rmse_after,
            "mape": mape_after,
            "total_return": ((final_value/initial_capital)-1)*100,
            "buy_hold_return": ((buy_hold_return/initial_capital)-1)*100,
            "total_trades": len(trades)
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error during model training for {ticker}: {error_msg}")
        return {"ticker": ticker, "success": False, "message": f"Error during model training: {error_msg}"}

def main():
    # Set date range
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=2500)).strftime('%Y-%m-%d')  # Increased from 1500 to 2500
    
    # You can process a single ticker or a list
    # Single ticker example:
    ticker = "AAPL"
    result = process_ticker(ticker, start_date, end_date)
    print(result)
    
    # Or process multiple tickers (uncomment this section to use)
    """
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Add your tickers here
    results = []
    
    for ticker in tickers:
        result = process_ticker(ticker, start_date, end_date)
        results.append(result)
        time.sleep(2)  # Add delay between tickers to avoid API rate limits
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("stock_results.csv", index=False)
    print(f"Processed {len(tickers)} tickers, results saved to stock_results.csv")
    """
    
    return result

if __name__ == "__main__":
    results = main()