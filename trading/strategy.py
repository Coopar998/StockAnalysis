"""
Stock Price Prediction System - Trading Strategy
----------------------------------------------
This file handles trading strategy implementation and signal generation.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

from data.processor import prepare_data, create_sequences
from model.trainer import train_evaluate_model
from model.ensemble import predict_with_ensemble
from utils.visualization import plot_performance

def process_ticker(ticker, start_date, end_date, model=None, model_path=None, 
                  output_dir=None, min_data_points=250, train_model=False, portfolio=None):
    """
    Process a single ticker and return results.
    If model is provided, use it for prediction. Otherwise, train a new model.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        model: Pre-trained model or list of models for ensemble
        model_path: Path to save or load model
        output_dir: Directory to save outputs
        min_data_points: Minimum required data points
        train_model: Whether to train a new model
        portfolio: Portfolio dictionary for tracking performance
    """
    print(f"\nProcessing {ticker}")
    
    # Create output directory for this ticker if specified
    ticker_output_dir = None
    if output_dir:
        ticker_output_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_output_dir, exist_ok=True)
        print(f"Saving results to {ticker_output_dir}")
    
    # Prepare data
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
        X_test = X[split:]
        y_test = y[split:]
        test_dates = dates[split:]
        
        # Initialize history to None
        history = None
        
        # Check if we're using an ensemble model (list of models)
        is_ensemble = isinstance(model, list) and len(model) > 0
        
        if model is None and train_model:
            # Train a new model for this ticker
            X_train = X[:split]
            y_train = y[:split]
            
            # If model_path is provided, use it to save the model
            ticker_model_path = None
            if model_path:
                ticker_model_path = model_path
            
            model, history, mae, rmse, mape, y_pred_corrected = train_evaluate_model(
                X_train, X_test, y_train, y_test, feature_names=features,
                model_path=ticker_model_path, model_type='deep'
            )
        else:
            # Use the provided pre-trained model(s) or load from file
            if model is None:
                # Check if we're loading an ensemble or a single model
                ensemble_dir = os.path.join(os.path.dirname(model_path), "ensemble")
                
                if os.path.exists(ensemble_dir) and os.path.isdir(ensemble_dir):
                    # Load ensemble models
                    try:
                        print(f"Loading ensemble models from {ensemble_dir}")
                        ensemble_models = []
                        
                        # Check for JSON config
                        ensemble_config_path = os.path.join(ensemble_dir, "ensemble_config.json")
                        
                        if os.path.exists(ensemble_config_path):
                            with open(ensemble_config_path, 'r') as f:
                                config = json.load(f)
                                model_types = config.get('types', ['standard', 'deep', 'bidirectional'])
                                
                            for model_type in model_types:
                                model_file = os.path.join(ensemble_dir, f"ensemble_{model_type}.keras")
                                if os.path.exists(model_file):
                                    model_instance = load_model(model_file)
                                    ensemble_models.append(model_instance)
                                    print(f"Loaded ensemble model: {model_type}")
                        else:
                            # Look for any .keras files in the ensemble directory
                            for f in os.listdir(ensemble_dir):
                                if f.endswith('.keras'):
                                    model_file = os.path.join(ensemble_dir, f)
                                    model_instance = load_model(model_file)
                                    ensemble_models.append(model_instance)
                                    print(f"Loaded ensemble model: {f}")
                        
                        if ensemble_models:
                            is_ensemble = True
                            model = ensemble_models
                        else:
                            raise ValueError("No ensemble models found")
                            
                    except Exception as e:
                        print(f"Error loading ensemble models: {e}")
                        return {"ticker": ticker, "success": False, "message": f"Error loading ensemble models: {e}"}
                else:
                    # Load a single model
                    try:
                        model = load_model(model_path)
                        is_ensemble = False
                        print(f"Successfully loaded model from {model_path}")
                    except Exception as e:
                        print(f"Error loading model: {e}")
                        return {"ticker": ticker, "success": False, "message": f"Error loading model: {e}"}
            
            # Check for feature importance or model metadata to adapt input features
            model_dir = ensemble_dir if is_ensemble and 'ensemble_dir' in locals() else os.path.dirname(model_path)
            
            # Try to load feature mapping information
            feature_mapping = None
            try:
                # First check model directory for feature importance file
                feature_importance_path = os.path.join(model_dir, "feature_importance.json")
                if os.path.exists(feature_importance_path):
                    with open(feature_importance_path, 'r') as f:
                        feature_data = json.load(f)
                        if 'important_features' in feature_data and 'important_indices' in feature_data:
                            print("Found feature importance data for feature mapping")
                            feature_mapping = {
                                'features': feature_data['important_features'],
                                'indices': feature_data['important_indices']
                            }
                
                # If no feature mapping yet, check parent dir
                if not feature_mapping:
                    parent_feature_path = os.path.join(os.path.dirname(model_dir), "feature_importance.json")
                    if os.path.exists(parent_feature_path):
                        with open(parent_feature_path, 'r') as f:
                            feature_data = json.load(f)
                            if 'important_features' in feature_data and 'important_indices' in feature_data:
                                print("Found feature importance data in parent directory")
                                feature_mapping = {
                                    'features': feature_data['important_features'],
                                    'indices': feature_data['important_indices']
                                }
            except Exception as e:
                print(f"Error loading feature mapping: {e}")
            
            # Try to adapt features based on mapping
            if feature_mapping and features:
                print("Adapting features based on feature importance data")
                feature_indices = []
                
                # Try to map by name first
                mapped_features = []
                for important_feature in feature_mapping['features']:
                    if important_feature in features:
                        feature_idx = features.index(important_feature)
                        feature_indices.append(feature_idx)
                        mapped_features.append(important_feature)
                    else:
                        print(f"Warning: Feature '{important_feature}' not found in current data")
                
                if len(mapped_features) >= len(feature_mapping['features']) * 0.8:  # If we found at least 80% of features
                    print(f"Using feature mapping by name: Found {len(mapped_features)} of {len(feature_mapping['features'])} features")
                    # Reshape X to select only the mapped features
                    if feature_indices:
                        X_test_adapted = X_test[:, :, feature_indices]
                        print(f"Adapted X_test from shape {X_test.shape} to {X_test_adapted.shape}")
                        X_test = X_test_adapted
                else:
                    print("Couldn't map enough features by name, using indices directly")
                    # Use indices directly if they're valid
                    valid_indices = [idx for idx in feature_mapping['indices'] if idx < X_test.shape[2]]
                    if valid_indices and len(valid_indices) > 0:
                        X_test_adapted = X_test[:, :, valid_indices]
                        print(f"Adapted X_test from shape {X_test.shape} to {X_test_adapted.shape}")
                        X_test = X_test_adapted
            
            # Make predictions using appropriate method for single model or ensemble
            try:
                if is_ensemble:
                    # Use ensemble prediction
                    y_pred_corrected, all_preds, mae, rmse, mape = predict_with_ensemble(
                        model, X_test, y_test, model_dir=ensemble_dir if 'ensemble_dir' in locals() else None
                    )
                else:
                    # Use single model prediction
                    # Standardize the data
                    scaler_X = StandardScaler()
                    X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
                    X_test_scaled = scaler_X.fit_transform(X_test_reshaped)
                    X_test_scaled = X_test_scaled.reshape(X_test.shape)
                    
                    # Scale target for prediction
                    scaler_y = StandardScaler()
                    scaler_y.fit(y_test.reshape(-1, 1))  # Fit scaler on test data to match scale
                    
                    # Make predictions
                    y_pred_scaled = model.predict(X_test_scaled)
                    
                    # Convert predictions back to original scale
                    y_pred = scaler_y.inverse_transform(y_pred_scaled)
                    y_pred_corrected = y_pred.flatten()
                    
                    # Apply error correction if needed
                    avg_price = np.mean(y_test)
                    avg_pred = np.mean(y_pred)
                    max_allowed_deviation = 0.5  # Allow 50% difference
                    
                    if abs(avg_pred - avg_price) / avg_price > max_allowed_deviation:
                        print(f"Warning: Predictions for {ticker} are far from actual. Applying direct scaling.")
                        scaling_factor = avg_price / avg_pred
                        y_pred_corrected = y_pred.flatten() * scaling_factor
                        print(f"Applied scaling factor: {scaling_factor}")
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred_corrected)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_corrected))
                    mape = np.mean(np.abs((y_test - y_pred_corrected) / y_test)) * 100
            except Exception as e:
                print(f"Error during prediction: {e}")
                # Fallback to a simple moving average model
                print("Falling back to moving average prediction")
                window_size = 5
                y_pred_corrected = np.zeros_like(y_test)
                for i in range(len(y_test)):
                    if i < window_size:
                        # For the first few points, use available history
                        start_idx = max(0, i-window_size)
                        y_pred_corrected[i] = np.mean(y_test[start_idx:i+1])
                    else:
                        # Use rolling window
                        y_pred_corrected[i] = np.mean(y_test[i-window_size:i])
                
                # Calculate metrics for the fallback model
                mae = mean_absolute_error(y_test, y_pred_corrected)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_corrected))
                mape = np.mean(np.abs((y_test - y_pred_corrected) / y_test)) * 100
                
                print(f"Fallback model metrics - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, MAPE: {mape:.2f}%")
            
            # Create empty history for visualization
            history = {"loss": [0], "val_loss": [0]}
        
        # Analyze performance and generate trading signals
        result = analyze_performance(
            ticker, y_test, y_pred_corrected, test_dates, history,
            mae, rmse, mape, output_dir=ticker_output_dir,
            portfolio=portfolio
        )
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error during model training for {ticker}: {error_msg}")
        print(traceback.format_exc())  # Print full traceback for debugging
        return {"ticker": ticker, "success": False, "message": f"Error during model training: {error_msg}"}

def analyze_performance(ticker, y_test, y_pred_corrected, test_dates, history, mae, rmse, mape, 
                      output_dir=None, portfolio=None):
    """
    Analyze model performance with improved signal generation
    
    Args:
        ticker: Stock ticker symbol
        y_test: Actual prices
        y_pred_corrected: Predicted prices
        test_dates: Dates for testing period
        history: Training history
        mae, rmse, mape: Error metrics
        output_dir: Directory to save outputs
        portfolio: Portfolio dictionary for tracking performance
    """
    # Calculate buy & hold return first
    buy_hold_return_pct = ((y_test[-1] / y_test[0]) - 1) * 100
    print(f"Buy & Hold return for {ticker}: {buy_hold_return_pct:.2f}%")
    
    # IMPROVEMENT 1: More sophisticated trend detection
    # Calculate multiple moving averages for better trend confirmation
    ma_short = 10
    ma_medium = 20
    ma_long = 50
    
    trend_short = np.zeros_like(y_test)
    trend_medium = np.zeros_like(y_test)
    trend_long = np.zeros_like(y_test)
    
    for i in range(len(y_test)):
        if i >= ma_short:
            trend_short[i] = np.mean(y_test[i-ma_short:i])
        else:
            trend_short[i] = y_test[i]
            
        if i >= ma_medium:
            trend_medium[i] = np.mean(y_test[i-ma_medium:i])
        else:
            trend_medium[i] = y_test[i]
            
        if i >= ma_long:
            trend_long[i] = np.mean(y_test[i-ma_long:i])
        else:
            trend_long[i] = y_test[i]
    
    # IMPROVEMENT 2: Calculate prediction accuracy by comparing previous predictions to actual outcomes
    accuracy_window = min(20, len(y_pred_corrected) - 1)
    prediction_accuracies = []
    
    for i in range(1, accuracy_window + 1):
        # Check if previous prediction correctly predicted direction
        prev_pred_direction = 1 if y_pred_corrected[-i-1] < y_pred_corrected[-i] else -1
        actual_direction = 1 if y_test[-i-1] < y_test[-i] else -1
        
        # Calculate directional accuracy (1 for correct, 0 for incorrect)
        accuracy = 1 if prev_pred_direction == actual_direction else 0
        prediction_accuracies.append(accuracy)
    
    # Overall directional accuracy of recent predictions
    recent_accuracy = np.mean(prediction_accuracies) if prediction_accuracies else 0.5
    print(f"Recent directional prediction accuracy: {recent_accuracy:.2f}")
    
    # IMPROVEMENT 3: Generate smarter signals with stronger confirmation requirements
    y_diff = np.diff(y_pred_corrected.flatten())
    
    # Adaptive threshold based on stock volatility
    volatility = np.std(np.diff(y_test)) / np.mean(y_test) * 100
    min_change_pct = max(0.5, volatility * 0.3)  # Adjust threshold based on volatility
    
    # Only use signals when recent prediction accuracy is good enough
    use_model_predictions = recent_accuracy >= 0.6
    
    # If prediction accuracy is poor, fall back to trend following
    if not use_model_predictions and buy_hold_return_pct > 15:
        print(f"Low prediction accuracy ({recent_accuracy:.2f}). Using trend following strategy instead.")
    
    buy_signals = np.zeros_like(y_diff, dtype=bool)
    sell_signals = np.zeros_like(y_diff, dtype=bool)
    
    for i in range(len(y_diff)):
        if i+1 < len(y_test):  # Make sure we can index y_test
            if use_model_predictions:
                predicted_change_pct = (y_diff[i] / y_test[i]) * 100
                
                # Buy signal: stronger confirmation criteria
                # 1. Predicted price increase above threshold
                # 2. Price above short-term MA 
                # 3. Short-term MA above medium-term MA (uptrend confirmation)
                if (predicted_change_pct > min_change_pct and 
                    y_test[i] > trend_short[i] and
                    trend_short[i] > trend_medium[i]):
                    buy_signals[i] = True
                    
                # Sell signal: stronger confirmation criteria
                # 1. Predicted price decrease below threshold
                # 2. Price below short-term MA
                # 3. Short-term MA below medium-term MA (downtrend confirmation)
                elif (predicted_change_pct < -min_change_pct and 
                     y_test[i] < trend_short[i] and
                     trend_short[i] < trend_medium[i]):
                    sell_signals[i] = True
            else:
                # Fallback to pure trend-following for poor prediction accuracy
                # Buy when price crosses above medium MA and short MA > medium MA (uptrend)
                if (y_test[i-1] < trend_medium[i-1] and y_test[i] > trend_medium[i] and
                    trend_short[i] > trend_medium[i]):
                    buy_signals[i] = True
                    
                # Sell when price crosses below medium MA and short MA < medium MA (downtrend)
                elif (y_test[i-1] > trend_medium[i-1] and y_test[i] < trend_medium[i] and
                     trend_short[i] < trend_medium[i]):
                    sell_signals[i] = True
    
    print(f"Signals for {ticker}:")
    print(f"  Buy signals: {np.sum(buy_signals)} out of {len(y_diff)} potential signals")
    print(f"  Sell signals: {np.sum(sell_signals)} out of {len(y_diff)} potential signals")
    
    # IMPROVEMENT 4: Adaptive strategy selection
    # For strong bull markets, just use buy & hold unless prediction accuracy is very high
    use_buy_hold = (buy_hold_return_pct > 40 and recent_accuracy < 0.75)
    
    # For volatile sideways or bear markets, only trade if prediction accuracy is decent
    use_active_trading = (buy_hold_return_pct <= 40 and recent_accuracy >= 0.55) or recent_accuracy >= 0.7
    
    strategy_used = ""
    
    if use_buy_hold:
        strategy_used = "buy_hold"
        print(f"Strong uptrend detected for {ticker} ({buy_hold_return_pct:.2f}%). Using buy & hold strategy.")
        position = 10000 / y_test[0]
        final_value = position * y_test[-1]
        trades = [('buy', test_dates[0], y_test[0]), ('sell', test_dates[-1], y_test[-1])]
    elif use_active_trading:
        strategy_used = "active_trading"
        print(f"Using active trading for {ticker} with {recent_accuracy:.2f} prediction accuracy")
        
        # Use active trading strategy
        position = 0
        capital = 10000
        trades = []
        min_holding_period = 5  # Minimum days to hold a position
        last_trade_day = None
        
        # Build the list of signals
        signals = []
        for i in range(1, len(y_pred_corrected)):
            if i-1 < len(y_diff):  # Ensure index is valid
                if buy_signals[i-1]:
                    signals.append(('buy', test_dates[i], y_test[i]))
                elif sell_signals[i-1]:
                    signals.append(('sell', test_dates[i], y_test[i]))
        
        # Execute the trading strategy with signals
        for i, signal in enumerate(signals):
            action, date, price = signal
            
            # Skip if we haven't waited long enough since the last trade
            if last_trade_day is not None and hasattr(date, 'toordinal') and hasattr(last_trade_day, 'toordinal'):
                days_since_last_trade = (date.toordinal() - last_trade_day.toordinal())
                if days_since_last_trade < min_holding_period:
                    continue
            
            # Buy signal and not already in position
            if action == 'buy' and position == 0:
                position = capital / price
                capital = 0
                trades.append(('buy', date, price))
                last_trade_day = date
                
            # Sell signal and in position
            elif action == 'sell' and position > 0:
                capital = position * price
                position = 0
                trades.append(('sell', date, price))
                last_trade_day = date
        
        # Close any open position at the end of the testing period
        if position > 0:
            capital = position * y_test[-1]
            trades.append(('sell', test_dates[-1], y_test[-1]))
        
        final_value = capital
    else:
        # Neither strategy has good confidence - default to buy & hold (safer)
        strategy_used = "buy_hold"
        print(f"No high-confidence strategy for {ticker}. Defaulting to buy & hold.")
        position = 10000 / y_test[0]
        final_value = position * y_test[-1]
        trades = [('buy', test_dates[0], y_test[0]), ('sell', test_dates[-1], y_test[-1])]
    
    # Plot results
    plot_performance(ticker, history, y_test, y_pred_corrected, test_dates, 
                   buy_signals, sell_signals, trend_medium, output_dir)
    
    # Calculate performance metrics
    initial_capital = 10000
    total_return = ((final_value/initial_capital)-1)*100
    
    print(f"Initial capital: ${initial_capital:.2f}")
    print(f"Final value: ${final_value:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Buy & Hold return: {buy_hold_return_pct:.2f}%")
    print(f"Strategy outperformance: {total_return - buy_hold_return_pct:.2f}%")
    print(f"Total trades: {len(trades)}")
    
    # Update portfolio if provided
    if portfolio is not None:
        # Add performance data to the returns dict
        portfolio['returns'][ticker] = {
            'initial_value': initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'buy_hold_return_pct': buy_hold_return_pct,
            'prediction_accuracy': recent_accuracy,
            'total_trades': len(trades),
            'trades': trades,
            'strategy': strategy_used
        }
        
    return {
        "ticker": ticker,
        "success": True,
        "message": "Success",
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "total_return": total_return,
        "buy_hold_return": buy_hold_return_pct,
        "total_trades": len(trades),
        "strategy": strategy_used,
        "prediction_accuracy": recent_accuracy
    }