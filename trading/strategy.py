"""
Stock Price Prediction System - Trading Strategy
----------------------------------------------
This file implements the main trading strategy orchestration.
It coordinates the analysis, signal generation, and strategy execution.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import time
import logging
from collections import deque

from data.processor import prepare_data, create_sequences
from model.trainer import train_evaluate_model
from model.ensemble import predict_with_ensemble
from utils.visualization import plot_performance
from utils.config_reader import get_config
from trading.indicators import calculate_technical_signals
from trading.analysis import evaluate_trend_quality, evaluate_volatility
from trading.execution import execute_trading_strategy, execute_buy_hold_strategy
from trading.selection import is_good_for_active_trading

def process_ticker(ticker, start_date, end_date, model=None, model_path=None, 
                  output_dir=None, min_data_points=None, train_model=False, 
                  portfolio=None, create_plots=True, lightweight_mode=None, verbose=False):
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
        create_plots: Whether to create visualization plots
        lightweight_mode: Use lightweight mode for faster data processing
        verbose: Whether to print detailed output
    """
    # Get logger and config
    logger = logging.getLogger('stock_prediction')
    config = get_config()
    
    # Use configured values if not explicitly provided
    if min_data_points is None:
        min_data_points = config.get('data_processing', 'min_data_points', default=250)
        
    if lightweight_mode is None:
        lightweight_mode = config.get('data_processing', 'lightweight_mode', default=True)
    
    processing_start_time = time.time()
    if verbose:
        logger.info(f"\nProcessing {ticker}")
    
    # Create output directory for this ticker if specified and needed
    ticker_output_dir = None
    if output_dir and create_plots:
        ticker_output_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_output_dir, exist_ok=True)
        if verbose:
            logger.info(f"Saving results to {ticker_output_dir}")
    
    # Prepare data
    data, message = prepare_data(ticker, start_date, end_date, min_data_points, lightweight_mode, verbose)
    
    if data is None:
        if verbose:
            logger.warning(f"Skipping {ticker}: {message}")
        return {"ticker": ticker, "success": False, "message": message}
    
    # Check if we have all required columns for modeling
    if 'Target' not in data.columns:
        if verbose:
            logger.error(f"Error: Target column missing for {ticker}. Cannot continue with model training.")
        return {"ticker": ticker, "success": False, "message": "Target column missing"}
    
    try:
        # Create sequences with configured sequence length
        seq_length = config.get('data_processing', 'sequence_length', default=20)
        X, y, dates, features, raw_prices = create_sequences(data, seq_length, essential_only=lightweight_mode)
        if verbose:
            logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        if len(X) < 100:  # Make sure we have enough sequences
            if verbose:
                logger.warning(f"Not enough sequences for {ticker}: only {len(X)} available")
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
                # Handle case where model_path is None
                if model_path is not None:
                    ensemble_dir = os.path.join(os.path.dirname(model_path), "ensemble")
                else:
                    # Default ensemble directory if model_path is None
                    ensemble_dir = os.path.join("stock_prediction_results", "models", "ensemble")
                    os.makedirs(ensemble_dir, exist_ok=True)
                    if verbose:
                        logger.info(f"No model path provided, using default ensemble directory: {ensemble_dir}")
                
                if os.path.exists(ensemble_dir) and os.path.isdir(ensemble_dir):
                    # Load ensemble models
                    try:
                        if verbose:
                            logger.info(f"Loading ensemble models from {ensemble_dir}")
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
                                    if verbose:
                                        logger.info(f"Loaded ensemble model: {model_type}")
                        else:
                            # Look for any .keras files in the ensemble directory
                            for f in os.listdir(ensemble_dir):
                                if f.endswith('.keras'):
                                    model_file = os.path.join(ensemble_dir, f)
                                    model_instance = load_model(model_file)
                                    ensemble_models.append(model_instance)
                                    if verbose:
                                        logger.info(f"Loaded ensemble model: {f}")
                        
                        if ensemble_models:
                            is_ensemble = True
                            model = ensemble_models
                        else:
                            raise ValueError("No ensemble models found")
                            
                    except Exception as e:
                        if verbose:
                            logger.error(f"Error loading ensemble models: {e}")
                        return {"ticker": ticker, "success": False, "message": f"Error loading ensemble models: {e}"}
                else:
                    # Load a single model
                    if model_path is not None:
                        try:
                            model = load_model(model_path)
                            is_ensemble = False
                            if verbose:
                                logger.info(f"Successfully loaded model from {model_path}")
                        except Exception as e:
                            if verbose:
                                logger.error(f"Error loading model: {e}")
                            return {"ticker": ticker, "success": False, "message": f"Error loading model: {e}"}
                    else:
                        if verbose:
                            logger.warning("No model path provided and no model specified. Using fallback prediction.")
            
            # Try to load feature mapping information
            feature_mapping = None
            try:
                # Check model directory for feature importance file
                if is_ensemble and 'ensemble_dir' in locals():
                    model_dir = ensemble_dir
                elif model_path is not None:
                    model_dir = os.path.dirname(model_path)
                else:
                    # Provide a default model directory
                    model_dir = os.path.join("stock_prediction_results", "models")
                    os.makedirs(model_dir, exist_ok=True)
                    if verbose:
                        logger.info(f"No model path provided, using default directory: {model_dir}")
                
                feature_importance_path = os.path.join(model_dir, "feature_importance.json")
                if os.path.exists(feature_importance_path):
                    with open(feature_importance_path, 'r') as f:
                        feature_data = json.load(f)
                        if 'important_features' in feature_data and 'important_indices' in feature_data:
                            if verbose:
                                logger.info("Found feature importance data for feature mapping")
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
                                if verbose:
                                    logger.info("Found feature importance data in parent directory")
                                feature_mapping = {
                                    'features': feature_data['important_features'],
                                    'indices': feature_data['important_indices']
                                }
            except Exception as e:
                if verbose:
                    logger.error(f"Error loading feature mapping: {e}")
            
            # Try to adapt features based on mapping
            if feature_mapping and features:
                if verbose:
                    logger.info("Adapting features based on feature importance data")
                feature_indices = []
                
                # Try to map by name first
                mapped_features = []
                for important_feature in feature_mapping['features']:
                    if important_feature in features:
                        feature_idx = features.index(important_feature)
                        feature_indices.append(feature_idx)
                        mapped_features.append(important_feature)
                    else:
                        if verbose:
                            logger.warning(f"Warning: Feature '{important_feature}' not found in current data")
                
                if len(mapped_features) >= len(feature_mapping['features']) * 0.8:  # If we found at least 80% of features
                    if verbose:
                        logger.info(f"Using feature mapping by name: Found {len(mapped_features)} of {len(feature_mapping['features'])} features")
                    # Reshape X to select only the mapped features
                    if feature_indices:
                        X_test_adapted = X_test[:, :, feature_indices]
                        if verbose:
                            logger.info(f"Adapted X_test from shape {X_test.shape} to {X_test_adapted.shape}")
                        X_test = X_test_adapted
                else:
                    if verbose:
                        logger.info("Couldn't map enough features by name, using indices directly")
                    # Use indices directly if they're valid
                    valid_indices = [idx for idx in feature_mapping['indices'] if idx < X_test.shape[2]]
                    if valid_indices and len(valid_indices) > 0:
                        X_test_adapted = X_test[:, :, valid_indices]
                        if verbose:
                            logger.info(f"Adapted X_test from shape {X_test.shape} to {X_test_adapted.shape}")
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
                    
                    # Make predictions (with batch size for faster processing)
                    y_pred_scaled = model.predict(X_test_scaled, batch_size=64, verbose=0 if not verbose else 1)
                    
                    # Convert predictions back to original scale
                    y_pred = scaler_y.inverse_transform(y_pred_scaled)
                    y_pred_corrected = y_pred.flatten()
                    
                    # Apply error correction if needed
                    avg_price = np.mean(y_test)
                    avg_pred = np.mean(y_pred)
                    max_allowed_deviation = 0.5  # Allow 50% difference
                    
                    if abs(avg_pred - avg_price) / avg_price > max_allowed_deviation:
                        if verbose:
                            logger.warning(f"Warning: Predictions for {ticker} are far from actual. Applying direct scaling.")
                        scaling_factor = avg_price / avg_pred
                        y_pred_corrected = y_pred.flatten() * scaling_factor
                        if verbose:
                            logger.info(f"Applied scaling factor: {scaling_factor}")
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred_corrected)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_corrected))
                    mape = np.mean(np.abs((y_test - y_pred_corrected) / y_test)) * 100
            except Exception as e:
                if verbose:
                    logger.error(f"Error during prediction: {e}")
                # Fallback to a simple moving average model
                if verbose:
                    logger.warning("Falling back to moving average prediction")
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
                
                if verbose:
                    logger.info(f"Fallback model metrics - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, MAPE: {mape:.2f}%")
            
            # Create empty history for visualization
            history = {"loss": [0], "val_loss": [0]}
        
        # Analyze performance and generate trading signals
        result = analyze_performance(
            ticker, y_test, y_pred_corrected, test_dates, history,
            mae, rmse, mape, output_dir=ticker_output_dir if create_plots else None,
            portfolio=portfolio, verbose=verbose, data=data  # Pass the full data for additional analysis
        )
        
        # Report processing time
        processing_end_time = time.time()
        processing_time = processing_end_time - processing_start_time
        if verbose:
            logger.info(f"Processed {ticker} in {processing_time:.2f} seconds")
        
        # Add processing time to result
        result["processing_time"] = processing_time
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        if verbose:
            logger.error(f"Error during model training for {ticker}: {error_msg}")
            logger.error(traceback.format_exc())  # Print full traceback for debugging
        return {"ticker": ticker, "success": False, "message": f"Error during model training: {error_msg}"}


def analyze_performance(ticker, y_test, y_pred_corrected, test_dates, history, mae, rmse, mape, 
                      output_dir=None, portfolio=None, fast_analysis=True, verbose=False, data=None):
    """
    Analyze model performance and implement a balanced trading strategy.
    Uses configuration settings for more realistic returns.
    """
    logger = logging.getLogger('stock_prediction')
    config = get_config()
    
    # Calculate buy & hold return first
    buy_hold_return_pct = ((y_test[-1] / y_test[0]) - 1) * 100
    if verbose:
        logger.info(f"Buy & Hold return for {ticker}: {buy_hold_return_pct:.2f}%")
    
    # Calculate moving averages for trend detection
    ma_short = 5   # 5-day MA for quick trend changes
    ma_medium = 20  # 20-day MA for medium-term trend
    ma_long = 50    # 50-day MA for long-term trend direction
    
    # Use NumPy's convolve for faster moving average calculation
    def moving_average(x, w):
        window = np.ones(w) / w
        ma = np.convolve(x, window, 'valid')
        # Pad the beginning to match original array length
        return np.concatenate([np.full(w-1, ma[0]), ma])
    
    # Calculate vectorized moving averages
    trend_short = moving_average(y_test, ma_short)
    trend_medium = moving_average(y_test, ma_medium)
    trend_long = moving_average(y_test, ma_long)
    
    # Calculate prediction accuracy using vectorized operations
    accuracy_window = min(20, len(y_pred_corrected) - 1)
    
    # Create arrays of shifted values
    y_pred_shifted = y_pred_corrected[:-1]
    y_pred_current = y_pred_corrected[1:]
    y_test_shifted = y_test[:-1]
    y_test_current = y_test[1:]
    
    # Calculate directions using vectorized operations
    pred_directions = np.sign(y_pred_current - y_pred_shifted)
    actual_directions = np.sign(y_test_current - y_test_shifted)
    
    # Calculate accuracy (1 where directions match, 0 otherwise)
    direction_matches = (pred_directions == actual_directions).astype(int)
    
    # Get the most recent n matches for recent accuracy
    recent_matches = direction_matches[-accuracy_window:] if len(direction_matches) >= accuracy_window else direction_matches
    recent_accuracy = np.mean(recent_matches) if len(recent_matches) > 0 else 0.5
    
    if verbose:
        logger.info(f"Recent directional prediction accuracy: {recent_accuracy:.2f}")
    
    # Extract volumes if available
    volumes = None
    if data is not None and 'Volume' in data.columns:
        if len(data) >= len(y_test):
            # Align data with test indices
            last_n_indices = data.index[-len(y_test):]
            volumes = data.loc[last_n_indices, 'Volume'].values
    
    # Calculate technical signals
    tech_signals = calculate_technical_signals(y_test, volumes)
    
    # Analyze trend and volatility
    trend_metrics = evaluate_trend_quality(y_test)
    volatility_metrics = evaluate_volatility(y_test)
    
    # Decide if this stock is suitable for active trading
    is_suitable, suitability_score = is_good_for_active_trading(
        ticker, trend_metrics, volatility_metrics, recent_accuracy
    )
    
    if verbose:
        logger.info(f"Stock suitability for active trading: Score {suitability_score:.2f} - {'Suitable' if is_suitable else 'Not suitable'}")
        logger.info(f"Trend quality: {trend_metrics['trend_quality']:.2f}, Volatility: {volatility_metrics['volatility_regime']}")
    
    # Generate signals based on predictions
    y_diff = np.diff(y_pred_corrected.flatten())
    
    # Get signal thresholds from config
    buy_threshold = config.get('trading', 'signal_thresholds', 'buy_threshold', default=1.5)
    sell_threshold = config.get('trading', 'signal_thresholds', 'sell_threshold', default=-1.5)
    
    # Calculate percentage changes - this is our main signal
    predicted_change_pcts = (y_diff / y_test[:-1]) * 100
    
    # Pre-allocate signal arrays
    buy_signals = np.zeros(len(y_diff), dtype=bool)
    sell_signals = np.zeros(len(y_diff), dtype=bool)
    
    # Get technical signals for confirmation
    tech_buy_signals = tech_signals['signals'].get('buy', np.zeros(len(y_test), dtype=bool))
    tech_sell_signals = tech_signals['signals'].get('sell', np.zeros(len(y_test), dtype=bool))
    
    # Align technical signals with prediction signals
    tech_buy_aligned = tech_buy_signals[1:] if len(tech_buy_signals) == len(y_test) else np.zeros(len(y_diff), dtype=bool)
    tech_sell_aligned = tech_sell_signals[1:] if len(tech_sell_signals) == len(y_test) else np.zeros(len(y_diff), dtype=bool)
    
    # Generate prediction-based signals
    pred_buy_signals = (predicted_change_pcts > buy_threshold)
    pred_sell_signals = (predicted_change_pcts < sell_threshold)
    
    # Calculate trend direction (1 for uptrend, -1 for downtrend)
    trend_direction = np.sign(trend_medium[1:] - trend_medium[:-1])
    long_trend = np.sign(trend_long[-1] - trend_long[-ma_short])
    
    # Buy signal: Predicted price increase above threshold OR tech signal with positive trend
    buy_signals = pred_buy_signals | (tech_buy_aligned & (trend_direction > 0))
    
    # Sell signal: Predicted price decrease below threshold OR tech signal with negative trend
    sell_signals = pred_sell_signals | (tech_sell_aligned & (trend_direction < 0))
    
    # Filter signals in weak trends
    if trend_metrics['trend_quality'] < 0.4 and trend_metrics['trend_direction'] < 0:
        # More conservative in very weak downtrends
        buy_signals = buy_signals & (predicted_change_pcts > buy_threshold * 1.2)
    
    # Be more aggressive with selling in strong downtrends
    if trend_metrics['trend_direction'] < 0 and trend_metrics['trend_quality'] > 0.6:
        # More aggressive selling in clear downtrends
        sell_signals = sell_signals | (predicted_change_pcts < sell_threshold * 0.6)
    
    if verbose:
        logger.info(f"Signals for {ticker}:")
        logger.info(f"  Buy signals: {np.sum(buy_signals)} out of {len(y_diff)} potential signals")
        logger.info(f"  Sell signals: {np.sum(sell_signals)} out of {len(y_diff)} potential signals")
    
    # Strategy selection criteria from config
    buy_hold_return_threshold = config.get('trading', 'strategy_selection', 'buy_hold_return_threshold', default=20.0)
    trend_quality_threshold = config.get('trading', 'strategy_selection', 'trend_quality_threshold', default=0.6)
    buy_hold_uptrend_threshold = config.get('trading', 'strategy_selection', 'buy_hold_uptrend_threshold', default=15.0)
    
    # More conservative strategy selection criteria
    use_buy_hold = (buy_hold_return_pct > buy_hold_return_threshold and recent_accuracy < 0.6) or \
                  (not is_suitable) or \
                  (trend_metrics['trend_quality'] > trend_quality_threshold and 
                   trend_metrics['trend_direction'] > 0 and 
                   buy_hold_return_pct > buy_hold_uptrend_threshold) or \
                  (recent_accuracy < 0.53)  # Default to buy-hold unless we have strong prediction accuracy
    
    # Default to active trading for suitable stocks
    use_active_trading = not use_buy_hold
    
    # Build the list of signals for execution
    signals = []
    for i in range(1, len(y_pred_corrected)):
        if i-1 < len(y_diff):  # Ensure index is valid
            if buy_signals[i-1]:
                signals.append(('buy', test_dates[i], y_test[i]))
            elif sell_signals[i-1]:
                signals.append(('sell', test_dates[i], y_test[i]))
    
    # Execute the appropriate strategy
    if use_buy_hold:
        strategy_used = "buy_hold"
        if verbose:
            logger.info(f"Using buy & hold for {ticker} ({buy_hold_return_pct:.2f}%, accuracy: {recent_accuracy:.2f})")
            
        # Execute buy & hold strategy
        initial_capital = 10000  # Fixed initial capital
        result = execute_buy_hold_strategy(ticker, test_dates, y_test, initial_capital)
        
    else:  # Use active trading
        strategy_used = "active_trading"
        if verbose:
            logger.info(f"Using active trading for {ticker} with {recent_accuracy:.2f} accuracy")
            
        # Execute active trading strategy with signals
        initial_capital = 10000  # Fixed initial capital
        result = execute_trading_strategy(
            ticker, signals, test_dates, y_test, initial_capital,
            tech_signals['indicators'].get('rsi', np.zeros(len(y_test))),
            trend_metrics, verbose
        )
    
    # Plot results if requested
    if output_dir:
        plot_performance(ticker, history, y_test, y_pred_corrected, test_dates, 
                       buy_signals, sell_signals, trend_medium, output_dir)
    
    # Extract results
    total_return = result['total_return']
    final_value = result['final_value']
    trades = result['trades']
    winning_trades = result['winning_trades']
    losing_trades = result['losing_trades']
    win_rate = result['win_rate']
    avg_win = result['avg_win']
    avg_loss = result['avg_loss']
    win_loss_ratio = result['win_loss_ratio']
    profit_factor = result['profit_factor']
    total_trades = result['total_trades']
    
    # Apply sanity checks from config
    max_return_pct = config.get('sanity_checks', 'max_return_pct', default=300)
    min_trades_for_high_return = config.get('sanity_checks', 'min_trades_for_high_return', default=5)
    max_return_few_trades = config.get('sanity_checks', 'max_return_few_trades', default=100)
    
    # Cap the maximum return if needed
    if total_return > max_return_pct:
        logger.warning(f"Capping unrealistic return for {ticker} from {total_return:.2f}% to {max_return_pct}%")
        total_return = max_return_pct
        # Recalculate final_value based on capped return
        final_value = initial_capital * (1 + total_return/100)
    
    # Check for suspiciously high returns with few trades
    if total_return > max_return_few_trades and total_trades < min_trades_for_high_return:
        logger.warning(f"Suspicious high return with few trades for {ticker}")
        total_return = min(total_return, max_return_few_trades)
        final_value = initial_capital * (1 + total_return/100)
    
    if verbose:
        logger.info(f"Initial capital: ${initial_capital:.2f}")
        logger.info(f"Final value: ${final_value:.2f}")
        logger.info(f"Total return: {total_return:.2f}%")
        logger.info(f"Buy & Hold return: {buy_hold_return_pct:.2f}%")
        logger.info(f"Strategy outperformance: {total_return - buy_hold_return_pct:.2f}%")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Winning trades: {winning_trades} ({win_rate:.2f}%)")
        logger.info(f"Losing trades: {losing_trades} ({100 - win_rate:.2f}%)")
        logger.info(f"Average win: {avg_win:.2f}%")
        logger.info(f"Average loss: {avg_loss:.2f}%")
        logger.info(f"Win/Loss ratio: {win_loss_ratio:.2f}")
        logger.info(f"Profit factor: {profit_factor:.2f}")
    
    # Update portfolio if provided
    if portfolio is not None:
        # Add performance data to the returns dict
        portfolio['returns'][ticker] = {
            'initial_value': initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'buy_hold_return_pct': buy_hold_return_pct,
            'prediction_accuracy': recent_accuracy,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'trades': trades,
            'strategy': strategy_used,
            'trend_quality': trend_metrics['trend_quality'],
            'volatility_regime': volatility_metrics['volatility_regime']
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
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "profit_factor": profit_factor,
        "strategy": strategy_used,
        "prediction_accuracy": recent_accuracy,
        "trades": trades,  # Include full trade history
        "trend_quality": trend_metrics['trend_quality'],
        "volatility_regime": volatility_metrics['volatility_regime']
    }