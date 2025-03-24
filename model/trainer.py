"""
Stock Price Prediction System - Model Trainer
-------------------------------------------
This file handles model training, evaluation, and predictions.
"""

import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from data.processor import prepare_data, create_sequences, combine_data_from_tickers
from model.builder import build_model
from model.ensemble import train_ensemble_models, predict_with_ensemble
from utils.feature_analysis import analyze_feature_importance, get_optimal_feature_subset

def train_evaluate_model(X_train, X_test, y_train, y_test, feature_names=None, model_path=None, load_saved=False, model_type='standard'):
    """
    Train and evaluate a model for stock price prediction
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing targets
        feature_names: List of feature names (for feature importance analysis)
        model_path: Path to save or load model
        load_saved: Whether to load a saved model if it exists
        model_type: Type of model architecture to use
        
    Returns:
        model: Trained model
        history: Training history
        mae, rmse, mape: Performance metrics
        y_pred_corrected: Corrected predictions
    """
    # Perform feature importance analysis if feature names are provided
    if feature_names is not None and len(feature_names) > 0:
        print("\nAnalyzing feature importance...")
        
        # Get the optimal feature subset
        feature_subset = get_optimal_feature_subset(
            X_train, y_train, feature_names,
            threshold=0.85,  # Use features up to 85% cumulative importance
            min_features=10  # Always include at least 10 features
        )
        
        # Log the selected features
        print("Selected features:")
        for i, name in enumerate(feature_subset['selected_names']):
            print(f"  {i+1}. {name} (Importance: {feature_subset['importance_scores'][i]:.4f})")
        
        # Filter X to only include important features
        important_indices = feature_subset['selected_indices']
        X_train = X_train[:, :, important_indices]
        X_test = X_test[:, :, important_indices]
        print(f"Reduced feature dimensionality from {len(feature_names)} to {len(important_indices)} features")
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
    
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
    X_test_scaled = scaler_X.transform(X_test_reshaped)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # Scale target values
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # Build model or load from file
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    
    if load_saved and model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = load_model(model_path)
        history = None  # No training history available for loaded models
    else:
        model = build_model(input_shape, model_type)
        model.summary()
        
        # Enhanced callbacks for better training
        callbacks = [
            EarlyStopping(patience=30, restore_best_weights=True, min_delta=0.0001),
            ReduceLROnPlateau(factor=0.5, patience=15, min_lr=0.00001, min_delta=0.0001)
        ]
        
        # Add model checkpoint if a path is provided
        if model_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'))
        
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=300,  # Increased max epochs for better convergence
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model if path is provided and checkpoint wasn't used
        if model_path and not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
            print(f"Saving model to {model_path}")
            model.save(model_path)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Inverse transform predictions back to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate prediction error patterns
    y_error = y_test - y_pred.flatten()
    error_mean = np.mean(y_error)
    error_std = np.std(y_error)
    
    print(f"Prediction error analysis:")
    print(f"  Mean error: ${error_mean:.2f}")
    print(f"  Error std: ${error_std:.2f}")
    
    # Apply error correction: Use a rolling-window average of recent errors
    window_size = 5
    rolling_errors = []
    
    for i in range(len(y_pred)):
        if i < window_size:
            # For the first few predictions, use what we have
            correction = np.mean(y_error[:max(1, i)])
        else:
            # For later predictions, use recent error pattern
            correction = np.mean(y_error[i-window_size:i])
        rolling_errors.append(correction)
    
    # Apply the adaptive correction
    y_pred_corrected = y_pred.flatten() + np.array(rolling_errors)
    
    # Calculate metrics before and after correction
    mae_before = mean_absolute_error(y_test, y_pred)
    rmse_before = np.sqrt(mean_squared_error(y_test, y_pred))
    mape_before = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
    
    mae_after = mean_absolute_error(y_test, y_pred_corrected)
    rmse_after = np.sqrt(mean_squared_error(y_test, y_pred_corrected))
    mape_after = np.mean(np.abs((y_test - y_pred_corrected) / y_test)) * 100
    
    print("Before error correction:")
    print(f"MAE: ${mae_before:.2f}")
    print(f"RMSE: ${rmse_before:.2f}")
    print(f"MAPE: {mape_before:.2f}%")
    
    print("\nAfter error correction:")
    print(f"MAE: ${mae_after:.2f}")
    print(f"RMSE: ${rmse_after:.2f}")
    print(f"MAPE: {mape_after:.2f}%")
    
    # Return the corrected predictions
    return model, history, mae_after, rmse_after, mape_after, y_pred_corrected

def advanced_train_evaluate_model(X_train, X_test, y_train, y_test, feature_names=None, test_dates=None, ticker=None, model_path=None):
    """
    Advanced training and evaluation with hyperparameter optimization and trading metrics
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing targets
        feature_names: List of feature names (for feature importance analysis)
        test_dates: Dates for the test set (for trading simulation)
        ticker: Stock ticker symbol
        model_path: Path to save the model
        
    Returns:
        model: Trained model
        performance: Dictionary of performance metrics
    """
    print(f"Starting advanced model training for {ticker if ticker else 'stock'}")
    
    # Perform feature importance analysis if feature names are provided
    if feature_names is not None and len(feature_names) > 0:
        print("\nAnalyzing feature importance...")
        
        # Get the optimal feature subset
        feature_subset = get_optimal_feature_subset(
            X_train, y_train, feature_names,
            threshold=0.85,  # Use features up to 85% cumulative importance
            min_features=10  # Always include at least 10 features
        )
        
        # Log the selected features
        print("Selected features:")
        for i, name in enumerate(feature_subset['selected_names']):
            print(f"  {i+1}. {name} (Importance: {feature_subset['importance_scores'][i]:.4f})")
        
        # Filter X to only include important features
        important_indices = feature_subset['selected_indices']
        X_train = X_train[:, :, important_indices]
        X_test = X_test[:, :, important_indices]
        print(f"Reduced feature dimensionality from {len(feature_names)} to {len(important_indices)} features")
        
        # Save reduced feature names
        reduced_feature_names = [feature_names[i] for i in important_indices]
    else:
        reduced_feature_names = feature_names
        feature_subset = None
    
    # Split training data to create a validation set
    val_size = int(len(X_train) * 0.2)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    # Optimize hyperparameters
    best_model, best_config = optimize_model_hyperparameters(X_train, y_train, X_val, y_val)
    
    # Scale the full dataset for final training
    scaler_X = StandardScaler()
    X_train_full = np.concatenate((X_train, X_val))
    y_train_full = np.concatenate((y_train, y_val))
    
    X_train_reshaped = X_train_full.reshape(-1, X_train_full.shape[2])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
    
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train_full.shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_full.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # Define callbacks for final training
    callbacks = [
        EarlyStopping(patience=30, restore_best_weights=True, min_delta=0.0001),
        ReduceLROnPlateau(factor=0.5, patience=15, min_lr=0.00001, min_delta=0.0001)
    ]
    
    # Add ModelCheckpoint if path is provided
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        callbacks.append(ModelCheckpoint(model_path, save_best_weights=True, monitor='val_loss'))
    
    # Train the final model on full training data
    history = best_model.fit(
        X_train_scaled, y_train_scaled,
        epochs=300,
        batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )
    
    # Generate predictions
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    # Apply error correction
    # Calculate error patterns
    y_error = y_test - y_pred
    error_mean = np.mean(y_error)
    error_std = np.std(y_error)
    
    # Use a rolling window error correction
    window_size = 5
    rolling_errors = []
    
    for i in range(len(y_pred)):
        if i < window_size:
            # For the first few predictions, use what we have
            correction = np.mean(y_error[:max(1, i)])
        else:
            # For later predictions, use recent error pattern
            correction = np.mean(y_error[i-window_size:i])
        rolling_errors.append(correction)
    
    # Apply the corrections
    y_pred_corrected = y_pred + np.array(rolling_errors)
    
    # Calculate standard metrics
    mae = mean_absolute_error(y_test, y_pred_corrected)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_corrected))
    mape = np.mean(np.abs((y_test - y_pred_corrected) / y_test)) * 100
    
    print(f"\nModel Performance Metrics:")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # If test_dates are provided, evaluate trading performance
    trading_metrics = None
    if test_dates is not None:
        from utils.trading_evaluation import evaluate_model_trading_performance
        trading_metrics = evaluate_model_trading_performance(
            best_model, X_test_scaled, y_test, test_dates, ticker
        )
    
    # Save the model if path provided
    if model_path and not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
        print(f"Saving model to {model_path}")
        best_model.save(model_path)
    
    # Combine performance metrics
    performance = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'error_mean': error_mean,
        'error_std': error_std,
        'hyperparameters': best_config,
        'trading_metrics': trading_metrics,
        'feature_subset': feature_subset,
        'reduced_feature_names': reduced_feature_names
    }
    
    return best_model, performance, history, y_pred_corrected

def optimize_model_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Optimize model hyperparameters for stock prediction
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Best model found and configuration details
    """
    print("Optimizing model hyperparameters...")
    
    # Scale the data
    scaler_X = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[2])
    
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_reshaped).reshape(X_val.shape)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
    
    # Define hyperparameter options to try
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    
    # Parameters to try for LSTM layers
    lstm_units_options = [
        [64, 32],
        [128, 64],
        [96, 64, 32]
    ]
    
    # Dropout rates to try
    dropout_options = [0.2, 0.3, 0.4]
    
    # Learning rates to try
    lr_options = [0.001, 0.0005, 0.0002]
    
    best_val_loss = float('inf')
    best_model = None
    best_config = None
    
    # Try different combinations
    for lstm_units in lstm_units_options:
        for dropout_rate in dropout_options:
            for lr in lr_options:
                print(f"\nTrying: LSTM units={lstm_units}, dropout={dropout_rate}, learning_rate={lr}")
                
                # Create model
                model = Sequential()
                
                # Add LSTM layers
                for i, units in enumerate(lstm_units):
                    return_sequences = i < len(lstm_units) - 1
                    
                    if i == 0:
                        model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
                    else:
                        model.add(LSTM(units, return_sequences=return_sequences))
                    
                    model.add(Dropout(dropout_rate))
                
                # Add Dense layers
                model.add(Dense(32, activation='relu'))
                model.add(Dense(1))
                
                # Compile model
                model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
                
                # Train model
                early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
                history = model.fit(
                    X_train_scaled, y_train_scaled,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val_scaled, y_val_scaled),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Evaluate model
                val_loss = history.history['val_loss'][-1]
                print(f"Validation loss: {val_loss:.6f}")
                
                # Check if this is the best model so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    best_config = {
                        'lstm_units': lstm_units,
                        'dropout_rate': dropout_rate,
                        'learning_rate': lr,
                        'val_loss': val_loss
                    }
                    print(f"New best model found!")
    
    print("\nBest model configuration:")
    print(f"LSTM units: {best_config['lstm_units']}")
    print(f"Dropout rate: {best_config['dropout_rate']}")
    print(f"Learning rate: {best_config['learning_rate']}")
    print(f"Validation loss: {best_config['val_loss']:.6f}")
    
    return best_model, best_config

def train_multi_stock_model(tickers, start_date, end_date, min_data_points=250, model_save_path=None, use_ensemble=True, verbose=False):
    """
    Train a model or ensemble of models on data from multiple stocks
    
    Args:
        tickers: List of ticker symbols to train on
        start_date, end_date: Date range for training data
        min_data_points: Minimum required data points per ticker
        model_save_path: Path to save the model
        use_ensemble: Whether to use ensemble of models or single model
        verbose: Whether to print detailed output
        
    Returns:
        Trained model(s) and ticker metadata
    """
    print("Training model on multiple stocks:", tickers[:5], "..." if len(tickers) > 5 else "")
    
    # Collect data from all tickers
    ticker_data = {}
    
    for ticker in tickers:
        try:
            data, message = prepare_data(ticker, start_date, end_date, min_data_points)
            if data is not None:
                ticker_data[ticker] = data
                print(f"Successfully processed data for {ticker}")
            else:
                print(f"Skipping {ticker}: {message}")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
            
        time.sleep(0.1)  # Short delay to avoid API rate limits
    
    if not ticker_data:
        print("No valid ticker data found. Cannot train model.")
        return None, None
    
    # Combine data from all tickers
    X_combined, y_combined, ticker_features = combine_data_from_tickers(ticker_data)
    
    if X_combined is None or y_combined is None:
        print("Failed to create combined dataset")
        return None, None
    
    print(f"Combined dataset shape: X={X_combined.shape}, y={y_combined.shape}")
    
    # Create validation set (10% of data)
    indices = np.random.permutation(len(X_combined))
    val_size = int(len(X_combined) * 0.1)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    X_train, X_val = X_combined[train_indices], X_combined[val_indices]
    y_train, y_val = y_combined[train_indices], y_combined[val_indices]
    
    # Perform feature importance analysis to get optimal feature subset
    # Extract all the feature names from one ticker
    sample_ticker = list(ticker_features.keys())[0]
    all_features = ticker_features[sample_ticker]
    
    # Run feature importance analysis
    print("\nAnalyzing feature importance across all stocks...")
    feature_subset = get_optimal_feature_subset(
        X_train, y_train, all_features,
        threshold=0.85,  # Use features up to 85% cumulative importance
        min_features=10  # Always include at least 10 features
    )
    
    # Log the selected features
    print("Selected features:")
    for i, name in enumerate(feature_subset['selected_names']):
        print(f"  {i+1}. {name} (Importance: {feature_subset['importance_scores'][i]:.4f})")
    
    # Filter X to only include important features
    important_indices = feature_subset['selected_indices']
    X_train = X_train[:, :, important_indices]
    X_val = X_val[:, :, important_indices]
    print(f"Reduced feature dimensionality from {len(all_features)} to {len(important_indices)} features")
    
    # Store reduced feature names for reference
    reduced_features = [all_features[i] for i in important_indices]
    
    if use_ensemble:
        # Create ensemble models directory
        model_dir = os.path.dirname(model_save_path)
        ensemble_dir = os.path.join(model_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # Train ensemble of 3 different model architectures
        models, _, _, _, _, _ = train_ensemble_models(
            X_train, X_val, y_train, y_val, 
            model_dir=ensemble_dir, 
            n_models=3,
            verbose=verbose  # Pass verbose parameter to ensemble training
        )
        
        # Store ticker metadata for reference
        ticker_metadata = {
            ticker: {
                "features": ticker_features.get(ticker, []),
                "important_features": reduced_features,
                "last_price": ticker_data[ticker]["Close"].iloc[-1] if ticker in ticker_data else None,
                "min_price": ticker_data[ticker]["Close"].min() if ticker in ticker_data else None,
                "max_price": ticker_data[ticker]["Close"].max() if ticker in ticker_data else None,
                "price_range": ticker_data[ticker]["Close"].max() - ticker_data[ticker]["Close"].min() if ticker in ticker_data else None
            } for ticker in ticker_data.keys()
        }
        
        # Save feature importance information
        import json
        with open(os.path.join(model_dir, "feature_importance.json"), 'w') as f:
            json.dump({
                'important_features': reduced_features,
                'important_indices': feature_subset['selected_indices'].tolist(),
                'importance_scores': feature_subset['importance_scores'].tolist(),
            }, f, indent=4)
        
        return models, ticker_metadata
    else:
        # Train a single model with hyperparameter optimization
        best_model, best_config = optimize_model_hyperparameters(X_train, y_train, X_val, y_val)
        
        # Save the model
        if model_save_path:
            print(f"Saving model to {model_save_path}")
            best_model.save(model_save_path)
            
            # Save model configuration
            import json
            config_path = os.path.join(os.path.dirname(model_save_path), "model_config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    'hyperparameters': best_config,
                    'important_features': reduced_features,
                    'important_indices': feature_subset['selected_indices'].tolist(),
                }, f, indent=4)
        
        # Store ticker metadata for reference
        ticker_metadata = {
            ticker: {
                "features": ticker_features.get(ticker, []),
                "important_features": reduced_features,
                "last_price": ticker_data[ticker]["Close"].iloc[-1] if ticker in ticker_data else None,
                "min_price": ticker_data[ticker]["Close"].min() if ticker in ticker_data else None,
                "max_price": ticker_data[ticker]["Close"].max() if ticker in ticker_data else None,
                "price_range": ticker_data[ticker]["Close"].max() - ticker_data[ticker]["Close"].min() if ticker in ticker_data else None
            } for ticker in ticker_data.keys()
        }
        
        return best_model, ticker_metadata