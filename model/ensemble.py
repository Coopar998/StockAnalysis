"""
Stock Price Prediction System - Ensemble Models
---------------------------------------------
This file handles training and predictions with ensemble models.
"""

import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.builder import build_model

def train_ensemble_models(X_train, X_test, y_train, y_test, model_dir=None, n_models=3):
    """
    Train an ensemble of models with different architectures
    
    Args:
        X_train, X_test: Training and testing feature data
        y_train, y_test: Training and testing target data
        model_dir: Directory to save models
        n_models: Number of models to train for the ensemble
        
    Returns:
        List of trained models, ensemble predictions, and performance metrics
    """
    print(f"\nTraining ensemble of {n_models} models with different architectures")
    
    # Define model types to use in the ensemble
    model_types = ['standard', 'deep', 'bidirectional', 'hybrid', 'attention']
    
    if n_models > len(model_types):
        n_models = len(model_types)
        print(f"Limiting ensemble to {n_models} models")
    
    # Train each model and collect predictions
    models = []
    all_predictions = []
    all_metrics = []
    
    for i in range(n_models):
        model_type = model_types[i]
        print(f"\nTraining model {i+1}/{n_models}: {model_type}")
        
        # Define model path if directory is provided
        model_path = None
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"ensemble_{model_type}.keras")
        
        # Scale features for this model
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
        
        # Build and train model
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        model = build_model(input_shape, model_type)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=25, restore_best_weights=True)
        ]
        
        if model_path:
            callbacks.append(
                ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
            )
        
        # Train model
        model.fit(
            X_train_scaled, y_train_scaled,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Store results
        models.append(model)
        all_predictions.append(y_pred)
        all_metrics.append({
            'type': model_type,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        })
        
        # Print model performance
        print(f"Model {i+1} ({model_type}) performance:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        # Save input shape information with the model
        if model_dir:
            metadata_path = os.path.join(model_dir, f"ensemble_{model_type}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    'input_shape': input_shape,
                    'feature_dim': X_train.shape[2],
                    'sequence_length': X_train.shape[1],
                }, f, indent=4)
    
    # Create ensemble prediction (weighted average based on inverse RMSE)
    weights = [1/metrics['rmse'] for metrics in all_metrics]
    weights_sum = sum(weights)
    normalized_weights = [w/weights_sum for w in weights]
    
    print("\nEnsemble weights based on inverse RMSE:")
    for i, (w, metrics) in enumerate(zip(normalized_weights, all_metrics)):
        print(f"  Model {i+1} ({metrics['type']}): {w:.4f}")
    
    # Apply weighted average for ensemble prediction
    weighted_preds = np.zeros_like(all_predictions[0])
    for pred, weight in zip(all_predictions, normalized_weights):
        weighted_preds += pred * weight
        
    # Calculate ensemble metrics
    ens_mae = mean_absolute_error(y_test, weighted_preds)
    ens_rmse = np.sqrt(mean_squared_error(y_test, weighted_preds))
    ens_mape = np.mean(np.abs((y_test - weighted_preds) / y_test)) * 100
    
    print("\nEnsemble model performance:")
    print(f"  MAE: ${ens_mae:.2f}")
    print(f"  RMSE: ${ens_rmse:.2f}")
    print(f"  MAPE: {ens_mape:.2f}%")
    
    # Save ensemble weights if model directory is provided
    if model_dir:
        ensemble_config = {
            'types': [m['type'] for m in all_metrics],
            'weights': normalized_weights,
            'performance': {
                'mae': ens_mae,
                'rmse': ens_rmse,
                'mape': ens_mape
            },
            'feature_dim': X_train.shape[2]
        }
        
        with open(os.path.join(model_dir, 'ensemble_config.json'), 'w') as f:
            json.dump(ensemble_config, f, indent=4)
    
    return models, weighted_preds, all_predictions, ens_mae, ens_rmse, ens_mape

def predict_with_ensemble(models, X_test, y_test, model_dir=None):
    """
    Make predictions using an ensemble of models
    
    Args:
        models: List of trained models
        X_test: Test features
        y_test: Test targets for metrics calculation
        model_dir: Directory where models were saved
        
    Returns:
        Ensemble predictions and metrics
    """
    # Load ensemble configuration if available
    ensemble_config = None
    expected_feature_dim = None
    if model_dir and os.path.exists(os.path.join(model_dir, 'ensemble_config.json')):
        try:
            with open(os.path.join(model_dir, 'ensemble_config.json'), 'r') as f:
                ensemble_config = json.load(f)
                
                # Check if feature dimensions are stored in config
                if 'feature_dim' in ensemble_config:
                    expected_feature_dim = ensemble_config['feature_dim']
                    print(f"Expected feature dimension from model: {expected_feature_dim}")
                    print(f"Actual feature dimension in data: {X_test.shape[2]}")
        except Exception as e:
            print(f"Warning: Error loading ensemble config: {e}")
    
    # Check for feature dimension mismatch
    if expected_feature_dim is not None and expected_feature_dim != X_test.shape[2]:
        print(f"WARNING: Feature dimension mismatch. Model expects {expected_feature_dim} features, but {X_test.shape[2]} were provided.")
        print("Attempting to adapt feature dimensions...")
        
        # Try to load feature importance data to map features correctly
        feature_importance_path = os.path.join(os.path.dirname(model_dir), "feature_importance.json")
        if os.path.exists(feature_importance_path):
            try:
                with open(feature_importance_path, 'r') as f:
                    feature_data = json.load(f)
                    important_indices = feature_data.get('important_indices', [])
                    
                    # Check if we can use these indices to select the right features
                    if len(important_indices) == expected_feature_dim and max(important_indices) < X_test.shape[2]:
                        print(f"Using feature importance data to select the correct {expected_feature_dim} features.")
                        X_test = X_test[:, :, important_indices]
                    else:
                        print("Feature importance data doesn't match expected dimensions.")
            except Exception as e:
                print(f"Error loading feature importance data: {e}")
        
        # If we couldn't fix it with feature importance data, try a simple approach
        if X_test.shape[2] != expected_feature_dim:
            if X_test.shape[2] > expected_feature_dim:
                # If we have too many features, use the first expected_feature_dim features
                print(f"Taking the first {expected_feature_dim} features from the {X_test.shape[2]} available.")
                X_test = X_test[:, :, :expected_feature_dim]
            else:
                # If we have too few features, we can't proceed
                raise ValueError(f"Cannot adapt feature dimensions. Model requires {expected_feature_dim} features, but only {X_test.shape[2]} are available.")
    
    # Scale features
    scaler_X = StandardScaler()
    X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
    X_test_scaled = scaler_X.fit_transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # Get predictions from each model
    all_predictions = []
    
    for i, model in enumerate(models):
        try:
            # Check if this model has metadata with input shape info
            if model_dir:
                model_type = ensemble_config['types'][i] if ensemble_config and 'types' in ensemble_config else f"model_{i}"
                metadata_path = os.path.join(model_dir, f"ensemble_{model_type}_metadata.json")
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        model_feature_dim = metadata.get('feature_dim')
                        
                        # If model requires different features than we have available
                        if model_feature_dim and model_feature_dim != X_test_scaled.shape[2]:
                            if X_test_scaled.shape[2] > model_feature_dim:
                                # Take only the first n features needed by this model
                                print(f"Model {i} requires {model_feature_dim} features. Using first {model_feature_dim} of {X_test_scaled.shape[2]} features.")
                                X_test_model = X_test_scaled[:, :, :model_feature_dim]
                            else:
                                # Skip this model if we can't provide enough features
                                print(f"Skipping model {i} which requires {model_feature_dim} features.")
                                continue
                        else:
                            X_test_model = X_test_scaled
                else:
                    X_test_model = X_test_scaled
            else:
                X_test_model = X_test_scaled
            
            # Make prediction with this model
            y_pred_scaled = model.predict(X_test_model)
            
            # Create simple standardizer for each model's predictions
            scaler_y = StandardScaler()
            scaler_y.fit(y_test.reshape(-1, 1))
            
            # Transform predictions back to original scale
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
            all_predictions.append(y_pred)
            
        except Exception as e:
            print(f"Error with model {i}: {e}")
            # Skip this model
    
    if not all_predictions:
        raise ValueError("No models were able to make predictions. Check feature dimensions.")
    
    # Use ensemble weights if available, otherwise equal weighting
    if ensemble_config and 'weights' in ensemble_config:
        weights = ensemble_config['weights']
        # Adjust weights if we had to skip some models
        if len(weights) != len(all_predictions):
            weights = weights[:len(all_predictions)]
            weights_sum = sum(weights)
            if weights_sum > 0:
                weights = [w / weights_sum for w in weights]
            else:
                weights = [1/len(all_predictions)] * len(all_predictions)
        print(f"Using saved ensemble weights: {weights}")
    else:
        weights = [1/len(all_predictions)] * len(all_predictions)
        print(f"Using equal weights for ensemble: {weights}")
    
    # Apply weighted average
    weighted_preds = np.zeros_like(all_predictions[0])
    for pred, weight in zip(all_predictions, weights):
        weighted_preds += pred * weight
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, weighted_preds)
    rmse = np.sqrt(mean_squared_error(y_test, weighted_preds))
    mape = np.mean(np.abs((y_test - weighted_preds) / y_test)) * 100
    
    print("\nEnsemble prediction performance:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return weighted_preds, all_predictions, mae, rmse, mape