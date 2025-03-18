"""
Stock Price Prediction System - Model Builder
-------------------------------------------
This file contains various model architectures for stock price prediction.
"""

import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, GRU, Input, 
    Concatenate, Attention, LayerNormalization
)
from tensorflow.keras.optimizers import Adam

def build_basic_lstm_model(input_shape):
    """Build a standard LSTM model"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Use a custom name for the model to help with saving
    model._name = f"stock_standard_lstm_{int(time.time())}"
    
    return model

def build_deep_lstm_model(input_shape):
    """Build a deeper LSTM model with more layers"""
    model = Sequential([
        LSTM(160, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(120, return_sequences=True),
        Dropout(0.3),
        LSTM(80, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    
    # Use a custom name for the model to help with saving
    model._name = f"stock_deep_lstm_{int(time.time())}"
    
    return model

def build_bidirectional_model(input_shape):
    """Build a bidirectional LSTM model"""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Use a custom name for the model to help with saving
    model._name = f"stock_bidirectional_lstm_{int(time.time())}"
    
    return model

def build_hybrid_gru_lstm_model(input_shape):
    """Build a hybrid model with both GRU and LSTM layers"""
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Use a custom name for the model to help with saving
    model._name = f"stock_hybrid_gru_lstm_{int(time.time())}"
    
    return model

def build_attention_model(input_shape):
    """Build an LSTM model with attention mechanism"""
    # Define inputs
    inputs = Input(shape=input_shape)
    
    # LSTM layers
    lstm_out = LSTM(128, return_sequences=True)(inputs)
    lstm_out = Dropout(0.3)(lstm_out)
    lstm_out = LSTM(64, return_sequences=True)(lstm_out)
    
    # Apply attention
    attention_layer = Attention()([lstm_out, lstm_out])
    
    # Flatten and dense layers
    x = LSTM(32, return_sequences=False)(attention_layer)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Use a custom name for the model to help with saving
    model._name = f"stock_attention_lstm_{int(time.time())}"
    
    return model

def build_model(input_shape, model_type='standard'):
    """
    Build a neural network model based on specified architecture type
    
    Args:
        input_shape: Shape of input data (seq_length, n_features)
        model_type: Type of model architecture to build
        
    Returns:
        Compiled Keras model
    """
    if model_type == 'standard':
        model = build_basic_lstm_model(input_shape)
    elif model_type == 'deep':
        model = build_deep_lstm_model(input_shape)
    elif model_type == 'bidirectional':
        model = build_bidirectional_model(input_shape)
    elif model_type == 'hybrid':
        model = build_hybrid_gru_lstm_model(input_shape)
    elif model_type == 'attention':
        model = build_attention_model(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Built {model_type} model with input shape {input_shape}")
    
    return model