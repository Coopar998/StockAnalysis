"""
Stock Price Prediction System - Visualization Utilities
----------------------------------------------------
This file contains functions for plotting and visualization.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_performance(ticker, history, y_test, y_pred, test_dates, 
                   buy_signals, sell_signals, trend=None, output_dir=None):
    """
    Create and save performance plots for a stock prediction model
    
    Args:
        ticker: Stock ticker symbol
        history: Training history
        y_test: Actual prices
        y_pred: Predicted prices
        test_dates: Dates for testing period
        buy_signals: Boolean array of buy signals
        sell_signals: Boolean array of sell signals
        trend: Optional trend line to plot (e.g., moving average)
        output_dir: Directory to save the plots
    """
    plt.figure(figsize=(12, 9))
    
    # Plot 1: Training loss (if history is available)
    plt.subplot(2, 1, 1)
    if history and isinstance(history, dict) and 'loss' in history and len(history['loss']) > 1:
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{ticker} Model Training')
    else:
        plt.text(0.5, 0.5, 'No training history available (using pre-trained model)', 
                ha='center', va='center', fontsize=12)
        plt.title(f'{ticker} Using Pre-trained Model')
    plt.legend()
    
    # Plot 2: Price prediction and signals
    plt.subplot(2, 1, 2)
    
    # Check if predictions are reasonably close to actuals
    # If dramatically different, log a warning
    avg_actual = np.mean(y_test)
    avg_pred = np.mean(y_pred)
    max_allowed_deviation = 0.5  # Allow 50% difference
    
    if abs(avg_pred - avg_actual) / avg_actual > max_allowed_deviation:
        plt.figtext(0.5, 0.01, 
                   f"Warning: Large deviation between predictions and actuals. Check scaling.",
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    
    plt.plot(test_dates, y_test, label='Actual', color='blue')
    plt.plot(test_dates, y_pred, label='Predicted', color='orange')
    
    # Add trend line if provided
    if trend is not None:
        plt.plot(test_dates, trend, label='20-day MA', color='purple', linestyle='--', alpha=0.7)
    
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
    
    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # Save to output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{ticker}_prediction.png")
    else:
        save_path = f"{ticker}_prediction.png"
        
    plt.savefig(save_path)
    print(f"Saved prediction plot to {save_path}")
    plt.close()  # Close the figure to avoid display in non-interactive environments

def plot_model_comparison(ticker, y_test, predictions_dict, test_dates, output_dir=None):
    """
    Plot comparison of multiple model predictions for the same stock
    
    Args:
        ticker: Stock ticker symbol
        y_test: Actual prices
        predictions_dict: Dictionary of {model_name: predictions}
        test_dates: Dates for testing period
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual prices
    plt.plot(test_dates, y_test, label='Actual', color='black', linewidth=2)
    
    # Plot predictions from each model with different colors
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink']
    
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        plt.plot(test_dates, y_pred, label=model_name, color=color, alpha=0.7)
    
    plt.title(f'{ticker} - Prediction Comparison Across Models')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    
    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # Save to output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{ticker}_model_comparison.png")
    else:
        save_path = f"{ticker}_model_comparison.png"
        
    plt.savefig(save_path)
    print(f"Saved model comparison plot to {save_path}")
    plt.close()

def plot_feature_importance(ticker, feature_names, importance_scores, output_dir=None):
    """
    Plot feature importance for a stock prediction model
    
    Args:
        ticker: Stock ticker symbol
        feature_names: List of feature names
        importance_scores: List of importance scores corresponding to features
        output_dir: Directory to save the plot
    """
    # Sort features by importance
    indices = np.argsort(importance_scores)
    top_n = min(20, len(indices))  # Show at most top 20 features
    
    plt.figure(figsize=(10, 8))
    
    # Create horizontal bar chart
    y_pos = np.arange(top_n)
    plt.barh(y_pos, [importance_scores[i] for i in indices[-top_n:]])
    plt.yticks(y_pos, [feature_names[i] for i in indices[-top_n:]])
    
    plt.title(f'{ticker} - Feature Importance')
    plt.xlabel('Importance Score')
    
    plt.tight_layout()
    
    # Save to output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{ticker}_feature_importance.png")
    else:
        save_path = f"{ticker}_feature_importance.png"
        
    plt.savefig(save_path)
    print(f"Saved feature importance plot to {save_path}")
    plt.close()

def plot_error_analysis(ticker, y_test, y_pred, test_dates, output_dir=None):
    """
    Plot error analysis for a stock prediction model
    
    Args:
        ticker: Stock ticker symbol
        y_test: Actual prices
        y_pred: Predicted prices
        test_dates: Dates for testing period
        output_dir: Directory to save the plot
    """
    # Calculate errors
    errors = y_test - y_pred
    abs_errors = np.abs(errors)
    pct_errors = (abs_errors / y_test) * 100
    
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Absolute error over time
    plt.subplot(3, 1, 1)
    plt.plot(test_dates, abs_errors, color='red')
    plt.title(f'{ticker} - Absolute Prediction Error Over Time')
    plt.ylabel('Error (USD)')
    plt.axhline(y=np.mean(abs_errors), color='black', linestyle='--', 
               label=f'Mean: ${np.mean(abs_errors):.2f}')
    plt.legend()
    
    # Plot 2: Percentage error over time
    plt.subplot(3, 1, 2)
    plt.plot(test_dates, pct_errors, color='blue')
    plt.title('Percentage Error Over Time')
    plt.ylabel('Error (%)')
    plt.axhline(y=np.mean(pct_errors), color='black', linestyle='--', 
               label=f'Mean: {np.mean(pct_errors):.2f}%')
    plt.legend()
    
    # Plot 3: Error distribution (histogram)
    plt.subplot(3, 1, 3)
    plt.hist(errors, bins=30, alpha=0.7, color='green')
    plt.title('Error Distribution')
    plt.xlabel('Error (USD)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axvline(x=np.mean(errors), color='red', linestyle='-', 
               label=f'Mean: ${np.mean(errors):.2f}')
    plt.legend()
    
    plt.tight_layout()
    
    # Save to output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{ticker}_error_analysis.png")
    else:
        save_path = f"{ticker}_error_analysis.png"
        
    plt.savefig(save_path)
    print(f"Saved error analysis plot to {save_path}")
    plt.close()