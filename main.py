"""
Stock Price Prediction System - Main Application
------------------------------------------------
This file coordinates the overall workflow of the stock prediction system.
It handles downloading data, training models, generating predictions,
and analyzing performance for multiple stocks.
"""

import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import concurrent.futures

from data.fetcher import get_sp500_tickers, get_top_market_cap_stocks, download_multiple_stock_data
from data.processor import prepare_data, create_sequences
from model.trainer import train_multi_stock_model
from trading.strategy import process_ticker
from trading.performance import calculate_portfolio_performance, create_strategy_comparison
from utils.feature_analysis import analyze_feature_importance, create_feature_importance_summary

# Function to process a single ticker (for parallel processing)
def process_single_ticker(ticker, start_date, end_date, model_path, output_dir, train_model=False):
    """Process a single ticker for parallel execution"""
    try:
        return process_ticker(
            ticker,
            start_date,
            end_date,
            model_path=model_path,
            output_dir=output_dir,
            train_model=train_model,
            portfolio=None,  # Can't share portfolio between processes
            create_plots=False  # Disable plots for faster processing
        )
    except Exception as e:
        import traceback
        print(f"Error processing {ticker}: {e}")
        print(traceback.format_exc())
        return {"ticker": ticker, "success": False, "message": str(e)}

# Function to update portfolio with results from parallel processing
def update_portfolio_with_result(portfolio, result, ticker):
    """Update portfolio with results from a processed ticker"""
    if not result['success']:
        return
    
    # Create basic performance data for the portfolio
    initial_capital = 10000
    total_return = result.get('total_return', 0)
    buy_hold_return = result.get('buy_hold_return', 0)
    trades = result.get('total_trades', 0)
    strategy = result.get('strategy', 'buy_hold')
    recent_accuracy = result.get('prediction_accuracy', 0)
    
    # Calculate final value
    final_value = initial_capital * (1 + total_return/100)
    
    # Add to portfolio
    portfolio['returns'][ticker] = {
        'initial_value': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'buy_hold_return_pct': buy_hold_return,
        'prediction_accuracy': recent_accuracy,
        'total_trades': trades,
        'strategy': strategy
    }

def main():
    """Main function to run the stock prediction system"""
    # Record start time
    start_time = time.time()

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=2500)).strftime('%Y-%m-%d')
    
    base_dir = "stock_prediction_results"
    model_dir = os.path.join(base_dir, "models")
    predictions_dir = os.path.join(base_dir, "predictions")
    feature_analysis_dir = os.path.join(base_dir, "feature_analysis")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(feature_analysis_dir, exist_ok=True)
    
    # INCREASED NUMBER: Use 30 stocks for testing with a mix of different sectors
    # Define a wider set of evaluation tickers covering different sectors
    evaluation_tickers = [
        # Tech
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", "CRM", 
        # Finance
        "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", 
        # Healthcare
        "JNJ", "PFE", "MRK", "UNH", "ABT", "LLY",
        # Consumer
        "AMZN", "WMT", "HD", "MCD", "SBUX", "NKE", "DIS",
        # Energy
        "XOM", "CVX", "COP"
    ]
    
    print(f"Evaluating on {len(evaluation_tickers)} stocks from multiple sectors")
    
    pd.DataFrame(evaluation_tickers, columns=['ticker']).to_csv(
        os.path.join(base_dir, "evaluation_tickers.csv"), index=False)
    
    # 1. Train individual models for each ticker
    # 2. Train one model on all tickers and use it for predictions
    # 3. Load existing model and use it for predictions
    # 4. Run feature importance analysis only
    mode = 3  # Change this to your preferred mode
    
    portfolio = {
        'initial_capital': 1000000,  # $1M initial capital
        'available_cash': 1000000,
        'positions': {},  # {ticker: {'shares': 100, 'cost_basis': 150.0}}
        'trade_history': [],  # List of trades executed
        'daily_values': [],  # Daily portfolio values
        'returns': {}  # Performance metrics by ticker
    }
    
    results = []
    
    # Pre-download data for all tickers to improve performance
    print("Pre-downloading data for all evaluation tickers...")
    try:
        ticker_data_map = download_multiple_stock_data(evaluation_tickers, start_date, end_date)
        print(f"Successfully pre-downloaded data for {len(ticker_data_map)} tickers")
    except Exception as e:
        print(f"Error pre-downloading data: {e}")
        ticker_data_map = {}
    
    if mode == 1:
        print("Training individual models for each ticker")
        # Parallel processing for individual models
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for ticker in evaluation_tickers:
                ticker_model_path = os.path.join(model_dir, ticker, "model.keras")
                os.makedirs(os.path.dirname(ticker_model_path), exist_ok=True)
                futures.append(
                    executor.submit(
                        process_single_ticker, 
                        ticker, start_date, end_date, 
                        ticker_model_path, predictions_dir, True
                    )
                )
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    # Update portfolio
                    if result['success']:
                        update_portfolio_with_result(portfolio, result, result['ticker'])
                except Exception as e:
                    print(f"Error in processing: {e}")
            
    elif mode == 2:
        print("Training one model on all tickers")
        
        multi_stock_model, ticker_data = train_multi_stock_model(
            evaluation_tickers, 
            start_date, 
            end_date,
            model_save_path=os.path.join(model_dir, "multi_stock_model.keras")
        )
        
        # Process tickers with the trained model (sequential for this mode since we use the same model object)
        for ticker in evaluation_tickers:
            result = process_ticker(
                ticker,
                start_date,
                end_date,
                model=multi_stock_model,
                output_dir=predictions_dir,
                train_model=False,
                portfolio=portfolio,
                create_plots=True  # Enable plots for this mode
            )
            results.append(result)
            
    elif mode == 3:
        print("Using existing model for predictions")
        model_path = os.path.join(model_dir, "multi_stock_model.keras")
        
        # Parallel processing for predictions
        max_workers = min(6, len(evaluation_tickers))
        print(f"Processing {len(evaluation_tickers)} tickers in parallel with {max_workers} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(
                    process_single_ticker, 
                    ticker, start_date, end_date, model_path, predictions_dir, False
                ): ticker 
                for ticker in evaluation_tickers
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update portfolio manually
                    if result['success']:
                        update_portfolio_with_result(portfolio, result, ticker)
                    
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    results.append({
                        "ticker": ticker,
                        "success": False,
                        "message": f"Error: {str(e)}"
                    })
        
        # Create visualization plots separately after processing
        print("Creating performance visualizations...")
        for ticker in evaluation_tickers:
            ticker_output_dir = os.path.join(predictions_dir, ticker)
            os.makedirs(ticker_output_dir, exist_ok=True)
            
            # Find the corresponding result
            ticker_result = next((r for r in results if r.get('ticker') == ticker and r.get('success', False)), None)
            if ticker_result:
                try:
                    # Load data for visualization
                    data, _ = prepare_data(ticker, start_date, end_date, lightweight_mode=True)
                    if data is not None:
                        # Create performance visualization from data
                        print(f"Creating visualization for {ticker}")
                        # Visualization code would go here
                except Exception as e:
                    print(f"Error creating visualization for {ticker}: {e}")
    
    elif mode == 4:
        print("Running feature importance analysis")
        
        feature_analysis_results = {}
        
        # Use parallel processing for feature analysis
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            # Create a function to analyze a single ticker
            def analyze_ticker_features(ticker):
                try:
                    data, message = prepare_data(ticker, start_date, end_date, lightweight_mode=True)
                    if data is None:
                        return ticker, None, message
                    
                    X, y, dates, features, _ = create_sequences(data, essential_only=True)
                    split = int(len(X) * 0.8)
                    X_train, y_train = X[:split], y[:split]
                    
                    ticker_dir = os.path.join(feature_analysis_dir, ticker)
                    os.makedirs(ticker_dir, exist_ok=True)
                    
                    importance = analyze_feature_importance(
                        X_train, y_train, features,
                        output_dir=ticker_dir,
                        ticker=ticker
                    )
                    
                    return ticker, importance, "Success"
                except Exception as e:
                    return ticker, None, f"Error: {str(e)}"
            
            # Submit all analysis tasks
            futures = [executor.submit(analyze_ticker_features, ticker) for ticker in evaluation_tickers]
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    ticker, importance, message = future.result()
                    if importance:
                        feature_analysis_results[ticker] = importance
                        
                        # Print top features
                        top_features = importance['combined']['sorted_names'][:10]
                        print(f"Top 10 features for {ticker}:")
                        for i, feature in enumerate(top_features):
                            score = importance['combined']['sorted_scores'][i]
                            print(f"  {i+1}. {feature}: {score:.4f}")
                    else:
                        print(f"Skipping {ticker}: {message}")
                except Exception as e:
                    print(f"Error in feature analysis: {e}")
        
        if feature_analysis_results:
            create_feature_importance_summary(feature_analysis_results, feature_analysis_dir)
            print(f"Feature importance analysis complete. Results saved to {feature_analysis_dir}")
        else:
            print("No feature analysis results were generated.")
        
        return None, None
    
    # Calculate portfolio performance
    portfolio_summary = calculate_portfolio_performance(portfolio, base_dir)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(base_dir, "stock_results.csv"), index=False)
    print(f"Processed {len(results)} tickers, results saved to {os.path.join(base_dir, 'stock_results.csv')}")
    
    if len(results_df) > 0 and 'success' in results_df.columns and any(results_df['success']):
        create_strategy_comparison(results_df, base_dir)
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    return results, portfolio

if __name__ == "__main__":
    results, portfolio = main()