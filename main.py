"""
Stock Price Prediction System - Main Application
------------------------------------------------
This file coordinates the overall workflow of the stock prediction system.
It handles downloading data, training models, generating predictions,
and analyzing performance for multiple stocks.
Updated to use configuration settings for more realistic performance.
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
import logging

from data.fetcher import get_sp500_tickers, get_top_market_cap_stocks, download_multiple_stock_data
from data.processor import prepare_data, create_sequences
from model.trainer import train_multi_stock_model
from trading.strategy import process_ticker
from trading.performance import calculate_portfolio_performance, create_strategy_comparison
from utils.feature_analysis import analyze_feature_importance, create_feature_importance_summary
from utils.config_reader import get_config

# Configure logging
def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.INFO if verbose else logging.WARNING
    format_str = '%(asctime)s - %(levelname)s - %(message)s' if verbose else '%(message)s'
    logging.basicConfig(level=level, format=format_str)
    # Create logger
    logger = logging.getLogger('stock_prediction')
    logger.setLevel(level)
    # Return configured logger
    return logger

# Function to process a single ticker (for parallel processing)
def process_single_ticker(ticker, start_date, end_date, model_path, output_dir, train_model=False, verbose=False):
    """Process a single ticker for parallel execution"""
    try:
        # Get configuration
        config = get_config()
        min_data_points = config.get('data_processing', 'min_data_points', default=250)
        lightweight_mode = config.get('data_processing', 'lightweight_mode', default=True)
        
        # Process the ticker
        return process_ticker(
            ticker,
            start_date,
            end_date,
            model_path=model_path,
            output_dir=output_dir,
            train_model=train_model,
            portfolio=None,  # Can't share portfolio between processes
            create_plots=True,  # Enable plots for visualization
            min_data_points=min_data_points,
            lightweight_mode=lightweight_mode,
            verbose=verbose
        )
    except Exception as e:
        import traceback
        if verbose:
            print(f"Error processing {ticker}: {e}")
            print(traceback.format_exc())
        return {"ticker": ticker, "success": False, "message": str(e)}

# Function to update portfolio with results from parallel processing
def update_portfolio_with_result(portfolio, result, ticker, verbose=False):
    """Update portfolio with results from a processed ticker"""
    if not result.get('success', False):
        return
    
    # Get configuration
    config = get_config()
    
    # Create basic performance data for the portfolio
    initial_capital = config.get('portfolio', 'initial_capital', default=1000000) / config.get('portfolio', 'max_stocks', default=40)
    
    # Get results data
    total_return = result.get('total_return', 0)
    buy_hold_return = result.get('buy_hold_return', 0)
    trades = result.get('total_trades', 0)
    strategy = result.get('strategy', 'buy_hold')
    recent_accuracy = result.get('prediction_accuracy', 0)
    winning_trades = result.get('winning_trades', 0)
    losing_trades = result.get('losing_trades', 0)
    
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
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'strategy': strategy
    }
    
    # Extract trades if available in the result
    if 'trades' in result:
        portfolio['returns'][ticker]['trades'] = result['trades']
    
    if verbose:
        print(f"Added {ticker} performance to portfolio: {total_return:.2f}% return with {trades} trades")

def main(verbose=False):
    """Main function to run the stock prediction system"""
    # Setup logging based on verbosity
    logger = setup_logging(verbose)
    
    # Get configuration
    config = get_config()
    
    # Record start time
    start_time = time.time()

    # Set date range
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=3000)).strftime('%Y-%m-%d')
    
    # Setup directories
    base_dir = "stock_prediction_results"
    model_dir = os.path.join(base_dir, "models")
    predictions_dir = os.path.join(base_dir, "predictions")
    feature_analysis_dir = os.path.join(base_dir, "feature_analysis")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(feature_analysis_dir, exist_ok=True)
    
    # Define evaluation tickers - use a mix of different sectors
    evaluation_tickers = [
        # Tech
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", "CRM", "CSCO", "ORCL", 
        # Finance
        "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "WFC", "C", "BLK",
        # Healthcare
        "JNJ", "PFE", "MRK", "UNH", "ABT", "LLY", "ABBV", "TMO", "AMGN",
        # Consumer
        "AMZN", "WMT", "HD", "MCD", "SBUX", "NKE", "DIS", "PG", "KO", "PEP",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "OXY", "VLO", "MPC"
    ]
    
    # Limit to max_stocks from config
    max_stocks = config.get('portfolio', 'max_stocks', default=40)
    evaluation_tickers = evaluation_tickers[:max_stocks]
    
    if verbose:
        logger.info(f"Evaluating on {len(evaluation_tickers)} stocks from multiple sectors")
    
    pd.DataFrame(evaluation_tickers, columns=['ticker']).to_csv(
        os.path.join(base_dir, "evaluation_tickers.csv"), index=False)
    
    # 1. Train individual models for each ticker
    # 2. Train one model on all tickers and use it for predictions
    # 3. Load existing model and use it for predictions
    # 4. Run feature importance analysis only
    mode = 3  # Change this to your preferred mode
    
    # Initialize portfolio
    initial_capital = config.get('portfolio', 'initial_capital', default=1000000)
    portfolio = {
        'initial_capital': initial_capital,
        'available_cash': initial_capital,
        'positions': {},
        'trade_history': [],
        'daily_values': [],
        'returns': {}
    }
    
    results = []
    
    # Pre-download data for all tickers to improve performance
    if verbose:
        logger.info("Pre-downloading data for all evaluation tickers...")
    try:
        ticker_data_map = download_multiple_stock_data(evaluation_tickers, start_date, end_date)
        if verbose:
            logger.info(f"Successfully pre-downloaded data for {len(ticker_data_map)} tickers")
    except Exception as e:
        logger.error(f"Error pre-downloading data: {e}")
        ticker_data_map = {}
    
    if mode == 1:
        if verbose:
            logger.info("Training individual models for each ticker")
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
                        ticker_model_path, predictions_dir, True, verbose
                    )
                )
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    # Update portfolio
                    if result.get('success', False):
                        update_portfolio_with_result(portfolio, result, result['ticker'], verbose)
                except Exception as e:
                    logger.error(f"Error in processing: {e}")
            
    elif mode == 2:
        if verbose:
            logger.info("Training one model on all tickers")
        
        multi_stock_model, ticker_data = train_multi_stock_model(
            evaluation_tickers, 
            start_date, 
            end_date,
            model_save_path=os.path.join(model_dir, "multi_stock_model.keras"),
            verbose=verbose
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
                create_plots=True,
                verbose=verbose
            )
            results.append(result)
            
    elif mode == 3:
        if verbose:
            logger.info("Using existing model for predictions")
        model_path = os.path.join(model_dir, "multi_stock_model.keras")
        
        # Increase batch size for faster processing with more stocks
        batch_size = min(len(evaluation_tickers), 10)
        
        # Chunk the tickers into batches to avoid memory issues
        ticker_chunks = [evaluation_tickers[i:i+batch_size] for i in range(0, len(evaluation_tickers), batch_size)]
        
        for chunk_idx, ticker_chunk in enumerate(ticker_chunks):
            if verbose:
                logger.info(f"Processing chunk {chunk_idx+1}/{len(ticker_chunks)} with {len(ticker_chunk)} tickers")
                
            # Parallel processing for predictions
            max_workers = min(6, len(ticker_chunk))
            if verbose:
                logger.info(f"Processing {len(ticker_chunk)} tickers in parallel with {max_workers} workers")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_ticker = {
                    executor.submit(
                        process_single_ticker, 
                        ticker, start_date, end_date, model_path, predictions_dir, False, verbose
                    ): ticker 
                    for ticker in ticker_chunk
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update portfolio manually
                        if result.get('success', False):
                            update_portfolio_with_result(portfolio, result, ticker, verbose)
                        
                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {e}")
                        results.append({
                            "ticker": ticker,
                            "success": False,
                            "message": f"Error: {str(e)}"
                        })
        
        # Create visualization plots separately after processing
        if verbose:
            logger.info("Creating performance visualizations...")
        
        for ticker in evaluation_tickers:
            ticker_output_dir = os.path.join(predictions_dir, ticker)
            os.makedirs(ticker_output_dir, exist_ok=True)
            
            # Find the corresponding result
            ticker_result = next((r for r in results if r.get('ticker') == ticker and r.get('success', False)), None)
            if ticker_result:
                try:
                    # Load data for visualization
                    data, _ = prepare_data(ticker, start_date, end_date)
                    if data is not None and verbose:
                        # Create performance visualization from data
                        logger.info(f"Creating visualization for {ticker}")
                        # Visualization code would go here
                except Exception as e:
                    if verbose:
                        logger.error(f"Error creating visualization for {ticker}: {e}")
    
    elif mode == 4:
        if verbose:
            logger.info("Running feature importance analysis")
        
        feature_analysis_results = {}
        
        # Use parallel processing for feature analysis
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            # Create a function to analyze a single ticker
            def analyze_ticker_features(ticker):
                try:
                    config = get_config()
                    min_data_points = config.get('data_processing', 'min_data_points', default=250)
                    
                    data, message = prepare_data(ticker, start_date, end_date, min_data_points=min_data_points)
                    if data is None:
                        return ticker, None, message
                    
                    X, y, dates, features, _ = create_sequences(data)
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
                        if verbose:
                            top_features = importance['combined']['sorted_names'][:10]
                            logger.info(f"Top 10 features for {ticker}:")
                            for i, feature in enumerate(top_features):
                                score = importance['combined']['sorted_scores'][i]
                                logger.info(f"  {i+1}. {feature}: {score:.4f}")
                    else:
                        if verbose:
                            logger.warning(f"Skipping {ticker}: {message}")
                except Exception as e:
                    logger.error(f"Error in feature analysis: {e}")
        
        if feature_analysis_results:
            create_feature_importance_summary(feature_analysis_results, feature_analysis_dir)
            if verbose:
                logger.info(f"Feature importance analysis complete. Results saved to {feature_analysis_dir}")
        else:
            logger.warning("No feature analysis results were generated.")
        
        return None, None
    
    # Calculate portfolio performance
    portfolio_summary = calculate_portfolio_performance(portfolio, base_dir)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(base_dir, "stock_results.csv"), index=False)
    logger.info(f"Processed {len(results)} tickers, results saved to {os.path.join(base_dir, 'stock_results.csv')}")
    
    if len(results_df) > 0 and 'success' in results_df.columns and any(results_df['success']):
        create_strategy_comparison(results_df, base_dir)
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    return results, portfolio

if __name__ == "__main__":
    # Set verbose to True for detailed output
    results, portfolio = main(verbose=True)