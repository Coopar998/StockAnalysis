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

from data.fetcher import get_sp500_tickers, get_top_market_cap_stocks
from data.processor import prepare_data, create_sequences
from model.trainer import train_multi_stock_model
from trading.strategy import process_ticker
from trading.performance import calculate_portfolio_performance, create_strategy_comparison
from utils.feature_analysis import analyze_feature_importance, create_feature_importance_summary

def main():
    """Main function to run the stock prediction system"""
    

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=2500)).strftime('%Y-%m-%d')
    

    base_dir = "stock_prediction_results"
    model_dir = os.path.join(base_dir, "models")
    predictions_dir = os.path.join(base_dir, "predictions")
    feature_analysis_dir = os.path.join(base_dir, "feature_analysis")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(feature_analysis_dir, exist_ok=True)
    

    training_tickers = get_top_market_cap_stocks(100)
    print(f"Training on {len(training_tickers)} stocks")
    
    evaluation_tickers = training_tickers[:10]  
    print(f"Evaluating on: {evaluation_tickers}")
    
    pd.DataFrame(training_tickers, columns=['ticker']).to_csv(
        os.path.join(base_dir, "training_tickers.csv"), index=False)
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
    
    
    if mode == 1:
        print("Training individual models for each ticker")
        for ticker in evaluation_tickers:
            
            ticker_model_path = os.path.join(model_dir, ticker)
            os.makedirs(ticker_model_path, exist_ok=True)
            
            
            result = process_ticker(
                ticker, 
                start_date, 
                end_date, 
                model_path=os.path.join(ticker_model_path, "model.keras"),
                output_dir=predictions_dir,
                train_model=True,
                portfolio=portfolio
            )
            results.append(result)
            time.sleep(2) 
            
    elif mode == 2:
        print("Training one model on all tickers")
        
        multi_stock_model, ticker_data = train_multi_stock_model(
            training_tickers, 
            start_date, 
            end_date,
            model_save_path=os.path.join(model_dir, "multi_stock_model.keras")
        )
        
        for ticker in evaluation_tickers:
            result = process_ticker(
                ticker,
                start_date,
                end_date,
                model=multi_stock_model,
                output_dir=predictions_dir,
                train_model=False,
                portfolio=portfolio
            )
            results.append(result)
            time.sleep(1) 
            
    elif mode == 3:
        print("Using existing model for predictions")
        
        model_path = os.path.join(model_dir, "multi_stock_model.keras")
        
        for ticker in evaluation_tickers:
            result = process_ticker(
                ticker,
                start_date,
                end_date,
                model_path=model_path,
                output_dir=predictions_dir,
                train_model=False,
                portfolio=portfolio
            )
            results.append(result)
            time.sleep(1) 
    
    elif mode == 4:
        print("Running feature importance analysis")
       
        analysis_tickers = evaluation_tickers
        
        feature_analysis_results = {}
        
        for ticker in analysis_tickers:
            print(f"\nAnalyzing feature importance for {ticker}")
            data, message = prepare_data(ticker, start_date, end_date)
            
            if data is None:
                print(f"Skipping {ticker}: {message}")
                continue
            
        
            X, y, dates, features, _ = create_sequences(data)
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            

            importance = analyze_feature_importance(
                X_train, y_train, features,
                output_dir=os.path.join(feature_analysis_dir, ticker),
                ticker=ticker
            )

            top_features = importance['combined']['sorted_names'][:10]
            print(f"Top 10 features for {ticker}:")
            for i, feature in enumerate(top_features):
                score = importance['combined']['sorted_scores'][i]
                print(f"  {i+1}. {feature}: {score:.4f}")

            feature_analysis_results[ticker] = importance
            
            time.sleep(1)  
        
      
        if feature_analysis_results:
            create_feature_importance_summary(feature_analysis_results, feature_analysis_dir)
            print(f"Feature importance analysis complete. Results saved to {feature_analysis_dir}")
        else:
            print("No feature analysis results were generated.")
        
      
        return None, None
    
    
    portfolio_summary = calculate_portfolio_performance(portfolio, base_dir)
    

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(base_dir, "stock_results.csv"), index=False)
    print(f"Processed {len(results)} tickers, results saved to {os.path.join(base_dir, 'stock_results.csv')}")
    
    if len(results_df) > 0 and 'success' in results_df.columns and results_df['success'].all():
        create_strategy_comparison(results_df, base_dir)
    
    return results, portfolio

def run_feature_importance_comparison(tickers, start_date, end_date, output_dir):
    """
    Run a comparison of feature importance across multiple tickers
    
    Args:
        tickers: List of ticker symbols to analyze
        start_date, end_date: Date range for data
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    feature_analysis_results = {}
    all_features = set()
    
    for ticker in tickers:
        print(f"\nAnalyzing feature importance for {ticker}")
        data, message = prepare_data(ticker, start_date, end_date)
        
        if data is None:
            print(f"Skipping {ticker}: {message}")
            continue
        
        X, y, dates, features, _ = create_sequences(data)
        
        all_features.update(features)

        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        

        importance = analyze_feature_importance(
            X_train, y_train, features,
            output_dir=os.path.join(output_dir, ticker),
            ticker=ticker
        )
        
        feature_analysis_results[ticker] = importance
        
        time.sleep(1)  

    create_feature_importance_summary(feature_analysis_results, output_dir)

    try:
        import yfinance as yf
        from collections import defaultdict
        
       
        sectors = defaultdict(list)
        for ticker in tickers:
            if ticker in feature_analysis_results:
                try:
                    stock = yf.Ticker(ticker)
                    sector = stock.info.get('sector', 'Unknown')
                    sectors[sector].append(ticker)
                except:
                    sectors['Unknown'].append(ticker)
        
       
        for sector, sector_tickers in sectors.items():
            if len(sector_tickers) > 1:  
                sector_results = {ticker: feature_analysis_results[ticker] for ticker in sector_tickers}
                sector_dir = os.path.join(output_dir, f"sector_{sector.replace(' ', '_')}")
                os.makedirs(sector_dir, exist_ok=True)
                create_feature_importance_summary(sector_results, sector_dir)
                print(f"Created feature importance summary for sector: {sector}")
    except:
        print("Sector-based analysis failed. Continuing with overall analysis.")
    
    return feature_analysis_results

if __name__ == "__main__":
    results, portfolio = main()