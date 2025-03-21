"""
Stock Price Prediction System - Database Updater
----------------------------------------------
This script updates stock data in the database for active trading.
Run this daily to ensure you have the latest price data.
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import download_stock_data, get_sp500_tickers
from utils.database import StockDatabase
from data.processor import get_data_freshness_report

# Configure logging
def setup_logging(verbose=False):
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"db_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    level = logging.INFO if verbose else logging.WARNING
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('db_updater')

def update_database(tickers=None, days=2500, max_age=1, force=False, verbose=False):
    """
    Update stock data in the database
    
    Args:
        tickers: List of tickers to update (default: all in database + S&P 500)
        days: Number of days of historical data to keep
        max_age: Maximum age in days before data is considered stale
        force: Force update even if data is not stale
        verbose: Whether to print detailed output
        
    Returns:
        DataFrame with update results
    """
    logger = setup_logging(verbose)
    
    # Initialize database
    db = StockDatabase()
    
    # Date range for data
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Get tickers to update
    if tickers is None:
        # Get all tickers from database
        db_tickers = db.get_available_tickers()
        
        # Get S&P 500 tickers
        sp500_tickers = get_sp500_tickers()
        
        # Combine and remove duplicates
        tickers = list(set(db_tickers + sp500_tickers))
    
    logger.info(f"Updating {len(tickers)} tickers in database")
    
    # Get freshness report
    freshness_df = get_data_freshness_report()
    
    results = []
    
    # Process each ticker
    for i, ticker in enumerate(tickers):
        try:
            ticker_in_db = ticker in freshness_df['ticker'].values
            needs_update = force
            
            if ticker_in_db and not force:
                ticker_info = freshness_df[freshness_df['ticker'] == ticker].iloc[0]
                needs_update = ticker_info['needs_update']
            
            if needs_update or not ticker_in_db:
                logger.info(f"[{i+1}/{len(tickers)}] Updating {ticker}...")
                
                # Download data
                df = download_stock_data(
                    ticker, 
                    start_date, 
                    end_date,
                    force_download=force,
                    update_db=True,
                    max_age_days=max_age
                )
                
                if df is not None and not df.empty:
                    row_count = len(df)
                    results.append({
                        'ticker': ticker,
                        'status': 'updated',
                        'rows': row_count,
                        'update_time': datetime.now()
                    })
                    logger.info(f"Successfully updated {ticker} with {row_count} rows")
                else:
                    results.append({
                        'ticker': ticker,
                        'status': 'failed',
                        'rows': 0,
                        'update_time': datetime.now()
                    })
                    logger.warning(f"Failed to update {ticker}")
            else:
                # Already up to date
                results.append({
                    'ticker': ticker,
                    'status': 'current',
                    'rows': 0,
                    'update_time': freshness_df[freshness_df['ticker'] == ticker]['last_update'].iloc[0]
                })
                logger.info(f"[{i+1}/{len(tickers)}] {ticker} is already up to date")
            
            # Add delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error updating {ticker}: {e}")
            results.append({
                'ticker': ticker,
                'status': 'error',
                'rows': 0,
                'update_time': datetime.now(),
                'error': str(e)
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary
    updated = len(results_df[results_df['status'] == 'updated'])
    current = len(results_df[results_df['status'] == 'current'])
    failed = len(results_df[results_df['status'] == 'error']) + len(results_df[results_df['status'] == 'failed'])
    
    logger.info(f"Update complete: {updated} updated, {current} already current, {failed} failed")
    
    # Save results to CSV
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    results_file = os.path.join(log_dir, f"update_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(results_file, index=False)
    
    logger.info(f"Results saved to {results_file}")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update stock data in the database")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers to update")
    parser.add_argument("--days", type=int, default=2500, help="Number of days of historical data to keep")
    parser.add_argument("--max-age", type=int, default=1, help="Maximum age in days before data is considered stale")
    parser.add_argument("--force", action="store_true", help="Force update even if data is not stale")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    
    args = parser.parse_args()
    
    # Parse ticker list if provided
    ticker_list = None
    if args.tickers:
        ticker_list = [t.strip() for t in args.tickers.split(',')]
    
    update_database(
        tickers=ticker_list,
        days=args.days,
        max_age=args.max_age,
        force=args.force,
        verbose=args.verbose
    )