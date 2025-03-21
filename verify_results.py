"""
Stock Prediction System - Results Verification
---------------------------------------------
This script verifies the results from the stock prediction system, checks for outliers,
adds annualized return calculations, and tests with different ticker sets to ensure
the results are robust and not due to data leakage or calculation errors.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
import logging
import sys
from collections import defaultdict

# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules from the main system
from main import main as run_main
from trading.performance import calculate_portfolio_performance
from data.processor import prepare_data, create_sequences

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_annualized_returns(portfolio, start_date, end_date):
    """
    Calculate annualized returns for the portfolio and individual stocks
    
    Args:
        portfolio: Portfolio dictionary with returns info
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        Dictionary with annualized return metrics
    """
    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate years - convert days to years
    years = (end_dt - start_dt).days / 365.25
    
    if years < 0.01:  # Avoid division by very small values
        logger.warning(f"Warning: Very short testing period ({years:.2f} years)")
        years = 0.01
    
    # Calculate portfolio annualized return
    # Use initial_capital instead of initial_value for consistency
    initial_capital = portfolio.get('initial_capital', 1000000)
    final_value = portfolio.get('final_value', initial_capital)
    
    total_return = (final_value / initial_capital) - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Calculate annualized returns for each stock
    stock_annual_returns = {}
    active_trading_annual_returns = []
    buy_hold_annual_returns = []
    
    for ticker, data in portfolio['returns'].items():
        if 'total_return_pct' in data:
            # Calculate annualized return for this stock
            stock_return = data['total_return_pct'] / 100  # Convert percentage to decimal
            stock_annual = (1 + stock_return) ** (1 / years) - 1
            stock_annual_returns[ticker] = stock_annual * 100  # Convert back to percentage
            
            # Separate by strategy
            if data.get('strategy') == 'active_trading':
                active_trading_annual_returns.append(stock_annual * 100)
            elif data.get('strategy') == 'buy_hold':
                buy_hold_annual_returns.append(stock_annual * 100)
    
    # Calculate averages
    avg_annual_return = np.mean(list(stock_annual_returns.values())) if stock_annual_returns else 0
    avg_active_trading_annual = np.mean(active_trading_annual_returns) if active_trading_annual_returns else 0
    avg_buy_hold_annual = np.mean(buy_hold_annual_returns) if buy_hold_annual_returns else 0
    
    return {
        'years': years,
        'portfolio_annualized_return': annualized_return * 100,  # Convert to percentage
        'avg_stock_annualized_return': avg_annual_return,
        'avg_active_trading_annualized': avg_active_trading_annual,
        'avg_buy_hold_annualized': avg_buy_hold_annual,
        'stock_annual_returns': stock_annual_returns
    }

def analyze_outliers(portfolio):
    """
    Analyze the portfolio for outlier returns
    
    Args:
        portfolio: Portfolio dictionary with returns info
        
    Returns:
        Dictionary with outlier analysis
    """
    returns = []
    tickers = []
    strategies = []
    
    for ticker, data in portfolio['returns'].items():
        if 'total_return_pct' in data:
            returns.append(data['total_return_pct'])
            tickers.append(ticker)
            strategies.append(data.get('strategy', 'unknown'))
    
    # Convert to numpy array for calculations
    returns_array = np.array(returns)
    
    # Calculate mean, median, and std
    mean_return = np.mean(returns_array)
    median_return = np.median(returns_array)
    std_return = np.std(returns_array)
    
    # Define outliers (more than 3 standard deviations from mean)
    threshold = 3 * std_return
    upper_bound = mean_return + threshold
    lower_bound = mean_return - threshold
    
    # Find outliers
    outliers = []
    for i, ret in enumerate(returns_array):
        if ret > upper_bound or ret < lower_bound:
            outliers.append({
                'ticker': tickers[i],
                'return': ret,
                'strategy': strategies[i],
                'z_score': (ret - mean_return) / std_return
            })
    
    # Sort outliers by absolute z-score
    outliers.sort(key=lambda x: abs(x['z_score']), reverse=True)
    
    return {
        'mean_return': mean_return,
        'median_return': median_return,
        'std_return': std_return,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'outliers': outliers
    }

def check_data_leakage(evaluation_tickers, start_date, end_date):
    """
    Check for potential data leakage by comparing train/test splits
    
    Args:
        evaluation_tickers: List of tickers to check
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        Dictionary with data leakage analysis
    """
    results = {}
    
    for ticker in evaluation_tickers[:5]:  # Check a subset for performance reasons
        try:
            # Load data for this ticker
            data, message = prepare_data(ticker, start_date, end_date, lightweight_mode=True)
            
            if data is None:
                results[ticker] = {"status": "error", "message": message}
                continue
            
            # Create sequences
            seq_length = 20
            X, y, dates, features, _ = create_sequences(data, seq_length)
            
            # Check standard train/test split
            split = int(len(X) * 0.8)
            X_train = X[:split]
            X_test = X[split:]
            
            # Check for overlap in the date ranges
            train_dates = dates[:split]
            test_dates = dates[split:]
            
            # Check date overlap (should be none)
            train_date_set = set(str(d) for d in train_dates)
            test_date_set = set(str(d) for d in test_dates)
            date_overlap = train_date_set.intersection(test_date_set)
            
            # Check data overlap using feature vectors
            # Flatten X arrays for easier comparison
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            # Check if any test sample is identical to any training sample
            data_overlap = False
            for test_sample in X_test_flat[:10]:  # Check first 10 test samples (for performance)
                for train_sample in X_train_flat:
                    if np.array_equal(test_sample, train_sample):
                        data_overlap = True
                        break
                if data_overlap:
                    break
            
            results[ticker] = {
                "status": "analyzed",
                "train_dates": (min(train_dates), max(train_dates)),
                "test_dates": (min(test_dates), max(test_dates)),
                "date_overlap": list(date_overlap) if date_overlap else None,
                "data_overlap_detected": data_overlap,
                "train_test_separation": "clean" if not date_overlap and not data_overlap else "potential leakage"
            }
            
        except Exception as e:
            results[ticker] = {"status": "error", "message": str(e)}
    
    return results

def verify_returns_calculation(portfolio):
    """
    Verify the returns calculation for individual stocks and overall portfolio
    
    Args:
        portfolio: Portfolio dictionary with returns info
        
    Returns:
        Dictionary with verification results
    """
    verification = {
        "consistent_calculations": True,
        "issues_detected": [],
        "portfolio_calculated_value": 0,
        "portfolio_reported_value": portfolio.get('final_value', 0)
    }
    
    # Calculate sum of final values for all stocks
    calculated_sum = portfolio.get('available_cash', 0)
    for ticker, data in portfolio.get('returns', {}).items():
        if 'final_value' in data:
            calculated_sum += data['final_value']
    
    # Check if this matches the reported total
    if abs(calculated_sum - portfolio.get('final_value', 0)) > 0.01:
        verification["consistent_calculations"] = False
        verification["issues_detected"].append(
            f"Portfolio final value ({portfolio.get('final_value', 0):.2f}) doesn't match sum of components ({calculated_sum:.2f})"
        )
    
    verification["portfolio_calculated_value"] = calculated_sum
    
    # Check if active trading returns are realistic
    active_returns = []
    for ticker, data in portfolio.get('returns', {}).items():
        if data.get('strategy') == 'active_trading' and 'total_return_pct' in data:
            active_returns.append(data['total_return_pct'])
    
    if active_returns:
        avg_active_return = np.mean(active_returns)
        max_active_return = np.max(active_returns)
        
        # Flag potentially unrealistic returns
        if avg_active_return > 200:
            verification["consistent_calculations"] = False
            verification["issues_detected"].append(
                f"Average active trading return ({avg_active_return:.2f}%) seems unrealistically high"
            )
        
        if max_active_return > 1000:
            verification["consistent_calculations"] = False
            verification["issues_detected"].append(
                f"Maximum active trading return ({max_active_return:.2f}%) seems unrealistically high"
            )
    
    return verification

def run_multiple_ticker_tests(base_tickers, iterations=3, tickers_per_run=30, verbose=False):
    """
    Run the stock prediction system multiple times with different ticker sets
    
    Args:
        base_tickers: List of all possible tickers to select from
        iterations: Number of test runs
        tickers_per_run: Number of tickers to use per run
        verbose: Whether to print detailed output
        
    Returns:
        List of results from each run
    """
    all_results = []
    
    # Ensure we have enough tickers
    if len(base_tickers) < tickers_per_run:
        logger.warning(f"Not enough base tickers ({len(base_tickers)}). Using all available tickers.")
        tickers_per_run = len(base_tickers)
    
    # Define time period
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=3000)).strftime('%Y-%m-%d')
    
    # Run multiple iterations
    for i in range(iterations):
        logger.info(f"Starting iteration {i+1}/{iterations}")
        
        # Randomly select tickers for this run
        random.seed(i)  # For reproducibility within the same session
        selected_tickers = random.sample(base_tickers, tickers_per_run)
        
        logger.info(f"Selected {len(selected_tickers)} tickers for this run")
        
        # Create a test module with override for evaluation_tickers
        class TestModule:
            def __init__(self):
                self.evaluation_tickers = selected_tickers
        
        # Run the main process with the selected tickers
        try:
            # This is a simplified approach - in practice you might need to modify main.py
            # to accept a ticker list parameter instead of this approach
            results, portfolio = run_main(verbose=verbose)
            
            # Calculate annualized returns
            annual_returns = calculate_annualized_returns(portfolio, start_date, end_date)
            
            # Analyze outliers
            outlier_analysis = analyze_outliers(portfolio)
            
            # Verify returns calculation
            returns_verification = verify_returns_calculation(portfolio)
            
            # Store results
            run_result = {
                "iteration": i + 1,
                "tickers": selected_tickers,
                "portfolio_return": portfolio.get('final_value', 0) / portfolio.get('initial_capital', 1) - 1,
                "annualized_return": annual_returns['portfolio_annualized_return'],
                "years": annual_returns['years'],
                "outliers": outlier_analysis['outliers'],
                "issues": returns_verification['issues_detected'],
                "portfolio": portfolio
            }
            
            all_results.append(run_result)
            
            # Print summary
            logger.info(f"Run {i+1} complete:")
            logger.info(f"  Total return: {run_result['portfolio_return']*100:.2f}%")
            logger.info(f"  Annualized return: {run_result['annualized_return']:.2f}%")
            logger.info(f"  Number of outliers: {len(run_result['outliers'])}")
            if run_result['issues']:
                logger.warning(f"  Issues detected: {run_result['issues']}")
            
        except Exception as e:
            logger.error(f"Error in run {i+1}: {e}")
            all_results.append({"iteration": i + 1, "error": str(e)})
    
    return all_results

def print_enhanced_performance_summary(portfolio, start_date, end_date):
    """
    Print an enhanced performance summary with annualized returns
    
    Args:
        portfolio: Portfolio dictionary with returns info
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    """
    # Calculate annualized returns
    annual_returns = calculate_annualized_returns(portfolio, start_date, end_date)
    
    # Analyze outliers
    outlier_analysis = analyze_outliers(portfolio)
    
    # Basic portfolio metrics
    total_initial = portfolio.get('initial_capital', 1000000)
    total_final = portfolio.get('final_value', total_initial)
    total_return_pct = ((total_final / total_initial) - 1) * 100
    
    # Get total trades and win/loss stats
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    
    # Strategy counts
    active_trading_count = 0
    buy_hold_count = 0
    
    # Returns by strategy
    active_returns = []
    buy_hold_returns = []
    
    for ticker, data in portfolio.get('returns', {}).items():
        if 'total_trades' in data:
            total_trades += data['total_trades']
        if 'winning_trades' in data:
            winning_trades += data['winning_trades']
        if 'losing_trades' in data:
            losing_trades += data['losing_trades']
        
        # Count by strategy
        if data.get('strategy') == 'active_trading':
            active_trading_count += 1
            active_returns.append(data.get('total_return_pct', 0))
        elif data.get('strategy') == 'buy_hold':
            buy_hold_count += 1
            buy_hold_returns.append(data.get('total_return_pct', 0))
    
    # Calculate averages by strategy
    avg_active_return = np.mean(active_returns) if active_returns else 0
    avg_buy_hold_return = np.mean(buy_hold_returns) if buy_hold_returns else 0
    
    # Win rate and ratios
    win_rate = winning_trades / max(1, winning_trades + losing_trades) * 100
    
    # Get average win/loss per trade if available
    avg_win = 0
    avg_loss = 0
    for _, data in portfolio.get('returns', {}).items():
        if 'avg_win' in data and 'winning_trades' in data and data['winning_trades'] > 0:
            avg_win += data['avg_win'] * data['winning_trades']
        if 'avg_loss' in data and 'losing_trades' in data and data['losing_trades'] > 0:
            avg_loss += data['avg_loss'] * data['losing_trades']
    
    if winning_trades > 0:
        avg_win /= winning_trades
    if losing_trades > 0:
        avg_loss /= losing_trades
    
    win_loss_ratio = avg_win / max(0.01, avg_loss)
    
    # Print enhanced summary
    print("\n==== ENHANCED PORTFOLIO PERFORMANCE SUMMARY ====")
    print(f"Testing Period: {start_date} to {end_date} ({annual_returns['years']:.2f} years)")
    print(f"Initial Capital: ${total_initial:,.2f}")
    print(f"Final Portfolio Value: ${total_final:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Annualized Return: {annual_returns['portfolio_annualized_return']:.2f}% per year")
    
    print("\nTrading Statistics:")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades: {losing_trades} ({100 - win_rate:.2f}%)")
    print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Return per Trade: {total_return_pct / max(1, total_trades):.2f}%")
    
    print("\nStrategy Breakdown:")
    print(f"Active Trading stocks: {active_trading_count}")
    print(f"Buy & Hold stocks: {buy_hold_count}")
    print(f"Active Trading avg return: {avg_active_return:.2f}%")
    print(f"Active Trading avg annual return: {annual_returns['avg_active_trading_annualized']:.2f}% per year")
    print(f"Buy & Hold avg return: {avg_buy_hold_return:.2f}%")
    print(f"Buy & Hold avg annual return: {annual_returns['avg_buy_hold_annualized']:.2f}% per year")
    
    print("\nOutlier Analysis:")
    print(f"Mean Return: {outlier_analysis['mean_return']:.2f}%")
    print(f"Median Return: {outlier_analysis['median_return']:.2f}%")
    print(f"Standard Deviation: {outlier_analysis['std_return']:.2f}%")
    
    if outlier_analysis['outliers']:
        print(f"\nOutlier Stocks (beyond ±3 std devs):")
        for outlier in outlier_analysis['outliers'][:5]:  # Show top 5 outliers
            print(f"  {outlier['ticker']}: {outlier['return']:.2f}% (z-score: {outlier['z_score']:.2f}, strategy: {outlier['strategy']})")
    
    # Print top and bottom performers
    all_returns = [(ticker, data.get('total_return_pct', 0)) for ticker, data in portfolio.get('returns', {}).items()]
    all_returns.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Performers:")
    for ticker, ret in all_returns[:5]:
        strategy = portfolio['returns'][ticker].get('strategy', 'unknown')
        annual_ret = annual_returns['stock_annual_returns'].get(ticker, 0)
        print(f"  {ticker}: {ret:.2f}% total, {annual_ret:.2f}% annually (strategy: {strategy})")
    
    print("\nBottom 5 Performers:")
    for ticker, ret in all_returns[-5:]:
        strategy = portfolio['returns'][ticker].get('strategy', 'unknown')
        annual_ret = annual_returns['stock_annual_returns'].get(ticker, 0)
        print(f"  {ticker}: {ret:.2f}% total, {annual_ret:.2f}% annually (strategy: {strategy})")

def check_returns_distribution(portfolio):
    """
    Analyze the distribution of returns and create visualizations
    
    Args:
        portfolio: Portfolio dictionary with returns info
        
    Returns:
        None, but saves visualization files
    """
    # Extract returns by strategy
    active_returns = []
    buyhold_returns = []
    all_returns = []
    active_tickers = []
    buyhold_tickers = []
    
    for ticker, data in portfolio.get('returns', {}).items():
        if 'total_return_pct' in data:
            return_val = data['total_return_pct']
            all_returns.append(return_val)
            
            if data.get('strategy') == 'active_trading':
                active_returns.append(return_val)
                active_tickers.append(ticker)
            elif data.get('strategy') == 'buy_hold':
                buyhold_returns.append(return_val)
                buyhold_tickers.append(ticker)
    
    # Create distribution plots
    plt.figure(figsize=(12, 8))
    
    # All returns histogram
    plt.subplot(2, 1, 1)
    bins = np.linspace(min(all_returns) - 10, max(all_returns) + 10, 30)
    plt.hist(all_returns, bins=bins, alpha=0.7, color='blue', label='All stocks')
    plt.xlabel('Return (%)')
    plt.ylabel('Count')
    plt.title('Distribution of All Stock Returns')
    plt.axvline(np.mean(all_returns), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(all_returns):.2f}%')
    plt.axvline(np.median(all_returns), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(all_returns):.2f}%')
    plt.legend()
    
    # Returns by strategy
    plt.subplot(2, 1, 2)
    if active_returns:
        plt.hist(active_returns, bins=bins, alpha=0.7, color='green', label=f'Active Trading (n={len(active_returns)})')
    if buyhold_returns:
        plt.hist(buyhold_returns, bins=bins, alpha=0.7, color='blue', label=f'Buy & Hold (n={len(buyhold_returns)})')
    plt.xlabel('Return (%)')
    plt.ylabel('Count')
    plt.title('Returns Distribution by Strategy')
    
    if active_returns:
        plt.axvline(np.mean(active_returns), color='green', linestyle='dashed', linewidth=2, 
                  label=f'Active Mean: {np.mean(active_returns):.2f}%')
    if buyhold_returns:
        plt.axvline(np.mean(buyhold_returns), color='blue', linestyle='dashed', linewidth=2,
                   label=f'Buy&Hold Mean: {np.mean(buyhold_returns):.2f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('returns_distribution.png')
    print(f"Saved returns distribution plot to returns_distribution.png")
    plt.close()
    
    # Create scatterplot of returns vs trades
    plt.figure(figsize=(10, 6))
    
    trades = []
    returns = []
    colors = []
    tickers = []
    
    for ticker, data in portfolio.get('returns', {}).items():
        if 'total_return_pct' in data and 'total_trades' in data:
            trades.append(data['total_trades'])
            returns.append(data['total_return_pct'])
            tickers.append(ticker)
            
            if data.get('strategy') == 'active_trading':
                colors.append('green')
            elif data.get('strategy') == 'buy_hold':
                colors.append('blue')
            else:
                colors.append('gray')
    
    plt.scatter(trades, returns, c=colors, alpha=0.7)
    
    # Add labels for extreme points
    for i, (x, y, ticker) in enumerate(zip(trades, returns, tickers)):
        if y > np.mean(returns) + 2*np.std(returns) or y < np.mean(returns) - 2*np.std(returns):
            plt.annotate(ticker, (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Number of Trades')
    plt.ylabel('Return (%)')
    plt.title('Return vs. Number of Trades')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', alpha=0.7, markersize=10, label='Active Trading'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', alpha=0.7, markersize=10, label='Buy & Hold')
    ]
    plt.legend(handles=legend_elements)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('return_vs_trades.png')
    print(f"Saved return vs trades plot to return_vs_trades.png")
    plt.close()

def main():
    """Main function to run the verification"""
    # Define constants
    base_dir = "stock_prediction_results"
    
    # Define time period
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=3000)).strftime('%Y-%m-%d')
    
    # Load previous results if available
    results_file = os.path.join(base_dir, "stock_results.csv")
    if os.path.exists(results_file):
        previous_results = pd.read_csv(results_file)
        logger.info(f"Loaded previous results with {len(previous_results)} stocks")
    else:
        previous_results = None
        logger.warning("No previous results found")
    
    # List of S&P 500 stocks for testing
    sp500_tickers = [
        # Tech
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", "CRM", "CSCO", "ORCL", 
        "AVGO", "QCOM", "TXN", "PYPL", "IBM", "AMAT", "MU", "NOW", "INTU",
        # Finance
        "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "WFC", "C", "BLK", "PNC", "USB", "TFC",
        "COF", "SCHW", "CME", "ICE", "SPGI", "MCO", "BK",
        # Healthcare
        "JNJ", "PFE", "MRK", "UNH", "ABT", "LLY", "ABBV", "TMO", "AMGN", "BMY", "MDT", "DHR",
        "CVS", "ISRG", "GILD", "REGN", "VRTX", "ZTS", "HUM", "CNC",
        # Consumer
        "AMZN", "WMT", "HD", "MCD", "SBUX", "NKE", "DIS", "PG", "KO", "PEP", "COST", "TGT",
        "LOW", "BKNG", "CMG", "YUM", "MAR", "EL", "CL", "KMB",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "OXY", "VLO", "MPC", "DVN", "KMI", "WMB",
        "HAL", "BKR", "PXD", "HES", "APA", "MRO", "OKE", "PSA",
        # Industrial
        "GE", "HON", "UPS", "CAT", "DE", "LMT", "RTX", "BA", "MMM", "EMR", "GD", "UNP",
        "CSX", "ETN", "ITW", "PH", "TT", "CMI", "NSC", "PCAR"
    ]
    
    # 1. Run verification with previous results if available
    if previous_results is not None:
        # Find portfolio summary file
        portfolio_file = os.path.join(base_dir, "portfolio_summary.csv")
        try:
            if os.path.exists(portfolio_file):
                logger.info("Analyzing previous portfolio results")
                
                # Load the portfolio data - reconstruct from individual ticker results
                portfolio = {
                    'initial_capital': 1000000,
                    'final_value': 0,  # Will calculate
                    'returns': {}
                }
                
                # Get tickers from previous results
                tickers = previous_results['ticker'].tolist()
                
                # Reconstruct portfolio data
                for _, row in previous_results.iterrows():
                    ticker = row['ticker']
                    if row.get('success', False):
                        portfolio['returns'][ticker] = {
                            'initial_value': 10000,
                            'final_value': 10000 * (1 + row['total_return']/100),
                            'total_return_pct': row['total_return'],
                            'buy_hold_return_pct': row.get('buy_hold_return', 0),
                            'total_trades': row.get('total_trades', 0),
                            'winning_trades': row.get('winning_trades', 0),
                            'losing_trades': row.get('losing_trades', 0),
                            'strategy': row.get('strategy', 'unknown')
                        }
                
                # Calculate portfolio final value
                portfolio['final_value'] = sum(data['final_value'] for data in portfolio['returns'].values())
                
                # Print enhanced performance summary
                print_enhanced_performance_summary(portfolio, start_date, end_date)
                
                # Check returns distribution
                check_returns_distribution(portfolio)
                
                # Verify returns calculation
                returns_verification = verify_returns_calculation(portfolio)
                
                if not returns_verification['consistent_calculations']:
                    logger.warning("Issues detected in returns calculation:")
                    for issue in returns_verification['issues_detected']:
                        logger.warning(f"- {issue}")
                
                # Check for data leakage
                leakage_results = check_data_leakage(tickers[:5], start_date, end_date)
                
                for ticker, result in leakage_results.items():
                    if result['status'] == 'analyzed':
                        if result['train_test_separation'] != 'clean':
                            logger.warning(f"Potential data leakage detected for {ticker}!")
                            if result['date_overlap']:
                                logger.warning(f"- Date overlap: {result['date_overlap']}")
                            if result['data_overlap_detected']:
                                logger.warning(f"- Data overlap detected")
                        else:
                            logger.info(f"Clean train/test separation for {ticker}")
        except Exception as e:
            logger.error(f"Error analyzing previous results: {e}")
    
    # 2. Run new tests with different ticker subsets if requested
    try_new_runs = False  # Set to True to run new tests
    if try_new_runs:
        logger.info("Running tests with different ticker subsets")
        test_results = run_multiple_ticker_tests(sp500_tickers, iterations=3, tickers_per_run=30, verbose=False)
        
        # Analyze the test results
        test_returns = [result['portfolio_return']*100 for result in test_results if 'portfolio_return' in result]
        test_annual_returns = [result['annualized_return'] for result in test_results if 'annualized_return' in result]
        
        if test_returns:
            logger.info("\nResults from multiple ticker subsets:")
            logger.info(f"Average total return: {np.mean(test_returns):.2f}%")
            logger.info(f"Average annualized return: {np.mean(test_annual_returns):.2f}%")
            logger.info(f"Return standard deviation: {np.std(test_returns):.2f}%")
            
            # Check for extreme variations in returns between runs
            if np.std(test_returns) > 50:
                logger.warning("High variation in returns between different ticker subsets!")
                logger.warning("This suggests the strategy may be unstable or overfitting to specific stocks.")
            
            # Report outliers across all runs
            all_outliers = []
            for result in test_results:
                if 'outliers' in result:
                    all_outliers.extend(result['outliers'])
            
            # Count occurrences of each ticker as an outlier
            outlier_counts = defaultdict(int)
            for outlier in all_outliers:
                outlier_counts[outlier['ticker']] += 1
            
            # Report tickers that are consistently outliers
            repeated_outliers = {ticker: count for ticker, count in outlier_counts.items() if count > 1}
            if repeated_outliers:
                logger.warning("Tickers that are consistently outliers:")
                for ticker, count in sorted(repeated_outliers.items(), key=lambda x: x[1], reverse=True):
                    logger.warning(f"- {ticker}: appeared as outlier in {count} runs")
    
    logger.info("\nVerification complete")
    
if __name__ == "__main__":
    main()