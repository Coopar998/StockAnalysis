"""
Stock Price Prediction System - Performance Evaluation
---------------------------------------------------
This file handles portfolio performance tracking and visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def calculate_portfolio_performance(portfolio, output_dir):
    """
    Calculate overall portfolio performance and save visualizations.
    
    Args:
        portfolio: Portfolio dictionary with all trading data
        output_dir: Directory to save outputs
    """
    print("\n==== PORTFOLIO PERFORMANCE SUMMARY ====")
    
    # Calculate portfolio metrics across all tickers
    total_initial = portfolio['initial_capital']
    total_positions_value = 0
    
    # Get final values for all positions in the portfolio
    ticker_returns = []
    for ticker, data in portfolio['returns'].items():
        ticker_return = data['total_return_pct']
        
        # Create dictionary for ticker performance data
        ticker_data = {
            'ticker': ticker,
            'return_pct': ticker_return,
            'final_value': data['final_value'],
            'trades': data['total_trades'],
            'buy_hold_return': data.get('buy_hold_return_pct', 0),
            'outperformance': ticker_return - data.get('buy_hold_return_pct', 0)
        }
        
        # Add strategy if available
        if 'strategy' in data:
            ticker_data['strategy'] = data['strategy']
        
        ticker_returns.append(ticker_data)
        total_positions_value += data['final_value']
    
    total_value = portfolio['available_cash'] + total_positions_value
    total_return_pct = ((total_value / total_initial) - 1) * 100
    
    print(f"Initial Capital: ${total_initial:,.2f}")
    print(f"Final Portfolio Value: ${total_value:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    
    # Sort tickers by return
    ticker_returns.sort(key=lambda x: x['return_pct'], reverse=True)
    
    # Create dataframe for ticker performance
    ticker_df = pd.DataFrame(ticker_returns)
    
    # Save portfolio summary to CSV
    summary_data = {
        'initial_capital': [total_initial],
        'final_value': [total_value],
        'total_return_pct': [total_return_pct],
        'cash_remaining': [portfolio['available_cash']],
        'total_stocks': [len(portfolio['returns'])]
    }
    
    # Add strategy counts if strategy column exists
    if 'strategy' in ticker_df.columns:
        summary_data['active_trading_stocks'] = [len(ticker_df[ticker_df['strategy'] == 'active_trading'])]
        summary_data['buy_hold_stocks'] = [len(ticker_df[ticker_df['strategy'] == 'buy_hold'])]
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "portfolio_summary.csv"), index=False)
    
    # Save ticker performance to CSV
    ticker_df.to_csv(os.path.join(output_dir, "ticker_performance.csv"), index=False)
    
    # Plot ticker returns
    plt.figure(figsize=(12, 8))
    colors = ['green' if r > 0 else 'red' for r in ticker_df['return_pct']]
    
    plt.bar(ticker_df['ticker'], ticker_df['return_pct'], color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Stock Returns by Ticker')
    plt.xlabel('Ticker')
    plt.ylabel('Return (%)')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(ticker_df['return_pct']):
        plt.text(i, v + (5 if v > 0 else -10), 
                f"{v:.1f}%", 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ticker_returns.png"))
    plt.close()
    
    # Plot strategy outperformance
    plt.figure(figsize=(12, 8))
    outperf_colors = ['green' if r > 0 else 'red' for r in ticker_df['outperformance']]
    
    plt.bar(ticker_df['ticker'], ticker_df['outperformance'], color=outperf_colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Strategy Outperformance vs Buy & Hold')
    plt.xlabel('Ticker')
    plt.ylabel('Outperformance (%)')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(ticker_df['outperformance']):
        plt.text(i, v + (5 if v > 0 else -10), 
                f"{v:.1f}%", 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_outperformance.png"))
    plt.close()
    
    # Plot number of trades by ticker
    plt.figure(figsize=(12, 6))
    
    # Check if strategy column exists
    if 'strategy' in ticker_df.columns:
        trade_colors = ['blue' if s == 'active_trading' else 'green' for s in ticker_df['strategy']]
    else:
        trade_colors = ['blue' for _ in ticker_df['ticker']]
    
    plt.bar(ticker_df['ticker'], ticker_df['trades'], color=trade_colors, alpha=0.7)
    plt.title('Number of Trades by Ticker')
    plt.xlabel('Ticker')
    plt.ylabel('Number of Trades')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(ticker_df['trades']):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ticker_trades.png"))
    plt.close()
    
    # Calculate total portfolio efficiency (return per trade)
    total_trades = ticker_df['trades'].sum()
    if total_trades > 0:
        efficiency = total_return_pct / total_trades
        print(f"Total Trades: {total_trades}")
        print(f"Return per Trade: {efficiency:.2f}%")
    else:
        print("No trades were executed.")
    
    # Add comparisons with market benchmarks
    print("\nComparison with Buy & Hold Strategy:")
    buy_hold_returns = ticker_df['buy_hold_return'].tolist()
    
    avg_buy_hold = sum(buy_hold_returns) / len(buy_hold_returns) if buy_hold_returns else 0
    print(f"Average Buy & Hold Return: {avg_buy_hold:.2f}%")
    print(f"Strategy Outperformance: {total_return_pct - avg_buy_hold:.2f}%")
    
    # Analyze which stocks were better for active trading vs buy & hold
    if 'strategy' in ticker_df.columns and len(ticker_df) > 0:
        print("\nStrategy Analysis:")
        active_df = ticker_df[ticker_df['strategy'] == 'active_trading']
        buyhold_df = ticker_df[ticker_df['strategy'] == 'buy_hold']
        
        if len(active_df) > 0:
            avg_active_return = active_df['return_pct'].mean()
            avg_active_outperf = active_df['outperformance'].mean()
            print(f"Active Trading stocks ({len(active_df)}): Avg return {avg_active_return:.2f}%, Avg outperformance {avg_active_outperf:.2f}%")
        
        if len(buyhold_df) > 0:
            avg_buyhold_return = buyhold_df['return_pct'].mean()
            print(f"Buy & Hold stocks ({len(buyhold_df)}): Avg return {avg_buyhold_return:.2f}%")
    
    return {
        'total_initial': total_initial,
        'total_final': total_value,
        'total_return_pct': total_return_pct,
        'total_trades': total_trades,
        'ticker_performance': ticker_df.to_dict('records')
    }

def create_strategy_comparison(results_df, output_dir):
    """Create a visualization comparing strategy performance to buy & hold"""
    plt.figure(figsize=(14, 8))
    
    # Calculate outperformance
    results_df['outperformance'] = results_df['total_return'] - results_df['buy_hold_return']
    results_df = results_df.sort_values('outperformance', ascending=False)
    
    # Plot side-by-side comparison
    width = 0.35
    x = np.arange(len(results_df))
    
    # Strategy returns
    plt.bar(x - width/2, results_df['total_return'], width, label='Strategy', color='blue', alpha=0.7)
    
    # Buy & Hold returns
    plt.bar(x + width/2, results_df['buy_hold_return'], width, label='Buy & Hold', color='green', alpha=0.7)
    
    # Add labels and formatting
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Ticker')
    plt.ylabel('Return (%)')
    plt.title('Strategy vs Buy & Hold Performance Comparison')
    plt.xticks(x, results_df['ticker'], rotation=45)
    plt.legend()
    
    # Add value labels
    for i, (strategy, buyhold) in enumerate(zip(results_df['total_return'], results_df['buy_hold_return'])):
        plt.text(i - width/2, strategy + (2 if strategy > 0 else -8), 
                f"{strategy:.1f}%", ha='center', va='bottom', rotation=90, fontsize=8)
        plt.text(i + width/2, buyhold + (2 if buyhold > 0 else -8), 
                f"{buyhold:.1f}%", ha='center', va='bottom', rotation=90, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_comparison.png"))
    plt.close()
    
    # Create a chart showing which strategy was used for each stock
    if 'strategy' in results_df.columns:
        plt.figure(figsize=(14, 6))
        
        # Convert strategy to numeric for coloring
        strategy_map = {'buy_hold': 1, 'active_trading': 0}
        if 'strategy' in results_df.columns:
            results_df['strategy_num'] = results_df['strategy'].map(lambda x: strategy_map.get(x, 0.5))
            
            # Create a colormap
            colors = ['blue', 'green']
            strategy_colors = [colors[int(s)] if s in [0, 1] else 'gray' for s in results_df['strategy_num']]
            
            # Plot the returns with strategy-based colors
            plt.bar(results_df['ticker'], results_df['total_return'], color=strategy_colors)
            
            # Create a custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', label='Active Trading'),
                Patch(facecolor='green', label='Buy & Hold')
            ]
            plt.legend(handles=legend_elements)
        else:
            plt.bar(results_df['ticker'], results_df['total_return'])
        
        plt.title('Returns by Strategy Type')
        plt.xlabel('Ticker')
        plt.ylabel('Return (%)')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add value labels
        for i, v in enumerate(results_df['total_return']):
            plt.text(i, v + (2 if v > 0 else -5), 
                    f"{v:.1f}%", ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "returns_by_strategy.png"))
        plt.close()