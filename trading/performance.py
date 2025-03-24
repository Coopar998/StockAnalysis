"""
Stock Price Prediction System - Performance Evaluation
---------------------------------------------------
This file handles portfolio performance tracking and visualization.
Updated to use configuration settings and fix portfolio valuation discrepancies.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from utils.config_reader import get_config

def calculate_portfolio_performance(portfolio, output_dir):
    """
    Calculate overall portfolio performance and save visualizations.
    
    Args:
        portfolio: Portfolio dictionary with all trading data
        output_dir: Directory to save outputs
    """
    print("\n==== PORTFOLIO PERFORMANCE SUMMARY ====")
    config = get_config()
    
    # Calculate portfolio metrics across all tickers
    total_initial = portfolio.get('initial_capital', config.get('portfolio', 'initial_capital', default=1000000))
    total_positions_value = 0
    
    # Get final values for all positions in the portfolio
    ticker_returns = []
    
    # Track overall trading statistics
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    breakeven_trades = 0
    total_win_amount = 0
    total_loss_amount = 0
    
    # Process each ticker
    for ticker, data in portfolio.get('returns', {}).items():
        ticker_return = data.get('total_return_pct', 0)
        
        # Create dictionary for ticker performance data
        ticker_data = {
            'ticker': ticker,
            'return_pct': ticker_return,
            'final_value': data.get('final_value', 0),
            'trades': data.get('total_trades', 0),
            'buy_hold_return': data.get('buy_hold_return_pct', 0),
            'outperformance': ticker_return - data.get('buy_hold_return_pct', 0)
        }
        
        # Extract detailed trade information if available
        if 'trades' in data and isinstance(data['trades'], list):
            trades_list = data['trades']
            
            # Process trades for this ticker
            ticker_winning_trades = 0
            ticker_losing_trades = 0
            ticker_breakeven_trades = 0
            ticker_win_amount = 0
            ticker_loss_amount = 0
            
            # Track buy/sell pairs
            buy_price = None
            
            for trade in trades_list:
                action = trade[0]
                price = trade[2]
                
                if action == 'buy':
                    buy_price = price
                elif action == 'sell' and buy_price is not None:
                    # Calculate profit/loss for this trade
                    trade_pl = price - buy_price
                    trade_pl_pct = (price / buy_price - 1) * 100
                    
                    if trade_pl > 0:
                        ticker_winning_trades += 1
                        ticker_win_amount += trade_pl_pct
                    elif trade_pl < 0:
                        ticker_losing_trades += 1
                        ticker_loss_amount += abs(trade_pl_pct)
                    else:
                        ticker_breakeven_trades += 1
                    
                    buy_price = None  # Reset for next trade pair
            
            # Add trade statistics to ticker data
            ticker_data['winning_trades'] = ticker_winning_trades
            ticker_data['losing_trades'] = ticker_losing_trades
            ticker_data['breakeven_trades'] = ticker_breakeven_trades
            ticker_data['win_rate'] = ticker_winning_trades / max(1, ticker_winning_trades + ticker_losing_trades) * 100
            ticker_data['avg_win_pct'] = ticker_win_amount / max(1, ticker_winning_trades)
            ticker_data['avg_loss_pct'] = ticker_loss_amount / max(1, ticker_losing_trades)
            
            # Update overall statistics
            winning_trades += ticker_winning_trades
            losing_trades += ticker_losing_trades
            breakeven_trades += ticker_breakeven_trades
            total_win_amount += ticker_win_amount
            total_loss_amount += ticker_loss_amount
            total_trades += data.get('total_trades', 0)
        
        # Add strategy if available
        if 'strategy' in data:
            ticker_data['strategy'] = data['strategy']
        
        ticker_returns.append(ticker_data)
        total_positions_value += data.get('final_value', 0)
    
    # Calculate total portfolio value
    available_cash = portfolio.get('available_cash', 0)
    total_value = available_cash + total_positions_value
    
    # Update portfolio value in the portfolio dictionary to ensure consistency
    portfolio['final_value'] = total_value
    
    total_return_pct = ((total_value / total_initial) - 1) * 100
    
    # Calculate overall trading statistics
    win_rate = winning_trades / max(1, winning_trades + losing_trades) * 100
    avg_win_pct = total_win_amount / max(1, winning_trades)
    avg_loss_pct = total_loss_amount / max(1, losing_trades)
    win_loss_ratio = avg_win_pct / max(0.01, avg_loss_pct)  # Avoid division by zero
    profit_factor = total_win_amount / max(0.01, total_loss_amount)  # Avoid division by zero
    
    print(f"Initial Capital: ${total_initial:,.2f}")
    print(f"Final Portfolio Value: ${total_value:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    
    # Display trade statistics
    print(f"\nTrading Statistics:")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades: {losing_trades} ({100 - win_rate:.2f}%)")
    print(f"Breakeven Trades: {breakeven_trades}")
    print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    print(f"Average Win: {avg_win_pct:.2f}%")
    print(f"Average Loss: {avg_loss_pct:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Return per Trade: {total_return_pct / max(1, total_trades):.2f}%")
    
    # Sort tickers by return
    ticker_returns.sort(key=lambda x: x['return_pct'], reverse=True)
    
    # Create dataframe for ticker performance
    ticker_df = pd.DataFrame(ticker_returns)
    
    # Save portfolio summary to CSV
    summary_data = {
        'initial_capital': [total_initial],
        'final_value': [total_value],
        'total_return_pct': [total_return_pct],
        'cash_remaining': [portfolio.get('available_cash', 0)],
        'total_stocks': [len(portfolio.get('returns', {}))],
        'total_trades': [total_trades],
        'winning_trades': [winning_trades],
        'losing_trades': [losing_trades],
        'win_rate': [win_rate],
        'avg_win_pct': [avg_win_pct],
        'avg_loss_pct': [avg_loss_pct],
        'win_loss_ratio': [win_loss_ratio],
        'profit_factor': [profit_factor]
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
    
    # Plot win rate by ticker (for stocks with trades)
    if 'win_rate' in ticker_df.columns:
        # Filter tickers with actual trades
        trading_df = ticker_df[ticker_df['trades'] > 0].copy()
        if len(trading_df) > 0:
            plt.figure(figsize=(12, 8))
            bar_colors = ['green' if r > 50 else 'red' for r in trading_df['win_rate']]
            
            plt.bar(trading_df['ticker'], trading_df['win_rate'], color=bar_colors)
            plt.axhline(y=50, color='black', linestyle='--', alpha=0.5)
            plt.title('Win Rate by Ticker')
            plt.xlabel('Ticker')
            plt.ylabel('Win Rate (%)')
            plt.xticks(rotation=45)
            
            for i, v in enumerate(trading_df['win_rate']):
                plt.text(i, v + 3, f"{v:.1f}%", ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "ticker_win_rates.png"))
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
    
    # Add comparisons with market benchmarks
    print("\nComparison with Buy & Hold Strategy:")
    buy_hold_returns = ticker_df['buy_hold_return'].tolist()
    
    avg_buy_hold = sum(buy_hold_returns) / len(buy_hold_returns) if buy_hold_returns else 0
    print(f"Average Buy & Hold Return: {avg_buy_hold:.2f}%")
    print(f"Strategy Outperformance: {total_return_pct - avg_buy_hold:.2f}%")
    
    # Buy & Hold detailed stats
    buy_hold_positive = sum(1 for r in buy_hold_returns if r > 0)
    buy_hold_negative = sum(1 for r in buy_hold_returns if r <= 0)
    print(f"Buy & Hold Winners: {buy_hold_positive} ({buy_hold_positive/max(1, len(buy_hold_returns))*100:.2f}%)")
    print(f"Buy & Hold Losers: {buy_hold_negative} ({buy_hold_negative/max(1, len(buy_hold_returns))*100:.2f}%)")
    
    # Analyze which stocks were better for active trading vs buy & hold
    if 'strategy' in ticker_df.columns and len(ticker_df) > 0:
        print("\nStrategy Analysis:")
        active_df = ticker_df[ticker_df['strategy'] == 'active_trading']
        buyhold_df = ticker_df[ticker_df['strategy'] == 'buy_hold']
        
        if len(active_df) > 0:
            avg_active_return = active_df['return_pct'].mean()
            avg_active_outperf = active_df['outperformance'].mean()
            
            # Calculate active trading win statistics if available
            if 'win_rate' in active_df.columns:
                avg_active_win_rate = active_df['win_rate'].mean()
                avg_active_win_loss_ratio = active_df['avg_win_pct'].mean() / max(0.01, active_df['avg_loss_pct'].mean())
                print(f"Active Trading stocks ({len(active_df)}): Avg return {avg_active_return:.2f}%, Avg outperformance {avg_active_outperf:.2f}%")
                print(f"  Win Rate: {avg_active_win_rate:.2f}%, Win/Loss Ratio: {avg_active_win_loss_ratio:.2f}")
            else:
                print(f"Active Trading stocks ({len(active_df)}): Avg return {avg_active_return:.2f}%, Avg outperformance {avg_active_outperf:.2f}%")
        
        if len(buyhold_df) > 0:
            avg_buyhold_return = buyhold_df['return_pct'].mean()
            print(f"Buy & Hold stocks ({len(buyhold_df)}): Avg return {avg_buyhold_return:.2f}%")
    
    return {
        'total_initial': total_initial,
        'total_final': total_value,
        'total_return_pct': total_return_pct,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'win_loss_ratio': win_loss_ratio,
        'profit_factor': profit_factor,
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