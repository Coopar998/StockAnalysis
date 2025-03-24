"""
Stock Price Prediction System - Trading Execution
----------------------------------------------
This file contains functions for executing trading strategies based on signals.
"""

import numpy as np
import logging
from collections import deque
from utils.config_reader import get_config

def execute_buy_hold_strategy(ticker, test_dates, prices, initial_capital):
    """
    Execute a simple buy & hold strategy
    
    Args:
        ticker: Stock ticker symbol
        test_dates: List of dates
        prices: Array of prices
        initial_capital: Initial capital amount
        
    Returns:
        Dict with trading results
    """
    # Calculate position and final value
    position = initial_capital / prices[0]
    final_value = position * prices[-1]
    
    # Calculate return
    total_return = ((final_value / initial_capital) - 1) * 100
    
    # Create trades list
    trades = [('buy', test_dates[0], prices[0]), ('sell', test_dates[-1], prices[-1])]
    
    # Trading statistics
    winning_trades = 1 if total_return > 0 else 0
    losing_trades = 1 if total_return <= 0 else 0
    win_rate = 100 if winning_trades > 0 else 0
    
    # For buy and hold, avg_win and avg_loss are the same as total_return
    avg_win = total_return if winning_trades > 0 else 0
    avg_loss = abs(total_return) if losing_trades > 0 else 0
    win_loss_ratio = avg_win / max(0.01, avg_loss)
    profit_factor = avg_win / max(0.01, avg_loss)
    
    return {
        'total_return': total_return,
        'final_value': final_value,
        'trades': trades,
        'total_trades': 1,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'profit_factor': profit_factor
    }

def execute_trading_strategy(ticker, signals, test_dates, prices, initial_capital, 
                             rsi_values, trend_metrics, verbose=False):
    """
    Execute an active trading strategy based on signals with risk management
    
    Args:
        ticker: Stock ticker symbol
        signals: List of (action, date, price) signals
        test_dates: All test dates
        prices: All prices
        initial_capital: Initial capital amount
        rsi_values: RSI indicator values
        trend_metrics: Dict with trend metrics
        verbose: Whether to print detailed output
    
    Returns:
        Dict with trading results
    """
    logger = logging.getLogger('stock_prediction')
    config = get_config()
    
    # Get position sizing parameters from config
    base_position_pct = config.get('trading', 'position_sizing', 'base_position_pct', default=0.3)
    max_position_pct = config.get('trading', 'position_sizing', 'max_position_pct', default=0.5)  
    min_position_pct = config.get('trading', 'position_sizing', 'min_position_pct', default=0.1)
    
    # Get risk management parameters from config
    stop_loss_pct = config.get('trading', 'risk_management', 'stop_loss_pct', default=5.0)
    trailing_stop_pct = config.get('trading', 'risk_management', 'trailing_stop_pct', default=6.0)
    take_profit_pct = config.get('trading', 'risk_management', 'take_profit_pct', default=10.0)
    max_holding_period = config.get('trading', 'risk_management', 'max_holding_period', default=10)
    min_holding_period = config.get('trading', 'risk_management', 'min_holding_period', default=2)
    
    # Initialize trading variables
    capital = initial_capital
    position = 0
    trades = []
    
    # Stats for tracking trade performance
    winning_trades = 0
    losing_trades = 0
    breakeven_trades = 0
    total_win_pct = 0
    total_loss_pct = 0
    longest_win_streak = 0
    longest_lose_streak = 0
    current_win_streak = 0
    current_lose_streak = 0
    entry_price = None
    
    # Track highest price since entry for trailing stop
    highest_since_entry = None
    
    # Track holding periods
    last_trade_day = None
    days_in_position = 0
    
    # Recent trade outcomes for adaptive strategy
    recent_trades = deque(maxlen=5)  # Store recent trade results
    
    # Create date-to-index mapping for faster lookup
    date_to_idx = {date: i for i, date in enumerate(test_dates)}
    
    # Execute the trading strategy with signals
    for i in range(len(prices)):
        current_date = test_dates[i]
        current_price = prices[i]
        
        # If we're in a position, update days counter and check for exits
        if position > 0:
            days_in_position += 1
            
            # Update highest price since entry for trailing stop
            if highest_since_entry is None or current_price > highest_since_entry:
                highest_since_entry = current_price
            
            # Check stop loss
            if entry_price is not None:
                price_change_pct = ((current_price / entry_price) - 1) * 100
                
                # Adaptive stops based on recent performance
                current_stop_loss = stop_loss_pct
                if len(recent_trades) >= 3:
                    # Count recent losses
                    recent_losses = sum(1 for trade in recent_trades if trade < 0)
                    if recent_losses >= 2:
                        # Tighten stops after multiple losses
                        current_stop_loss = stop_loss_pct * 0.85
                
                # Fixed stop loss
                if price_change_pct < -current_stop_loss:
                    if verbose:
                        logger.info(f"Stop loss triggered at {current_price:.2f} (entry: {entry_price:.2f})")
                    capital = position * current_price
                    trades.append(('sell', current_date, current_price))
                    
                    # Update statistics
                    losing_trades += 1
                    total_loss_pct += abs(price_change_pct)
                    current_lose_streak += 1
                    current_win_streak = 0
                    longest_lose_streak = max(longest_lose_streak, current_lose_streak)
                    
                    # Track recent trade outcome
                    recent_trades.append(price_change_pct)
                    
                    # Reset position
                    position = 0
                    entry_price = None
                    highest_since_entry = None
                    days_in_position = 0
                    last_trade_day = current_date
                    continue
                
                # Adaptive trailing stop
                current_trailing_stop = trailing_stop_pct
                if price_change_pct > 4.0:
                    # Tighten trailing stop when in good profit
                    current_trailing_stop = trailing_stop_pct * 0.75
                
                # Trailing stop
                if highest_since_entry is not None:
                    drawdown_pct = ((current_price / highest_since_entry) - 1) * 100
                    if price_change_pct > 2.5 and drawdown_pct < -current_trailing_stop:
                        if verbose:
                            logger.info(f"Trailing stop triggered at {current_price:.2f} (high: {highest_since_entry:.2f})")
                        capital = position * current_price
                        trades.append(('sell', current_date, current_price))
                        
                        # Determine if this is a win or loss overall
                        if price_change_pct > 0:
                            winning_trades += 1
                            total_win_pct += price_change_pct
                            current_win_streak += 1
                            current_lose_streak = 0
                            longest_win_streak = max(longest_win_streak, current_win_streak)
                        else:
                            losing_trades += 1
                            total_loss_pct += abs(price_change_pct)
                            current_lose_streak += 1
                            current_win_streak = 0
                            longest_lose_streak = max(longest_lose_streak, current_lose_streak)
                        
                        # Track recent trade outcome
                        recent_trades.append(price_change_pct)
                        
                        # Reset position
                        position = 0
                        entry_price = None
                        highest_since_entry = None
                        days_in_position = 0
                        last_trade_day = current_date
                        continue
                
                # Adaptive take profit based on trend strength
                current_take_profit = take_profit_pct
                if trend_metrics['trend_quality'] > 0.65 and trend_metrics['trend_direction'] > 0:
                    # Higher targets in strong uptrends
                    current_take_profit = take_profit_pct * 1.4
                
                # Take profit
                if price_change_pct > current_take_profit:
                    if verbose:
                        logger.info(f"Take profit triggered at {current_price:.2f} (entry: {entry_price:.2f})")
                    capital = position * current_price
                    trades.append(('sell', current_date, current_price))
                    
                    # Update statistics
                    winning_trades += 1
                    total_win_pct += price_change_pct
                    current_win_streak += 1
                    current_lose_streak = 0
                    longest_win_streak = max(longest_win_streak, current_win_streak)
                    
                    # Track recent trade outcome
                    recent_trades.append(price_change_pct)
                    
                    # Reset position
                    position = 0
                    entry_price = None
                    highest_since_entry = None
                    days_in_position = 0
                    last_trade_day = current_date
                    continue
                
                # Adaptive time-based exit
                # Exit sooner in deteriorating conditions
                current_max_holding = max_holding_period
                if i < len(rsi_values) and rsi_values[i] > 75:
                    # Exit sooner in overbought conditions
                    current_max_holding = int(max_holding_period * 0.65)
                
                # Time-based exit (max holding period)
                if days_in_position >= current_max_holding and price_change_pct < 3.0:
                    if verbose:
                        logger.info(f"Time-based exit triggered after {days_in_position} days at {current_price:.2f}")
                    capital = position * current_price
                    trades.append(('sell', current_date, current_price))
                    
                    # Determine if this is a win or loss
                    if price_change_pct > 0:
                        winning_trades += 1
                        total_win_pct += price_change_pct
                        current_win_streak += 1
                        current_lose_streak = 0
                        longest_win_streak = max(longest_win_streak, current_win_streak)
                    else:
                        losing_trades += 1
                        total_loss_pct += abs(price_change_pct)
                        current_lose_streak += 1
                        current_win_streak = 0
                        longest_lose_streak = max(longest_lose_streak, current_lose_streak)
                    
                    # Track recent trade outcome
                    recent_trades.append(price_change_pct)
                    
                    # Reset position
                    position = 0
                    entry_price = None
                    highest_since_entry = None
                    days_in_position = 0
                    last_trade_day = current_date
                    continue
        
        # Check if this date has a signal
        signal_match = None
        for signal in signals:
            signal_date = signal[1]
            if signal_date == current_date:
                signal_match = signal
                break
        
        if signal_match:
            action, date, signal_price = signal_match
            
            # Skip if we haven't waited long enough since the last trade
            if last_trade_day is not None and hasattr(date, 'toordinal') and hasattr(last_trade_day, 'toordinal'):
                days_since_last_trade = (date.toordinal() - last_trade_day.toordinal())
                if days_since_last_trade < min_holding_period:
                    continue
            
            # Buy signal with enhanced entry conditions and position sizing
            if action == 'buy' and position == 0 and capital > 0:
                # Check additional entry conditions
                proceed_with_buy = True
                
                # Check RSI for extreme conditions
                if i < len(rsi_values) and rsi_values[i] > 80:
                    proceed_with_buy = False  # Don't buy in extreme overbought conditions
                
                # Check recent performance
                if current_lose_streak >= 4:
                    # After 4 consecutive losses, be more selective
                    if trend_metrics['trend_quality'] < 0.5 or recent_accuracy < 0.52:
                        proceed_with_buy = False
                
                if proceed_with_buy:
                    # Calculate position size based on comprehensive factors
                    position_modifier = 1.0
                    
                    # Adjust for recent performance
                    if len(recent_trades) > 0:
                        recent_win_rate = sum(1 for trade in recent_trades if trade > 0) / len(recent_trades)
                        if recent_win_rate > 0.6:
                            position_modifier *= 1.25
                        elif recent_win_rate < 0.4:
                            position_modifier *= 0.85
                    
                    # Adjust for trend strength
                    if trend_metrics['trend_quality'] > 0.65:
                        position_modifier *= 1.15
                    
                    # Adjust for win/loss streak
                    if current_win_streak >= 2:
                        position_modifier *= 1.15
                    elif current_lose_streak >= 2:
                        position_modifier *= 0.75
                    
                    # Cap position size
                    final_position_pct = min(max(min_position_pct, base_position_pct * position_modifier), max_position_pct)
                    
                    # Calculate shares to buy
                    position = (capital * final_position_pct) / current_price
                    capital -= position * current_price  # Remaining capital
                    
                    trades.append(('buy', date, current_price))
                    entry_price = current_price
                    highest_since_entry = current_price  # Initialize highest price
                    days_in_position = 0
                    last_trade_day = date
                    
            # Sell signal and in position
            elif action == 'sell' and position > 0:
                # Calculate return on this trade
                trade_return_pct = ((current_price / entry_price) - 1) * 100
                
                # Execute sell
                trade_value = position * current_price
                capital += trade_value
                
                # Update trade statistics
                if trade_return_pct > 0:
                    winning_trades += 1
                    total_win_pct += trade_return_pct
                    current_win_streak += 1
                    current_lose_streak = 0
                    longest_win_streak = max(longest_win_streak, current_win_streak)
                elif trade_return_pct < 0:
                    losing_trades += 1
                    total_loss_pct += abs(trade_return_pct)
                    current_lose_streak += 1
                    current_win_streak = 0
                    longest_lose_streak = max(longest_lose_streak, current_lose_streak)
                else:
                    breakeven_trades += 1
                    current_win_streak = 0
                    current_lose_streak = 0
                
                # Track recent trade outcome
                recent_trades.append(trade_return_pct)
                
                position = 0
                trades.append(('sell', date, current_price))
                last_trade_day = date
                entry_price = None
                highest_since_entry = None
                days_in_position = 0
    
    # Close any open position at the end of the testing period
    if position > 0:
        final_price = prices[-1]
        
        # Calculate final trade profit/loss if we still have open position
        if entry_price is not None:
            trade_return_pct = ((final_price / entry_price) - 1) * 100
            
            # Update trade statistics
            if trade_return_pct > 0:
                winning_trades += 1
                total_win_pct += trade_return_pct
            elif trade_return_pct < 0:
                losing_trades += 1
                total_loss_pct += abs(trade_return_pct)
            else:
                breakeven_trades += 1
        
        capital += position * final_price
        trades.append(('sell', test_dates[-1], final_price))
        position = 0
    
    final_value = capital
    
    # Calculate performance metrics
    total_return = ((final_value / initial_capital) - 1) * 100
    total_trades = len(trades) // 2  # Each buy/sell pair is one trade
    
    # Calculate win rate
    win_rate = winning_trades / max(1, winning_trades + losing_trades) * 100
    
    # Calculate average win and loss
    avg_win = total_win_pct / max(1, winning_trades)
    avg_loss = total_loss_pct / max(1, losing_trades)
    
    # Calculate win/loss ratio and profit factor
    win_loss_ratio = avg_win / max(0.01, avg_loss)  # Avoid division by zero
    profit_factor = total_win_pct / max(0.01, total_loss_pct)  # Avoid division by zero
    
    return {
        'total_return': total_return,
        'final_value': final_value,
        'trades': trades,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'profit_factor': profit_factor,
        'breakeven_trades': breakeven_trades
    }