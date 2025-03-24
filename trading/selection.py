"""
Stock Price Prediction System - Stock Selection
--------------------------------------------
This file contains functions for selecting stocks suitable for different
trading strategies based on their characteristics.
"""

from utils.config_reader import get_config

def is_good_for_active_trading(ticker, trend_metrics, volatility_metrics, prediction_accuracy):
    """
    Determine if a stock is good for active trading based on its characteristics
    
    Args:
        ticker: Stock ticker symbol
        trend_metrics: Dict with trend metrics
        volatility_metrics: Dict with volatility metrics
        prediction_accuracy: Model prediction accuracy
        
    Returns:
        Boolean indicating if stock is good for active trading and a confidence score
    """
    config = get_config()
    
    # Start with base score
    score = 0.0
    
    # Check sector-specific factors (tech and energy are often better for active trading)
    tech_tickers = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", "CRM"]
    energy_tickers = ["XOM", "CVX", "COP"]
    finance_tickers = ["JPM", "BAC", "GS", "MS", "V", "MA", "AXP"]
    
    if ticker in tech_tickers:
        score += 0.15
    elif ticker in energy_tickers:
        score += 0.1
    elif ticker in finance_tickers:
        score += 0.05
    
    # Add points for prediction accuracy (critical factor)
    if prediction_accuracy > 0.59:
        score += 0.5
    elif prediction_accuracy > 0.55:
        score += 0.4
    elif prediction_accuracy > 0.52:
        score += 0.3
    elif prediction_accuracy > 0.5:
        score += 0.2
    else:
        score -= 0.1
    
    # Add points for good trend quality
    if trend_metrics['trend_quality'] > 0.7:
        score += 0.25
    elif trend_metrics['trend_quality'] > 0.6:
        score += 0.2
    elif trend_metrics['trend_quality'] > 0.5:
        score += 0.15
    elif trend_metrics['trend_quality'] > 0.4:
        score += 0.1
    
    # Add points for appropriate volatility
    vol_regime = volatility_metrics['volatility_regime']
    if vol_regime == 'high':
        score += 0.2
    elif vol_regime == 'medium':
        score += 0.15
    elif vol_regime == 'low':
        score += 0.05
    elif vol_regime == 'very_high':
        score += 0.05
    elif vol_regime == 'very_low':
        score -= 0.1
    
    # Add points for decreasing volatility (more stable conditions)
    if volatility_metrics['volatility_trend'] == 'decreasing':
        score += 0.1
    
    # Get threshold from config
    use_buy_hold_threshold = config.get('trading', 'strategy_selection', 'use_buy_hold_threshold', default=0.55)
    
    # Is it good for active trading?
    is_good = score >= use_buy_hold_threshold
    
    return is_good, score

def rank_stocks_for_trading(stock_data, prediction_results):
    """
    Rank stocks based on their suitability for active trading
    
    Args:
        stock_data: Dictionary of stock dataframes
        prediction_results: Dictionary of prediction results
    
    Returns:
        DataFrame with ranked stocks and scores
    """
    import pandas as pd
    import numpy as np
    
    ranking_data = []
    
    for ticker, data in stock_data.items():
        if ticker not in prediction_results:
            continue
        
        result = prediction_results[ticker]
        
        if not result.get('success', False):
            continue
        
        # Extract key metrics
        trend_quality = result.get('trend_quality', 0)
        volatility_regime = result.get('volatility_regime', 'unknown')
        prediction_accuracy = result.get('prediction_accuracy', 0)
        
        # Convert volatility regime to numeric score
        vol_scores = {
            'very_low': 0.2,
            'low': 0.5,
            'medium': 1.0,
            'high': 0.8,
            'very_high': 0.3,
            'unknown': 0.0
        }
        vol_score = vol_scores.get(volatility_regime, 0)
        
        # Calculate a trading score
        trading_score = (
            prediction_accuracy * 0.5 +
            trend_quality * 0.3 +
            vol_score * 0.2
        )
        
        # Determine if suitable for active trading
        is_suitable, suitability_score = is_good_for_active_trading(
            ticker, 
            {'trend_quality': trend_quality, 'trend_direction': 1 if result.get('buy_hold_return', 0) > 0 else -1},
            {'volatility_regime': volatility_regime, 'volatility_trend': 'unknown'},
            prediction_accuracy
        )
        
        # Add to ranking data
        ranking_data.append({
            'ticker': ticker,
            'trading_score': trading_score,
            'prediction_accuracy': prediction_accuracy,
            'trend_quality': trend_quality,
            'volatility': volatility_regime,
            'suitable_for_active': is_suitable,
            'suitability_score': suitability_score,
            'strategy': result.get('strategy', 'unknown')
        })
    
    # Create DataFrame and sort by trading score
    if ranking_data:
        df = pd.DataFrame(ranking_data)
        return df.sort_values('trading_score', ascending=False)
    else:
        return pd.DataFrame(columns=['ticker', 'trading_score', 'prediction_accuracy', 
                                     'trend_quality', 'volatility', 'suitable_for_active',
                                     'suitability_score', 'strategy'])

def select_stocks_for_portfolio(ranked_stocks, max_stocks=10, min_active_trading=5):
    """
    Select top stocks for a portfolio with a mix of active trading and buy & hold
    
    Args:
        ranked_stocks: DataFrame with ranked stocks
        max_stocks: Maximum number of stocks to select
        min_active_trading: Minimum number of active trading stocks
        
    Returns:
        List of selected tickers
    """
    if len(ranked_stocks) == 0:
        return []
    
    # Get active trading candidates
    active_candidates = ranked_stocks[ranked_stocks['suitable_for_active']].copy()
    
    # Get buy & hold candidates
    buyhold_candidates = ranked_stocks[~ranked_stocks['suitable_for_active']].copy()
    
    # Select top active trading stocks
    active_count = min(len(active_candidates), min_active_trading)
    selected_active = active_candidates.nlargest(active_count, 'trading_score')
    
    # Fill remaining slots with buy & hold stocks
    remaining_slots = max_stocks - active_count
    if remaining_slots > 0 and len(buyhold_candidates) > 0:
        selected_buyhold = buyhold_candidates.nlargest(remaining_slots, 'trading_score')
        
        # Combine selections
        selected = pd.concat([selected_active, selected_buyhold])
    else:
        selected = selected_active
    
    # Return the ticker list
    return selected['ticker'].tolist()