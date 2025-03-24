"""
Stock Price Prediction System - Market Analysis
---------------------------------------------
This file contains functions for analyzing price trends, volatility,
and other market characteristics.
"""

import numpy as np
from scipy.signal import argrelextrema
from utils.config_reader import get_config

def evaluate_trend_quality(prices, window=20):
    """
    Evaluate the quality and strength of a price trend
    
    Args:
        prices: Array of price data
        window: Window size for trend calculation
        
    Returns:
        Dict with trend metrics
    """
    if len(prices) < window * 2:
        return {'trend_strength': 0, 'trend_direction': 0, 'trend_quality': 0}
        
    # Calculate moving average
    ma = np.convolve(prices, np.ones(window)/window, mode='valid')
    
    # Calculate slope of MA (trend direction and strength)
    ma_diff = np.diff(ma)
    trend_direction = 1 if np.mean(ma_diff[-window:]) > 0 else -1
    
    # Trend strength: normalized slope
    trend_strength = abs(np.mean(ma_diff[-window:]) / np.mean(prices[-window:]))
    
    # Trend quality: R-squared of linear fit to recent prices
    x = np.arange(window)
    y = prices[-window:]
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate correlation coefficient
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    r = numerator / denominator if denominator != 0 else 0
    
    # R-squared
    r_squared = r**2
    
    # Count local extrema to measure choppiness
    local_maxima = len(argrelextrema(prices[-window*2:], np.greater, order=3)[0])
    local_minima = len(argrelextrema(prices[-window*2:], np.less, order=3)[0])
    choppiness = (local_maxima + local_minima) / (window * 2)
    
    # Higher quality = higher r-squared and lower choppiness
    trend_quality = r_squared * (1 - choppiness)
    
    return {
        'trend_strength': float(trend_strength),
        'trend_direction': int(trend_direction),
        'trend_quality': float(trend_quality),
        'r_squared': float(r_squared),
        'choppiness': float(choppiness)
    }

def evaluate_volatility(prices, window=20):
    """
    Evaluate the volatility characteristics of a stock
    
    Args:
        prices: Array of price data
        window: Window size for calculation
        
    Returns:
        Dict with volatility metrics
    """
    if len(prices) < window:
        return {'volatility': 0, 'normalized_volatility': 0, 'volatility_regime': 'unknown'}
        
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    
    # Recent volatility (standard deviation of returns)
    recent_volatility = np.std(returns[-window:])
    
    # Normalized volatility (volatility / average price)
    normalized_volatility = recent_volatility / np.mean(prices[-window:])
    
    # Longer-term volatility
    long_window = min(window * 3, len(returns))
    long_volatility = np.std(returns[-long_window:])
    
    # Volatility regime
    if normalized_volatility < 0.01:
        regime = 'very_low'
    elif normalized_volatility < 0.02:
        regime = 'low'
    elif normalized_volatility < 0.03:
        regime = 'medium'
    elif normalized_volatility < 0.04:
        regime = 'high'
    else:
        regime = 'very_high'
    
    # Volatility trend (increasing or decreasing)
    half_window = window // 2
    if len(returns) >= window + half_window:
        prev_volatility = np.std(returns[-(window+half_window):-half_window])
        vol_trend = 'increasing' if recent_volatility > prev_volatility else 'decreasing'
    else:
        vol_trend = 'unknown'
    
    return {
        'volatility': float(recent_volatility),
        'normalized_volatility': float(normalized_volatility),
        'volatility_regime': regime,
        'volatility_trend': vol_trend,
        'long_term_volatility': float(long_volatility)
    }

def detect_support_resistance(prices, lookback=50, threshold=0.02):
    """
    Detect support and resistance levels in price data
    
    Args:
        prices: Array of price data
        lookback: Period to consider for support/resistance
        threshold: Proximity threshold to consider a level broken
        
    Returns:
        Dict with support and resistance levels
    """
    if len(prices) < lookback:
        return {'support': None, 'resistance': None}
    
    # Find recent high as resistance
    recent_high = np.max(prices[-lookback:])
    
    # Find recent low as support
    recent_low = np.min(prices[-lookback:])
    
    # Check if current price is near levels
    current_price = prices[-1]
    
    # Percentage distance from levels
    dist_to_resistance = (recent_high - current_price) / current_price
    dist_to_support = (current_price - recent_low) / current_price
    
    # Check if price is testing a level
    near_resistance = dist_to_resistance < threshold
    near_support = dist_to_support < threshold
    
    # Check if price has recently broken a level
    broke_resistance = False
    broke_support = False
    
    if len(prices) > 5:
        # Price crossed above previous resistance
        if prices[-5] < recent_high and current_price > recent_high:
            broke_resistance = True
        
        # Price crossed below previous support
        if prices[-5] > recent_low and current_price < recent_low:
            broke_support = True
    
    return {
        'support': float(recent_low),
        'resistance': float(recent_high),
        'near_support': near_support,
        'near_resistance': near_resistance,
        'broke_support': broke_support,
        'broke_resistance': broke_resistance,
        'distance_to_support': float(dist_to_support),
        'distance_to_resistance': float(dist_to_resistance)
    }

def identify_market_condition(trend_metrics, volatility_metrics):
    """
    Identify the overall market condition based on trend and volatility
    
    Args:
        trend_metrics: Dict with trend metrics
        volatility_metrics: Dict with volatility metrics
        
    Returns:
        String describing market condition
    """
    trend_quality = trend_metrics['trend_quality']
    trend_direction = trend_metrics['trend_direction']
    volatility_regime = volatility_metrics['volatility_regime']
    
    # Strong uptrend with low volatility
    if trend_direction > 0 and trend_quality > 0.7 and volatility_regime in ['low', 'very_low']:
        return 'stable_uptrend'
    
    # Strong uptrend with high volatility
    elif trend_direction > 0 and trend_quality > 0.6 and volatility_regime in ['high', 'very_high']:
        return 'volatile_uptrend'
    
    # Strong downtrend with low volatility
    elif trend_direction < 0 and trend_quality > 0.7 and volatility_regime in ['low', 'very_low']:
        return 'stable_downtrend'
    
    # Strong downtrend with high volatility
    elif trend_direction < 0 and trend_quality > 0.6 and volatility_regime in ['high', 'very_high']:
        return 'volatile_downtrend'
    
    # Weak trend with high volatility
    elif trend_quality < 0.4 and volatility_regime in ['high', 'very_high']:
        return 'choppy_volatile'
    
    # Weak trend with low volatility
    elif trend_quality < 0.4 and volatility_regime in ['low', 'very_low']:
        return 'sideways_consolidation'
    
    # Medium trend with medium volatility
    else:
        if trend_direction > 0:
            return 'moderate_uptrend'
        else:
            return 'moderate_downtrend'