"""
Stock Price Prediction System - Technical Indicators
--------------------------------------------------
This file contains functions for calculating technical indicators and generating signals.
"""

import numpy as np
from utils.config_reader import get_config

def calculate_technical_signals(prices, volumes=None):
    """
    Calculate technical indicators and trading signals
    
    Args:
        prices: Array of price data
        volumes: Array of volume data (optional)
        
    Returns:
        Dict with technical signals
    """
    if len(prices) < 50:
        return {'signals': {}, 'indicators': {}}
    
    signals = {}
    indicators = {}
    
    # Calculate moving averages
    ma20 = np.convolve(prices, np.ones(20)/20, mode='valid')
    ma50 = np.convolve(prices, np.ones(50)/50, mode='valid')
    
    # Pad the beginning to match the original array length
    ma20 = np.concatenate([np.full(20-1, ma20[0]), ma20])
    ma50 = np.concatenate([np.full(50-1, ma50[0]), ma50])
    
    # Store indicators
    indicators['ma20'] = ma20
    indicators['ma50'] = ma50
    
    # MA crossover - fixed to use NumPy array operations instead of pandas shift
    ma_cross = np.zeros(len(prices))
    for i in range(1, len(prices)):
        ma_cross[i] = np.sign(ma20[i] - ma50[i]) - np.sign(ma20[i-1] - ma50[i-1])
    
    signals['ma_cross_buy'] = (ma_cross > 0)
    signals['ma_cross_sell'] = (ma_cross < 0)
    
    # RSI calculation
    deltas = np.diff(prices)
    seed = deltas[:14]
    up = seed[seed >= 0].sum() / 14
    down = -seed[seed < 0].sum() / 14
    
    if down != 0:
        rs = up / down
    else:
        rs = 0
        
    rsi = np.zeros_like(prices)
    rsi[:14] = 100. - 100. / (1. + rs)
    
    for i in range(14, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
            
        up = (up * 13 + upval) / 14
        down = (down * 13 + downval) / 14
        
        if down != 0:
            rs = up / down
        else:
            rs = 0
            
        rsi[i] = 100. - 100. / (1. + rs)
    
    indicators['rsi'] = rsi
    signals['rsi_oversold'] = (rsi < 30)
    signals['rsi_overbought'] = (rsi > 70)
    
    # Bollinger Bands (20, 2)
    rolling_std = np.zeros_like(prices)
    for i in range(20, len(prices)):
        rolling_std[i] = np.std(prices[i-20:i])
    
    upper_band = ma20 + 2 * rolling_std
    lower_band = ma20 - 2 * rolling_std
    
    indicators['upper_band'] = upper_band
    indicators['lower_band'] = lower_band
    
    # Bollinger Band signals
    signals['bb_upper_break'] = (prices > upper_band)
    signals['bb_lower_break'] = (prices < lower_band)
    
    # Price and Volume patterns
    if volumes is not None and len(volumes) == len(prices):
        # Increasing price with increasing volume (bullish)
        price_diff = np.zeros_like(prices)
        price_diff[1:] = prices[1:] - prices[:-1]
        
        volume_diff = np.zeros_like(volumes)
        volume_diff[1:] = volumes[1:] - volumes[:-1]
        
        signals['price_volume_bullish'] = (price_diff > 0) & (volume_diff > 0)
        signals['price_volume_bearish'] = (price_diff < 0) & (volume_diff > 0)
    
    # MACD calculation (optional, if more indicators are needed)
    if len(prices) >= 26:
        # Calculate EMA-12 and EMA-26
        ema12 = np.zeros_like(prices)
        ema26 = np.zeros_like(prices)
        
        # Initialize with SMA
        ema12[11] = np.mean(prices[:12])
        ema26[25] = np.mean(prices[:26])
        
        # EMA multipliers
        mult_12 = 2 / (12 + 1)
        mult_26 = 2 / (26 + 1)
        
        # Calculate EMAs
        for i in range(12, len(prices)):
            ema12[i] = (prices[i] - ema12[i-1]) * mult_12 + ema12[i-1]
            
        for i in range(26, len(prices)):
            ema26[i] = (prices[i] - ema26[i-1]) * mult_26 + ema26[i-1]
        
        # Calculate MACD and signal line
        macd = ema12 - ema26
        
        # Calculate signal line (9-day EMA of MACD)
        signal_line = np.zeros_like(macd)
        if len(prices) >= 35:  # 26 + 9
            signal_line[34] = np.mean(macd[26:35])
            mult_9 = 2 / (9 + 1)
            
            for i in range(35, len(macd)):
                signal_line[i] = (macd[i] - signal_line[i-1]) * mult_9 + signal_line[i-1]
            
            # MACD crossover signals - fixed to use NumPy array operations instead of pandas shift
            macd_cross_buy = np.zeros(len(macd), dtype=bool)
            macd_cross_sell = np.zeros(len(macd), dtype=bool)
            
            for i in range(1, len(macd)):
                # Buy when MACD crosses above signal line
                if macd[i] > signal_line[i] and macd[i-1] <= signal_line[i-1]:
                    macd_cross_buy[i] = True
                # Sell when MACD crosses below signal line
                if macd[i] < signal_line[i] and macd[i-1] >= signal_line[i-1]:
                    macd_cross_sell[i] = True
                    
            signals['macd_cross_buy'] = macd_cross_buy
            signals['macd_cross_sell'] = macd_cross_sell
            
            indicators['macd'] = macd
            indicators['macd_signal'] = signal_line
    
    # Combine signals for overall buy/sell recommendation
    from trading.analysis import evaluate_trend_quality
    trend_metrics = evaluate_trend_quality(prices)
    
    buy_signals = signals.get('ma_cross_buy', np.zeros(len(prices), dtype=bool)) | signals.get('bb_lower_break', np.zeros(len(prices), dtype=bool)) | signals.get('rsi_oversold', np.zeros(len(prices), dtype=bool))
    sell_signals = signals.get('ma_cross_sell', np.zeros(len(prices), dtype=bool)) | signals.get('bb_upper_break', np.zeros(len(prices), dtype=bool)) | signals.get('rsi_overbought', np.zeros(len(prices), dtype=bool))
    
    # Apply trend filter
    if trend_metrics['trend_direction'] > 0:  # Uptrend
        buy_signals = buy_signals & ~signals.get('rsi_overbought', np.zeros(len(prices), dtype=bool))  # Don't buy at overbought in uptrend
        sell_signals = sell_signals & ~signals.get('bb_lower_break', np.zeros(len(prices), dtype=bool))  # Don't sell at support in uptrend
    else:  # Downtrend
        buy_signals = buy_signals & ~signals.get('bb_upper_break', np.zeros(len(prices), dtype=bool))  # Don't buy at resistance in downtrend
        sell_signals = sell_signals & ~signals.get('rsi_oversold', np.zeros(len(prices), dtype=bool))  # Don't sell at oversold in downtrend
    
    signals['buy'] = buy_signals
    signals['sell'] = sell_signals
    
    return {'signals': signals, 'indicators': indicators}