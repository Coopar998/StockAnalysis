"""
Stock Price Prediction System - Trading Package
---------------------------------------------
This package contains components for trading strategy implementation and evaluation.
"""

from trading.strategy import process_ticker, analyze_performance
from trading.execution import execute_trading_strategy, execute_buy_hold_strategy
from trading.indicators import calculate_technical_signals
from trading.analysis import evaluate_trend_quality, evaluate_volatility
from trading.selection import is_good_for_active_trading, rank_stocks_for_trading, select_stocks_for_portfolio