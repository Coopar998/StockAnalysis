"""
Stock Price Prediction System - Runner Script
-------------------------------------------
This script simplifies running the entire system with different configurations.
It allows adjusting key parameters before running the analysis.
"""

import os
import json
import argparse
import logging
from datetime import datetime

from main import main
from utils.config_reader import get_config
from verify_results import print_enhanced_performance_summary

def setup_logging(log_dir='logs'):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('run_system')

def print_config_summary(config):
    """Print a summary of the current configuration"""
    print("\n===== CONFIGURATION SUMMARY =====")
    
    # Position sizing
    pos_sizing = config.get('trading', 'position_sizing')
    print(f"Position Sizing: Base={pos_sizing.get('base_position_pct', 0.3) * 100:.0f}%, "
          f"Max={pos_sizing.get('max_position_pct', 0.5) * 100:.0f}%, "
          f"Min={pos_sizing.get('min_position_pct', 0.1) * 100:.0f}%")
    
    # Signal thresholds
    signals = config.get('trading', 'signal_thresholds')
    print(f"Signal Thresholds: Buy={signals.get('buy_threshold', 1.5):.1f}%, "
          f"Sell={signals.get('sell_threshold', -1.5):.1f}%")
    
    # Risk management
    risk = config.get('trading', 'risk_management')
    print(f"Risk Management: Stop Loss={risk.get('stop_loss_pct', 5.0):.1f}%, "
          f"Trailing Stop={risk.get('trailing_stop_pct', 6.0):.1f}%, "
          f"Take Profit={risk.get('take_profit_pct', 10.0):.1f}%")
    
    # Strategy selection
    strategy = config.get('trading', 'strategy_selection')
    print(f"Strategy Selection: Accuracy Threshold={strategy.get('use_buy_hold_threshold', 0.55):.2f}, "
          f"Buy & Hold Min Return={strategy.get('buy_hold_return_threshold', 20.0):.1f}%")
    
    # Sanity checks
    sanity = config.get('sanity_checks')
    print(f"Sanity Checks: Max Return={sanity.get('max_return_pct', 300):.1f}%, "
          f"Min Trades for High Return={sanity.get('min_trades_for_high_return', 5)}")
    
    # Portfolio
    portfolio = config.get('portfolio')
    print(f"Portfolio: Initial Capital=${portfolio.get('initial_capital', 1000000):,}, "
          f"Max Stocks={portfolio.get('max_stocks', 40)}")
    
    print("==================================\n")

def update_config(args):
    """Update configuration based on command line arguments"""
    logger = logging.getLogger('run_system')
    config = get_config()
    config_updated = False
    
    # Load current config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    try:
        with open(config_path, 'r') as f:
            current_config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return False
    
    # Update position sizing
    if args.base_position is not None:
        current_config['trading']['position_sizing']['base_position_pct'] = args.base_position / 100
        config_updated = True
    
    if args.max_position is not None:
        current_config['trading']['position_sizing']['max_position_pct'] = args.max_position / 100
        config_updated = True
    
    # Update signal thresholds
    if args.buy_threshold is not None:
        current_config['trading']['signal_thresholds']['buy_threshold'] = args.buy_threshold
        config_updated = True
        
    if args.sell_threshold is not None:
        current_config['trading']['signal_thresholds']['sell_threshold'] = args.sell_threshold
        config_updated = True
    
    # Update risk management
    if args.stop_loss is not None:
        current_config['trading']['risk_management']['stop_loss_pct'] = args.stop_loss
        config_updated = True
        
    if args.take_profit is not None:
        current_config['trading']['risk_management']['take_profit_pct'] = args.take_profit
        config_updated = True
    
    # Update sanity checks
    if args.max_return is not None:
        current_config['sanity_checks']['max_return_pct'] = args.max_return
        config_updated = True
    
    # Update portfolio
    if args.stocks is not None:
        current_config['portfolio']['max_stocks'] = args.stocks
        config_updated = True
    
    # Save updated config
    if config_updated:
        try:
            with open(config_path, 'w') as f:
                json.dump(current_config, f, indent=2)
            logger.info("Configuration updated successfully")
            # Reload config
            config.reload()
            return True
        except Exception as e:
            logger.error(f"Error saving updated config: {e}")
            return False
    else:
        logger.info("No configuration changes needed")
        return False

def run():
    """Run the stock prediction system with current configuration"""
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Run the Stock Price Prediction System")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--base-position", type=float, help="Base position size (percentage)")
    parser.add_argument("--max-position", type=float, help="Maximum position size (percentage)")
    parser.add_argument("--buy-threshold", type=float, help="Buy signal threshold (percentage)")
    parser.add_argument("--sell-threshold", type=float, help="Sell signal threshold (percentage)")
    parser.add_argument("--stop-loss", type=float, help="Stop loss percentage")
    parser.add_argument("--take-profit", type=float, help="Take profit percentage")
    parser.add_argument("--max-return", type=float, help="Maximum allowed return percentage")
    parser.add_argument("--stocks", type=int, help="Number of stocks to analyze")
    
    args = parser.parse_args()
    
    # Update configuration if needed
    if update_config(args):
        logger.info("Configuration updated, reloading settings")
    
    # Get current configuration
    config = get_config()
    print_config_summary(config)
    
    logger.info("Starting stock prediction system")
    
    # Run the main function
    try:
        results, portfolio = main(verbose=args.verbose)
        
        if results and portfolio:
            # Get date range for the analysis
            from datetime import datetime, timedelta
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=3000)).strftime('%Y-%m-%d')
            
            # Print enhanced performance summary
            print_enhanced_performance_summary(portfolio, start_date, end_date)
            
            logger.info("Stock prediction system completed successfully")
            return 0
        else:
            logger.error("Stock prediction system failed (no results returned)")
            return 1
    except Exception as e:
        logger.error(f"Error running stock prediction system: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = run()
    exit(exit_code)