"""
Stock Price Prediction System - Configuration Reader
---------------------------------------------------
This file handles loading and validation of the configuration file.
"""

import os
import json
import logging

# Default configuration
DEFAULT_CONFIG = {
    "version": "1.0.0",
    "trading": {
        "position_sizing": {
            "base_position_pct": 0.3,
            "max_position_pct": 0.5,
            "min_position_pct": 0.1
        },
        "signal_thresholds": {
            "buy_threshold": 1.5,
            "sell_threshold": -1.5
        },
        "risk_management": {
            "stop_loss_pct": 5.0,
            "trailing_stop_pct": 6.0,
            "take_profit_pct": 10.0,
            "max_holding_period": 10,
            "min_holding_period": 2
        },
        "strategy_selection": {
            "use_buy_hold_threshold": 0.55,
            "buy_hold_return_threshold": 20.0,
            "trend_quality_threshold": 0.6,
            "buy_hold_uptrend_threshold": 15.0
        }
    },
    "data_processing": {
        "sequence_stride": 5,
        "sequence_length": 20,
        "min_data_points": 250,
        "lightweight_mode": True
    },
    "sanity_checks": {
        "max_return_pct": 300,
        "min_trades_for_high_return": 5,
        "max_return_few_trades": 100
    },
    "portfolio": {
        "initial_capital": 1000000,
        "max_stocks": 40
    }
}

class Config:
    """Configuration singleton class"""
    _instance = None
    _config = None
    _logger = logging.getLogger('stock_prediction')
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from file or use defaults"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self._config = json.load(f)
                self._logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self._logger.error(f"Error loading config file: {e}")
                self._logger.info("Using default configuration")
                self._config = DEFAULT_CONFIG
        else:
            self._logger.warning(f"Configuration file not found at {config_path}")
            self._logger.info("Using default configuration")
            self._config = DEFAULT_CONFIG
            
            # Save default config for future reference
            try:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=2)
                self._logger.info(f"Saved default configuration to {config_path}")
            except Exception as e:
                self._logger.error(f"Error saving default config: {e}")
    
    def get(self, *keys, default=None):
        """Get a configuration value by its key path"""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
        return self._config

# Create a single instance to be used application-wide
config = Config()

def get_config():
    """Get the configuration instance"""
    return config