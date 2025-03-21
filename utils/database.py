"""
Stock Price Prediction System - Database Utilities
------------------------------------------------
This file provides database functionality for storing and retrieving stock data.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta

# Define database path
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'stock_data.db')

class StockDatabase:
    """Class to handle all database operations for stock data"""
    
    def __init__(self, db_path=None):
        """Initialize database connection"""
        self.db_path = db_path or DEFAULT_DB_PATH
        self.logger = logging.getLogger('stock_prediction')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create stock metadata table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_metadata (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                last_updated TIMESTAMP,
                data_start_date TEXT,
                data_end_date TEXT,
                additional_info TEXT
            )
            ''')
            
            # Create price data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                UNIQUE(ticker, date)
            )
            ''')
            
            # Create technical indicators table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                date TEXT,
                sma_20 REAL,
                sma_50 REAL,
                sma_200 REAL,
                rsi_14 REAL,
                macd REAL,
                macd_signal REAL,
                bb_upper REAL,
                bb_lower REAL,
                indicator_data TEXT,
                UNIQUE(ticker, date)
            )
            ''')
            
            # Create trading signals table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                date TEXT,
                buy_signal INTEGER,
                sell_signal INTEGER,
                prediction REAL,
                confidence REAL,
                signal_source TEXT,
                signal_data TEXT,
                UNIQUE(ticker, date, signal_source)
            )
            ''')
            
            conn.commit()
    
    def store_stock_data(self, ticker, df, metadata=None, compute_indicators=False):
        """
        Store stock OHLCV data in the database
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data
            metadata: Dict with additional stock metadata
            compute_indicators: Whether to compute and store technical indicators
        """
        if df is None or df.empty:
            self.logger.warning(f"No data provided for {ticker}")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store price data
                df_copy = df.copy()
                
                # Ensure date column is formatted correctly for SQL
                if isinstance(df_copy.index, pd.DatetimeIndex):
                    df_copy = df_copy.reset_index()
                    df_copy['date'] = df_copy['index'].astype(str)
                    df_copy = df_copy.drop('index', axis=1)
                elif 'date' not in df_copy.columns and 'Date' in df_copy.columns:
                    df_copy = df_copy.rename(columns={'Date': 'date'})
                
                # Extract OHLCV columns
                price_columns = ['date']
                
                # Map standard column names to database columns
                column_mapping = {
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Adj Close': 'adj_close',
                    'Volume': 'volume'
                }
                
                # Check for alternative column formats (_XOM suffix, etc.)
                for col_name, db_col in column_mapping.items():
                    if col_name in df_copy.columns:
                        df_copy[db_col] = df_copy[col_name]
                        price_columns.append(db_col)
                    else:
                        # Look for columns with suffix
                        suffix_cols = [c for c in df_copy.columns if c.startswith(f"{col_name}_")]
                        if suffix_cols:
                            df_copy[db_col] = df_copy[suffix_cols[0]]
                            price_columns.append(db_col)
                        else:
                            # If column is missing, add it with NaN values
                            df_copy[db_col] = np.nan
                            price_columns.append(db_col)
                
                # Add ticker column
                df_copy['ticker'] = ticker
                price_columns.append('ticker')
                
                # Only keep necessary columns
                price_df = df_copy[price_columns]
                
                # Store price data
                price_df.to_sql('price_data', conn, if_exists='replace', index=False)
                
                # Update metadata
                data_start_date = price_df['date'].min()
                data_end_date = price_df['date'].max()
                
                if metadata is None:
                    metadata = {}
                
                # Update stock metadata
                metadata_record = {
                    'ticker': ticker,
                    'company_name': metadata.get('company_name', ''),
                    'sector': metadata.get('sector', ''),
                    'industry': metadata.get('industry', ''),
                    'market_cap': metadata.get('market_cap', 0),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'data_start_date': data_start_date,
                    'data_end_date': data_end_date,
                    'additional_info': json.dumps(metadata.get('additional_info', {}))
                }
                
                # Insert or update metadata
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO stock_metadata 
                (ticker, company_name, sector, industry, market_cap, last_updated, 
                data_start_date, data_end_date, additional_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata_record['ticker'],
                    metadata_record['company_name'],
                    metadata_record['sector'],
                    metadata_record['industry'],
                    metadata_record['market_cap'],
                    metadata_record['last_updated'],
                    metadata_record['data_start_date'],
                    metadata_record['data_end_date'],
                    metadata_record['additional_info']
                ))
                
                # Compute and store technical indicators if requested
                if compute_indicators and 'close' in price_df.columns:
                    self._store_technical_indicators(conn, ticker, price_df)
                
                conn.commit()
                
            self.logger.info(f"Successfully stored data for {ticker} in database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing data for {ticker}: {e}")
            return False
    
    def _store_technical_indicators(self, conn, ticker, price_df):
        """Compute and store technical indicators for a stock"""
        from scipy.signal import lfilter
        
        try:
            # Create a DataFrame for indicators
            df = price_df.copy()
            df = df.sort_values('date')
            
            # Create indicators DataFrame
            indicators_df = pd.DataFrame()
            indicators_df['ticker'] = df['ticker']
            indicators_df['date'] = df['date']
            
            prices = df['close'].values
            
            # Calculate SMA
            def sma(data, window):
                return lfilter(np.ones(window)/window, 1, data)
            
            # SMA calculations
            indicators_df['sma_20'] = pd.Series(sma(prices, 20)).shift(1)
            indicators_df['sma_50'] = pd.Series(sma(prices, 50)).shift(1)
            indicators_df['sma_200'] = pd.Series(sma(prices, 200)).shift(1)
            
            # RSI calculation (simplified)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
            indicators_df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            indicators_df['macd'] = ema12 - ema26
            indicators_df['macd_signal'] = indicators_df['macd'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            rolling_std = df['close'].rolling(window=20).std()
            indicators_df['bb_upper'] = indicators_df['sma_20'] + (rolling_std * 2)
            indicators_df['bb_lower'] = indicators_df['sma_20'] - (rolling_std * 2)
            
            # Additional indicator data as JSON
            # Store other indicators that don't fit in standard columns
            indicator_data = {}
            
            # Store as JSON
            indicators_df['indicator_data'] = indicators_df.apply(
                lambda row: json.dumps(indicator_data), axis=1
            )
            
            # Replace NaN with NULL for SQLite
            indicators_df = indicators_df.replace({np.nan: None})
            
            # First, delete any existing indicators for this ticker and date range
            min_date = indicators_df['date'].min()
            max_date = indicators_df['date'].max()
            
            delete_query = f"""
            DELETE FROM technical_indicators 
            WHERE ticker = ? AND date >= ? AND date <= ?
            """
            conn.execute(delete_query, (ticker, min_date, max_date))
            
            # Now safely store the new indicators
            indicators_df.to_sql('technical_indicators', conn, if_exists='append', index=False)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {ticker}: {e}")
    
    def get_stock_data(self, ticker, start_date=None, end_date=None, include_indicators=False):
        """
        Retrieve stock data from database
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            include_indicators: Whether to include technical indicators
            
        Returns:
            DataFrame with stock data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Build query
                query = f"SELECT * FROM price_data WHERE ticker = '{ticker}'"
                if start_date:
                    query += f" AND date >= '{start_date}'"
                if end_date:
                    query += f" AND date <= '{end_date}'"
                query += " ORDER BY date"
                
                # Get price data
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    return None
                
                # Rename columns to standard format
                column_mapping = {
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'adj_close': 'Adj Close',
                    'volume': 'Volume'
                }
                df = df.rename(columns=column_mapping)
                
                # Set date as index
                df['Date'] = pd.to_datetime(df['date'])
                df = df.set_index('Date')
                df = df.drop(['date', 'id', 'ticker'], axis=1, errors='ignore')
                
                # Add indicators if requested
                if include_indicators:
                    indicators_query = f"SELECT * FROM technical_indicators WHERE ticker = '{ticker}'"
                    if start_date:
                        indicators_query += f" AND date >= '{start_date}'"
                    if end_date:
                        indicators_query += f" AND date <= '{end_date}'"
                    indicators_query += " ORDER BY date"
                    
                    indicators_df = pd.read_sql_query(indicators_query, conn)
                    
                    if not indicators_df.empty:
                        # Convert date to datetime for joining
                        indicators_df['Date'] = pd.to_datetime(indicators_df['date'])
                        indicators_df = indicators_df.set_index('Date')
                        
                        # Remove unnecessary columns
                        indicators_df = indicators_df.drop(['date', 'id', 'ticker'], axis=1, errors='ignore')
                        
                        # Rename indicator columns
                        indicator_mapping = {
                            'sma_20': 'SMA_20',
                            'sma_50': 'SMA_50',
                            'sma_200': 'SMA_200',
                            'rsi_14': 'RSI_14',
                            'macd': 'MACD',
                            'macd_signal': 'MACD_Signal',
                            'bb_upper': 'BB_Upper',
                            'bb_lower': 'BB_Lower'
                        }
                        indicators_df = indicators_df.rename(columns=indicator_mapping)
                        
                        # Parse JSON data for additional indicators
                        def parse_indicator_data(json_str):
                            if pd.isna(json_str) or json_str is None:
                                return {}
                            try:
                                return json.loads(json_str)
                            except:
                                return {}
                        
                        # Extract additional indicators from JSON
                        if 'indicator_data' in indicators_df.columns:
                            indicator_dict = indicators_df['indicator_data'].apply(parse_indicator_data)
                            
                            # Create new columns for each indicator in the JSON
                            for row_idx, row_dict in indicator_dict.items():
                                for indicator_name, value in row_dict.items():
                                    indicators_df.at[row_idx, indicator_name] = value
                            
                            # Remove the JSON column
                            indicators_df = indicators_df.drop('indicator_data', axis=1)
                        
                        # Join with price data
                        df = df.join(indicators_df)
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error retrieving data for {ticker}: {e}")
            return None
    
    def get_last_update_date(self, ticker):
        """
        Get the last date when data was updated for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            datetime object or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT last_updated FROM stock_metadata WHERE ticker = ?", (ticker,))
                result = cursor.fetchone()
                
                if result and result[0]:
                    return datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting last update date for {ticker}: {e}")
            return None
    
    def get_available_tickers(self):
        """
        Get list of tickers available in the database
        
        Returns:
            List of ticker strings
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ticker FROM stock_metadata")
                results = cursor.fetchall()
                return [row[0] for row in results]
                
        except Exception as e:
            self.logger.error(f"Error getting available tickers: {e}")
            return []
    
    def data_needs_update(self, ticker, max_age_days=1):
        """
        Check if data for a ticker needs to be updated
        
        Args:
            ticker: Stock ticker symbol
            max_age_days: Maximum age in days before data is considered stale
            
        Returns:
            Boolean indicating if data needs update
        """
        last_update = self.get_last_update_date(ticker)
        
        if last_update is None:
            return True
            
        # Check if data is older than max_age_days
        max_age = timedelta(days=max_age_days)
        return (datetime.now() - last_update) > max_age
    
    def get_data_date_range(self, ticker):
        """
        Get the date range of data available for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (start_date, end_date) strings or (None, None) if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data_start_date, data_end_date FROM stock_metadata WHERE ticker = ?", 
                    (ticker,)
                )
                result = cursor.fetchone()
                
                if result:
                    return result
                return (None, None)
                
        except Exception as e:
            self.logger.error(f"Error getting date range for {ticker}: {e}")
            return (None, None)
    
    def store_trading_signals(self, ticker, signals_df):
        """
        Store trading signals in the database
        
        Args:
            ticker: Stock ticker symbol
            signals_df: DataFrame with signals data
            
        Returns:
            Boolean indicating success
        """
        if signals_df is None or signals_df.empty:
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prepare DataFrame for storage
                df_copy = signals_df.copy()
                
                # Ensure we have date column
                if isinstance(df_copy.index, pd.DatetimeIndex):
                    df_copy = df_copy.reset_index()
                    df_copy['date'] = df_copy['index'].astype(str)
                    df_copy = df_copy.drop('index', axis=1)
                
                # Add ticker column
                df_copy['ticker'] = ticker
                
                # Ensure required columns exist
                if 'buy_signal' not in df_copy.columns:
                    df_copy['buy_signal'] = 0
                if 'sell_signal' not in df_copy.columns:
                    df_copy['sell_signal'] = 0
                if 'prediction' not in df_copy.columns:
                    df_copy['prediction'] = None
                if 'confidence' not in df_copy.columns:
                    df_copy['confidence'] = None
                if 'signal_source' not in df_copy.columns:
                    df_copy['signal_source'] = 'model'
                
                # Convert any additional columns to JSON
                cols_to_keep = ['ticker', 'date', 'buy_signal', 'sell_signal', 
                                'prediction', 'confidence', 'signal_source']
                extra_cols = [col for col in df_copy.columns if col not in cols_to_keep]
                
                if extra_cols:
                    df_copy['signal_data'] = df_copy.apply(
                        lambda row: json.dumps({col: row[col] for col in extra_cols 
                                               if not pd.isna(row[col])}),
                        axis=1
                    )
                else:
                    df_copy['signal_data'] = '{}'
                
                # Keep only required columns
                df_copy = df_copy[cols_to_keep + ['signal_data']]
                
                # Replace NaN with None for SQLite
                df_copy = df_copy.replace({np.nan: None})
                
                # Store in database
                df_copy.to_sql('trading_signals', conn, if_exists='append', index=False)
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing signals for {ticker}: {e}")
            return False
    
    def get_trading_signals(self, ticker, start_date=None, end_date=None, source=None):
        """
        Retrieve trading signals from database
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for signals (optional)
            end_date: End date for signals (optional)
            source: Signal source filter (optional)
            
        Returns:
            DataFrame with signals or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = f"SELECT * FROM trading_signals WHERE ticker = '{ticker}'"
                
                if start_date:
                    query += f" AND date >= '{start_date}'"
                if end_date:
                    query += f" AND date <= '{end_date}'"
                if source:
                    query += f" AND signal_source = '{source}'"
                
                query += " ORDER BY date"
                
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    return None
                
                # Convert date to datetime
                df['Date'] = pd.to_datetime(df['date'])
                df = df.set_index('Date')
                
                # Remove unnecessary columns
                df = df.drop(['date', 'id', 'ticker'], axis=1, errors='ignore')
                
                # Parse signal_data JSON
                def parse_signal_data(json_str):
                    if pd.isna(json_str) or json_str is None:
                        return {}
                    try:
                        return json.loads(json_str)
                    except:
                        return {}
                
                # Extract additional data from JSON
                if 'signal_data' in df.columns:
                    signal_dict = df['signal_data'].apply(parse_signal_data)
                    
                    # Create new columns for each field in the JSON
                    for row_idx, row_dict in signal_dict.items():
                        for field_name, value in row_dict.items():
                            df.at[row_idx, field_name] = value
                    
                    # Remove the JSON column
                    df = df.drop('signal_data', axis=1)
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error retrieving signals for {ticker}: {e}")
            return None