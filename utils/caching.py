"""
Stock Price Prediction System - Caching Utilities
----------------------------------------------
This file handles caching mechanisms for data and models.
"""

import os
import pickle
import json
import hashlib
from datetime import datetime

# Define default cache directory
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')

def get_cache_dir():
    """Get the cache directory, creating it if it doesn't exist"""
    cache_dir = DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def cache_key_from_args(*args, **kwargs):
    """
    Generate a cache key from function arguments
    
    Args:
        *args, **kwargs: Arguments to create cache key from
        
    Returns:
        String cache key
    """
    # Convert all arguments to strings and join them
    args_str = '_'.join([str(arg) for arg in args])
    kwargs_str = '_'.join([f"{k}_{v}" for k, v in sorted(kwargs.items())])
    combined = f"{args_str}_{kwargs_str}"
    
    # Hash the combined string to create a fixed-length key
    return hashlib.md5(combined.encode()).hexdigest()

def save_to_cache(key, data, cache_dir=None, expiry_seconds=None):
    """
    Save data to cache
    
    Args:
        key: Cache key (string)
        data: Data to cache (must be pickle-serializable)
        cache_dir: Cache directory (optional)
        expiry_seconds: Expiry time in seconds (optional)
        
    Returns:
        Path to cache file
    """
    cache_dir = cache_dir or get_cache_dir()
    
    # Create cache metadata
    metadata = {
        'created_timestamp': datetime.now().timestamp(),
        'expiry_seconds': expiry_seconds
    }
    
    # Combine metadata and data
    cache_data = {
        'metadata': metadata,
        'data': data
    }
    
    # Save to file
    cache_file = os.path.join(cache_dir, f"{key}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return cache_file

def load_from_cache(key, cache_dir=None):
    """
    Load data from cache
    
    Args:
        key: Cache key (string)
        cache_dir: Cache directory (optional)
        
    Returns:
        Cached data or None if not found or expired
    """
    cache_dir = cache_dir or get_cache_dir()
    cache_file = os.path.join(cache_dir, f"{key}.pkl")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        metadata = cache_data.get('metadata', {})
        created_timestamp = metadata.get('created_timestamp')
        expiry_seconds = metadata.get('expiry_seconds')
        
        # Check if cache is expired
        if created_timestamp and expiry_seconds:
            current_time = datetime.now().timestamp()
            if current_time - created_timestamp > expiry_seconds:
                print(f"Cache expired for key: {key}")
                return None
        
        return cache_data.get('data')
    
    except Exception as e:
        print(f"Error loading from cache: {e}")
        return None

def clear_cache(cache_dir=None, max_age_seconds=None):
    """
    Clear cache files
    
    Args:
        cache_dir: Cache directory (optional)
        max_age_seconds: Only remove files older than this many seconds (optional)
        
    Returns:
        Number of files removed
    """
    cache_dir = cache_dir or get_cache_dir()
    removed = 0
    
    if not os.path.exists(cache_dir):
        return 0
    
    current_time = datetime.now().timestamp()
    
    for filename in os.listdir(cache_dir):
        if not filename.endswith('.pkl'):
            continue
        
        file_path = os.path.join(cache_dir, filename)
        
        # Check file age if max_age_seconds is specified
        if max_age_seconds:
            file_time = os.path.getmtime(file_path)
            age_seconds = current_time - file_time
            
            if age_seconds <= max_age_seconds:
                continue
        
        try:
            os.remove(file_path)
            removed += 1
        except Exception as e:
            print(f"Error removing cache file {file_path}: {e}")
    
    return removed

def cache_decorator(expiry_seconds=86400):  # Default 1 day
    """
    Decorator to cache function results
    
    Args:
        expiry_seconds: Cache expiry time in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}_{cache_key_from_args(*args, **kwargs)}"
            
            # Try to load from cache
            cached_result = load_from_cache(cache_key)
            if cached_result is not None:
                print(f"Using cached result for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            save_to_cache(cache_key, result, expiry_seconds=expiry_seconds)
            
            return result
        return wrapper
    return decorator