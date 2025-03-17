"""
Stock Price Prediction System - Feature Importance Analysis
--------------------------------------------------------
This file contains functions for analyzing feature importance and selecting
the most predictive features for stock price prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from tensorflow.keras.models import Model
import os

def analyze_feature_importance(X, y, feature_names, output_dir=None, ticker=None):
    """
    Analyze feature importance using multiple methods
    
    Args:
        X: Features array
        y: Target array
        feature_names: List of feature names
        output_dir: Directory to save outputs
        ticker: Stock ticker symbol for labeling
        
    Returns:
        Dictionary with feature importance rankings
    """
    # Prepare data - reshape 3D sequences to 2D
    if len(X.shape) == 3:
        # For LSTM inputs, use the last time step only
        X_last_step = X[:, -1, :]
    else:
        X_last_step = X.copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_last_step)
    
    # Method 1: Random Forest feature importance
    rf_importance = get_random_forest_importance(X_scaled, y, feature_names)
    
    # Method 2: Correlation-based feature importance
    correlation_importance = get_correlation_importance(X_scaled, y, feature_names)
    
    # Method 3: Mutual Information feature importance
    mi_importance = get_mutual_info_importance(X_scaled, y, feature_names)
    
    # Combine results from different methods
    combined_importance = combine_feature_importance(
        [rf_importance, correlation_importance, mi_importance],
        ['Random Forest', 'Correlation', 'Mutual Information']
    )
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'RandomForest_Importance': rf_importance['scores'],
        'Correlation_Importance': correlation_importance['scores'],
        'MutualInfo_Importance': mi_importance['scores'],
        'Combined_Importance': combined_importance['scores']
    })
    
    # Sort by combined importance
    results_df = results_df.sort_values('Combined_Importance', ascending=False)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{ticker}_feature_importance.csv" if ticker else "feature_importance.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Feature importance saved to {csv_path}")
    
    # Plot feature importance
    plot_top_features(combined_importance, ticker, output_dir)
    
    return {
        'random_forest': rf_importance,
        'correlation': correlation_importance,
        'mutual_info': mi_importance,
        'combined': combined_importance,
        'dataframe': results_df
    }

def get_random_forest_importance(X, y, feature_names):
    """Get feature importance using Random Forest"""
    print("Calculating Random Forest feature importance...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Create sorted indices
    indices = np.argsort(importance)[::-1]
    
    return {
        'scores': importance,
        'sorted_scores': importance[indices],
        'sorted_indices': indices,
        'sorted_names': [feature_names[i] for i in indices]
    }

def get_correlation_importance(X, y, feature_names):
    """Get feature importance using correlation scores"""
    print("Calculating correlation-based feature importance...")
    # Use f_regression which returns F-scores (based on correlation for regression)
    selector = SelectKBest(score_func=f_regression, k='all')
    selector.fit(X, y)
    
    # Get scores and convert to importances (normalize to 0-1)
    scores = selector.scores_
    # Replace NaN values with zeros
    scores = np.nan_to_num(scores)
    importance = scores / np.max(scores) if np.max(scores) > 0 else scores
    
    # Create sorted indices
    indices = np.argsort(importance)[::-1]
    
    return {
        'scores': importance,
        'sorted_scores': importance[indices],
        'sorted_indices': indices,
        'sorted_names': [feature_names[i] for i in indices]
    }

def get_mutual_info_importance(X, y, feature_names):
    """Get feature importance using mutual information"""
    print("Calculating mutual information feature importance...")
    # Use mutual information which captures non-linear relationships
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    # Normalize scores to 0-1
    importance = mi_scores / np.max(mi_scores) if np.max(mi_scores) > 0 else mi_scores
    
    # Create sorted indices
    indices = np.argsort(importance)[::-1]
    
    return {
        'scores': importance,
        'sorted_scores': importance[indices],
        'sorted_indices': indices,
        'sorted_names': [feature_names[i] for i in indices]
    }

def combine_feature_importance(importance_list, method_names):
    """Combine feature importance scores from multiple methods"""
    print("Combining feature importance scores...")
    n_features = len(importance_list[0]['scores'])
    
    # Initialize combined scores
    combined_scores = np.zeros(n_features)
    
    # Add weighted scores from each method
    for i, importance in enumerate(importance_list):
        # Apply equal weights for now (can be adjusted)
        weight = 1.0 / len(importance_list)
        combined_scores += weight * importance['scores']
    
    # Normalize combined scores
    combined_scores = combined_scores / np.max(combined_scores) if np.max(combined_scores) > 0 else combined_scores
    
    # Create sorted indices
    indices = np.argsort(combined_scores)[::-1]
    
    # Get feature names from the first importance method (they should be the same across methods)
    feature_names = [importance_list[0]['sorted_names'][importance_list[0]['sorted_indices'].tolist().index(i)] for i in range(n_features)]
    
    return {
        'scores': combined_scores,
        'sorted_scores': combined_scores[indices],
        'sorted_indices': indices,
        'sorted_names': [feature_names[i] for i in indices]
    }

def plot_top_features(importance, ticker=None, output_dir=None, top_n=20):
    """Plot top features by importance"""
    # Get top features
    n_features = min(top_n, len(importance['sorted_names']))
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(n_features), importance['sorted_scores'][:n_features], align='center')
    plt.yticks(range(n_features), importance['sorted_names'][:n_features])
    plt.xlabel('Importance')
    plt.title(f'Top {n_features} Feature Importance' + (f' for {ticker}' if ticker else ''))
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{ticker}_top_features.png" if ticker else "top_features.png"))
        print(f"Feature importance plot saved to {output_dir}")
    
    plt.close()

def get_optimal_feature_subset(X, y, feature_names, threshold=0.8, min_features=10):
    """
    Select an optimal subset of features based on importance
    
    Args:
        X: Features array
        y: Target array
        feature_names: List of feature names
        threshold: Cumulative importance threshold (0-1)
        min_features: Minimum number of features to include
        
    Returns:
        List of selected feature names and indices
    """
    # Analyze feature importance
    importance = analyze_feature_importance(X, y, feature_names)
    combined = importance['combined']
    
    # Calculate cumulative importance
    sorted_scores = combined['sorted_scores']
    cumulative_importance = np.cumsum(sorted_scores) / np.sum(sorted_scores)
    
    # Find number of features needed to reach threshold
    n_features = np.searchsorted(cumulative_importance, threshold) + 1
    
    # Ensure minimum number of features
    n_features = max(n_features, min_features)
    
    # Get selected feature indices and names
    selected_indices = combined['sorted_indices'][:n_features]
    selected_names = combined['sorted_names'][:n_features]
    
    print(f"Selected {n_features} features with combined importance of {cumulative_importance[n_features-1]:.2f}")
    
    return {
        'selected_indices': selected_indices,
        'selected_names': selected_names,
        'importance_scores': sorted_scores[:n_features],
        'cumulative_importance': cumulative_importance[n_features-1]
    }