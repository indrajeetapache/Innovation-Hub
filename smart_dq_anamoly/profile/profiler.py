import pandas as pd
import numpy as np
from scipy import stats
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Set

def generate_profile_metrics(
    df: pd.DataFrame, 
    timestamp_col: str = 'timestamp', 
    window: str = '1D',
    correlation_columns: Optional[List[str]] = None,
    business_metrics: Optional[Dict[str, List[str]]] = None,
    rate_metrics: Optional[List[str]] = None,
    skip_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate comprehensive profiling metrics for any dataset with time series data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data to profile
    timestamp_col : str
        Column name containing timestamps
    window : str
        Time window for aggregation ('1D', '1W', '1M', etc.)
    correlation_columns : Optional[List[str]]
        List of numerical columns to calculate correlations for
        If None, no correlations are calculated
    business_metrics : Optional[Dict[str, List[str]]]
        Dictionary with keys as metric names and values as lists of columns to aggregate
        Example: {'total_value': ['price', 'tax'], 'transaction_metrics': ['quantity', 'discount']}
    rate_metrics : Optional[List[str]]
        List of columns to calculate rate of change metrics for
    skip_columns : Optional[List[str]]
        List of columns to skip during profiling
        
    Returns:
    --------
    pd.DataFrame with profiling metrics per time window
    """
    start_time = time.time()
    print("📊 Step 1: Starting data profiling process...")
    
    # Input validation
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in dataframe")
    
    # Initialize skip columns if not provided
    if skip_columns is None:
        skip_columns = []
    else:
        skip_columns = list(skip_columns)  # Create a copy to avoid modifying the input
        
    # Always skip the timestamp column
    if timestamp_col not in skip_columns:
        skip_columns.append(timestamp_col)
    
    # Ensure timestamp column is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Set timestamp as index for resampling
    df_indexed = df.set_index(timestamp_col)
    
    # Get numerical and categorical columns (excluding those in skip_columns)
    num_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                if col not in skip_columns]
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category', 'bool']).columns 
                if col not in skip_columns]
    
    print(f"🔍 Found {len(num_cols)} numerical columns and {len(cat_cols)} categorical columns")
    
    # Initialize results list
    profile_metrics = []
    
    # Calculate global distribution statistics first (for comparison)
    global_stats = {}
    for col in num_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            global_stats[f'{col}_global_mean'] = col_data.mean()
            global_stats[f'{col}_global_std'] = col_data.std()
            global_stats[f'{col}_global_min'] = col_data.min()
            global_stats[f'{col}_global_max'] = col_data.max()
            global_stats[f'{col}_global_p25'] = col_data.quantile(0.25)
            global_stats[f'{col}_global_p50'] = col_data.quantile(0.50)
            global_stats[f'{col}_global_p75'] = col_data.quantile(0.75)
            global_stats[f'{col}_global_p95'] = col_data.quantile(0.95)
    
    print(f"📅 Step 2: Processing data in {window} windows...")
    # Process each time window
    for i, (window_start, window_df) in enumerate(df_indexed.resample(window), 1):
        if i % 10 == 0:
            print(f"📦 Processing window {i} starting on {window_start.date()}...")
            
        window_metrics = {'window_start': window_start}
        
        # Skip empty windows
        if len(window_df) == 0:
            continue
            
        # Overall metrics
        window_metrics['row_count'] = len(window_df)
        window_metrics['column_count'] = len(window_df.columns)
        
        # Missing values overall
        window_metrics['missing_values_total'] = window_df.isna().sum().sum()
        window_metrics['missing_values_pct'] = (window_df.isna().sum().sum() / (len(window_df) * len(window_df.columns))) * 100
        
        # Duplicates
        window_metrics['duplicate_rows'] = window_df.duplicated().sum()
        window_metrics['duplicate_rows_pct'] = (window_df.duplicated().sum() / len(window_df)) * 100
        
        # -----------------------------------
        # Calculate metrics for numerical columns
        # -----------------------------------
        for col in num_cols:
            # Get non-missing values
            col_data = window_df[col].dropna()
            
            # Skip if no data
            if len(col_data) == 0:
                window_metrics[f'{col}_missing_pct'] = 100.0
                continue
                
            # Basic statistics
            window_metrics[f'{col}_count'] = len(col_data)
            window_metrics[f'{col}_missing_count'] = len(window_df) - len(col_data)
            window_metrics[f'{col}_missing_pct'] = ((len(window_df) - len(col_data)) / len(window_df)) * 100
            window_metrics[f'{col}_mean'] = col_data.mean()
            window_metrics[f'{col}_median'] = col_data.median()
            window_metrics[f'{col}_std'] = col_data.std()
            window_metrics[f'{col}_min'] = col_data.min()
            window_metrics[f'{col}_max'] = col_data.max()
            window_metrics[f'{col}_range'] = col_data.max() - col_data.min()
            
            # Percentiles
            window_metrics[f'{col}_p25'] = col_data.quantile(0.25)
            window_metrics[f'{col}_p75'] = col_data.quantile(0.75)
            window_metrics[f'{col}_p95'] = col_data.quantile(0.95)
            window_metrics[f'{col}_p99'] = col_data.quantile(0.99)
            window_metrics[f'{col}_iqr'] = window_metrics[f'{col}_p75'] - window_metrics[f'{col}_p25']
            
            # Distribution shape
            window_metrics[f'{col}_skew'] = col_data.skew()
            window_metrics[f'{col}_kurt'] = col_data.kurtosis()
            
            # Compare with global stats (drift detection)
            if f'{col}_global_mean' in global_stats:
                window_metrics[f'{col}_mean_vs_global'] = (window_metrics[f'{col}_mean'] - global_stats[f'{col}_global_mean']) / global_stats[f'{col}_global_mean'] if global_stats[f'{col}_global_mean'] != 0 else 0
                window_metrics[f'{col}_std_vs_global'] = (window_metrics[f'{col}_std'] - global_stats[f'{col}_global_std']) / global_stats[f'{col}_global_std'] if global_stats[f'{col}_global_std'] != 0 else 0
                
                # Z-score of the window mean compared to global distribution
                if global_stats[f'{col}_global_std'] > 0:
                    window_metrics[f'{col}_zscore'] = (window_metrics[f'{col}_mean'] - global_stats[f'{col}_global_mean']) / global_stats[f'{col}_global_std']
                else:
                    window_metrics[f'{col}_zscore'] = 0
            
            # Outliers using IQR
            Q1 = window_metrics[f'{col}_p25']
            Q3 = window_metrics[f'{col}_p75']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            window_metrics[f'{col}_outliers_count'] = outliers
            window_metrics[f'{col}_outliers_pct'] = (outliers / len(col_data)) * 100 if len(col_data) > 0 else 0
            
            # Distribution metrics (entropy)
            try:
                # Create histogram
                hist, bin_edges = np.histogram(col_data, bins='auto', density=True)
                # Remove zeros to avoid log(0)
                hist = hist[hist > 0]
                # Calculate entropy
                if len(hist) > 0:
                    entropy = -np.sum(hist * np.log(hist))
                    window_metrics[f'{col}_entropy'] = entropy
                else:
                    window_metrics[f'{col}_entropy'] = 0
            except:
                window_metrics[f'{col}_entropy'] = np.nan
                
            # Unique values
            window_metrics[f'{col}_unique_count'] = col_data.nunique()
            window_metrics[f'{col}_unique_pct'] = (col_data.nunique() / len(col_data)) * 100
            
            # Zero values
            zero_count = (col_data == 0).sum()
            window_metrics[f'{col}_zero_count'] = zero_count
            window_metrics[f'{col}_zero_pct'] = (zero_count / len(col_data)) * 100
            
            # Negative values
            neg_count = (col_data < 0).sum()
            window_metrics[f'{col}_negative_count'] = neg_count
            window_metrics[f'{col}_negative_pct'] = (neg_count / len(col_data)) * 100
            
            # Rate of change (velocity) for selected columns
            if rate_metrics is not None and col in rate_metrics:
                if len(col_data) > 1:
                    try:
                        window_metrics[f'{col}_velocity'] = (col_data.iloc[-1] - col_data.iloc[0]) / (len(col_data))
                    except:
                        window_metrics[f'{col}_velocity'] = np.nan
            
            # Value stability - coefficient of variation (lower is more stable)
            if window_metrics[f'{col}_mean'] != 0:
                window_metrics[f'{col}_cv'] = window_metrics[f'{col}_std'] / abs(window_metrics[f'{col}_mean'])
            else:
                window_metrics[f'{col}_cv'] = np.nan
        
        # -----------------------------------
        # Calculate metrics for categorical columns
        # -----------------------------------
        for col in cat_cols:
            # Get non-missing values
            col_data = window_df[col].dropna()
            
            # Skip if no data
            if len(col_data) == 0:
                window_metrics[f'{col}_missing_pct'] = 100.0
                continue
            
            # Basic counts
            window_metrics[f'{col}_count'] = len(col_data)
            window_metrics[f'{col}_missing_count'] = len(window_df) - len(col_data)
            window_metrics[f'{col}_missing_pct'] = ((len(window_df) - len(col_data)) / len(window_df)) * 100
            window_metrics[f'{col}_unique_count'] = col_data.nunique()
            window_metrics[f'{col}_unique_pct'] = (col_data.nunique() / len(col_data)) * 100
            
            # Most common value
            if len(col_data) > 0:
                most_common = col_data.value_counts().nlargest(1)
                if not most_common.empty:
                    window_metrics[f'{col}_most_common'] = str(most_common.index[0])
                    window_metrics[f'{col}_most_common_count'] = most_common.iloc[0]
                    window_metrics[f'{col}_most_common_pct'] = (most_common.iloc[0] / len(col_data)) * 100
            
            # Entropy (category distribution)
            if len(col_data) > 0:
                value_counts = col_data.value_counts(normalize=True)
                entropy = -np.sum(value_counts * np.log(value_counts))
                window_metrics[f'{col}_entropy'] = entropy
                
            # Category distribution stability
            if len(col_data) > 0:
                # Get top 5 categories
                top_categories = col_data.value_counts(normalize=True).nlargest(5)
                for i, (cat, pct) in enumerate(top_categories.items(), 1):
                    window_metrics[f'{col}_top{i}_category'] = str(cat)
                    window_metrics[f'{col}_top{i}_pct'] = pct * 100
        
        # -----------------------------------
        # Calculate relationship metrics (if correlation_columns provided)
        # -----------------------------------
        if correlation_columns is not None and len(correlation_columns) >= 2:
            # Filter to include only columns that exist in the current window
            avail_corr_cols = [col for col in correlation_columns if col in window_df.columns]
            
            if len(avail_corr_cols) >= 2:
                try:
                    corr_matrix = window_df[avail_corr_cols].corr()
                    
                    # Extract key correlations
                    for i, col1 in enumerate(avail_corr_cols):
                        for j, col2 in enumerate(avail_corr_cols):
                            if i < j:  # Only store upper triangle to avoid duplicates
                                window_metrics[f'corr_{col1}_{col2}'] = corr_matrix.loc[col1, col2]
                    
                    # Average absolute correlation
                    corr_values = corr_matrix.values
                    # Get upper triangle indices
                    mask = np.triu(np.ones(corr_values.shape), k=1).astype(bool)
                    # Get upper triangle values
                    upper_triangle = corr_values[mask]
                    # Calculate average absolute correlation
                    window_metrics['avg_abs_correlation'] = np.mean(np.abs(upper_triangle))
                except:
                    window_metrics['avg_abs_correlation'] = np.nan
        
        # -----------------------------------
        # Calculate business metrics (if provided)
        # -----------------------------------
        if business_metrics is not None:
            for metric_name, cols_to_aggregate in business_metrics.items():
                # Filter to include only columns that exist in the current window
                avail_cols = [col for col in cols_to_aggregate if col in window_df.columns]
                
                if not avail_cols:
                    continue
                
                for col in avail_cols:
                    # Sum
                    try:
                        if f'total_{metric_name}' not in window_metrics:
                            window_metrics[f'total_{metric_name}'] = window_df[col].sum()
                        else:
                            window_metrics[f'total_{metric_name}'] += window_df[col].sum()
                    except:
                        pass
                    
                    # Average
                    try:
                        window_metrics[f'avg_{col}_{metric_name}'] = window_df[col].mean()
                    except:
                        pass
                
                # Ratios between columns if multiple columns are provided
                if len(avail_cols) >= 2:
                    for i, col1 in enumerate(avail_cols):
                        for j, col2 in enumerate(avail_cols):
                            if i < j:  # Only calculate each pair once
                                try:
                                    if window_df[col2].sum() != 0:
                                        window_metrics[f'ratio_{col1}_{col2}_{metric_name}'] = window_df[col1].sum() / window_df[col2].sum()
                                except:
                                    pass
        
        # Add all metrics for this window
        profile_metrics.append(window_metrics)
    
    # Convert to DataFrame
    profile_df = pd.DataFrame(profile_metrics)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"✅ Step 3: Profiling complete in {elapsed_time:.2f} seconds")
    print(f"🧾 Generated {len(profile_df)} profile records with {len(profile_df.columns)} metrics each")
    
    return profile_df

def identify_anomalies_from_profile(profile_df: pd.DataFrame, threshold: float = 3.0, 
                                   min_anomalies_for_alert: int = 3,
                                   exclude_metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Identify anomalies from profiling metrics using multiple detection methods
    
    Parameters:
    -----------
    profile_df : pd.DataFrame
        Profiling metrics 
    threshold : float
        Z-score threshold for anomaly detection
    min_anomalies_for_alert : int
        Minimum number of anomalies required to flag a window as anomalous
    exclude_metrics : Optional[List[str]]
        List of metrics to exclude from anomaly detection
        
    Returns:
    --------
    DataFrame with anomaly flags
    """
    print("🚨 Step 1: Identifying anomalies in profiling metrics...")
    
    # Create copy to avoid modifying the original
    anomaly_df = profile_df.copy()
    
    # Get numerical columns from the profile
    num_cols = profile_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Initialize exclude_metrics if not provided
    if exclude_metrics is None:
        exclude_metrics = []
    
    # Initialize anomaly columns
    anomaly_df['total_anomalies'] = 0
    anomaly_df['anomaly_score'] = 0
    
    # -------------------------------
    # Method 1: Z-score based anomalies
    # -------------------------------
    print("📊 Applying Z-score based anomaly detection...")
    
    # Select only metrics columns that we want to analyze (skip metadata)
    metrics_to_analyze = [col for col in num_cols if 
                         not any(col.startswith(prefix) for prefix in 
                                ['window_', 'total_anomalies', 'anomaly_score']) and
                         not any(excluded in col for excluded in exclude_metrics)]
    
    # Dictionary to store anomaly flags
    anomaly_flags = {}
    
    # Apply Z-score method to each metric
    for col in metrics_to_analyze:
        # Skip columns with too many missing values
        if profile_df[col].isna().sum() > len(profile_df) * 0.3:
            continue
            
        # Calculate Z-scores
        mean_val = profile_df[col].mean()
        std_val = profile_df[col].std()
        
        if std_val > 0:
            z_scores = (profile_df[col] - mean_val) / std_val
            
            # Flag anomalies
            anomaly_flags[f'{col}_zscore_anomaly'] = (abs(z_scores) > threshold).astype(int)
            
            # Add to total anomalies
            anomaly_df['total_anomalies'] += anomaly_flags[f'{col}_zscore_anomaly']
            
            # Add to anomaly score (weighted by z-score magnitude)
            anomaly_df['anomaly_score'] += abs(z_scores) * anomaly_flags[f'{col}_zscore_anomaly']
    
    # -------------------------------
    # Method 2: IQR based anomalies
    # -------------------------------
    print("📊 Applying IQR based anomaly detection...")
    
    for col in metrics_to_analyze:
        # Skip columns with too many missing values
        if profile_df[col].isna().sum() > len(profile_df) * 0.3:
            continue
            
        # Calculate IQR
        Q1 = profile_df[col].quantile(0.25)
        Q3 = profile_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Flag anomalies
            anomaly_flags[f'{col}_iqr_anomaly'] = ((profile_df[col] < lower_bound) | 
                                                  (profile_df[col] > upper_bound)).astype(int)
            
            # Add to total anomalies
            anomaly_df['total_anomalies'] += anomaly_flags[f'{col}_iqr_anomaly']
            
            # Add to anomaly score (weighted by distance from bounds)
            distance = np.maximum(
                np.maximum(lower_bound - profile_df[col], 0),
                np.maximum(profile_df[col] - upper_bound, 0)
            ) / IQR
            
            anomaly_df['anomaly_score'] += distance * anomaly_flags[f'{col}_iqr_anomaly']
    
    # -------------------------------
    # Method 3: Rate of change anomalies
    # -------------------------------
    print("📊 Applying rate of change anomaly detection...")
    
    # Process adjacent window pairs
    if len(profile_df) > 1 and 'window_start' in profile_df.columns:
        # Sort by window_start
        profile_df_sorted = profile_df.sort_values('window_start')
        
        # For each metric column that might indicate rate changes
        rate_metrics = [col for col in metrics_to_analyze if 
                       any(substring in col for substring in 
                           ['_pct', '_mean', '_total', '_count', '_ratio'])]
        
        for col in rate_metrics:
            # Skip columns with too many missing values
            if profile_df[col].isna().sum() > len(profile_df) * 0.3:
                continue
                
            # Calculate rate of change
            profile_df_sorted[f'{col}_change'] = profile_df_sorted[col].pct_change()
            
            # Calculate mean and std of changes
            mean_change = profile_df_sorted[f'{col}_change'].mean()
            std_change = profile_df_sorted[f'{col}_change'].std()
            
            if std_change > 0:
                # Z-score of changes
                z_change = (profile_df_sorted[f'{col}_change'] - mean_change) / std_change
                
                # Flag as anomaly if change is too rapid
                change_anomaly = (abs(z_change) > threshold).astype(int)
                
                # Map back to original index
                anomaly_flags[f'{col}_change_anomaly'] = pd.Series(
                    change_anomaly.values, 
                    index=profile_df_sorted.index
                ).reindex(anomaly_df.index).fillna(0).astype(int)
                
                # Add to total anomalies
                anomaly_df['total_anomalies'] += anomaly_flags[f'{col}_change_anomaly']
                
                # Add to anomaly score
                anomaly_df['anomaly_score'] += abs(z_change).reindex(anomaly_df.index).fillna(0) * anomaly_flags[f'{col}_change_anomaly']
    
    # -------------------------------
    # Method 4: Correlation structure anomalies
    # -------------------------------
    print("📊 Applying correlation structure anomaly detection...")
    
    # Get correlation columns
    corr_cols = [col for col in num_cols if col.startswith('corr_') and
                not any(excluded in col for excluded in exclude_metrics)]
    
    if len(corr_cols) > 0:
        # For each correlation
        for col in corr_cols:
            # Skip columns with too many missing values
            if profile_df[col].isna().sum() > len(profile_df) * 0.3:
                continue
                
            # Calculate Z-scores of correlation
            mean_corr = profile_df[col].mean()
            std_corr = profile_df[col].std()
            
            if std_corr > 0:
                z_corr = (profile_df[col] - mean_corr) / std_corr
                
                # Flag as anomaly if correlation changes significantly
                corr_anomaly = (abs(z_corr) > threshold).astype(int)
                
                # Store flag
                anomaly_flags[f'{col}_anomaly'] = corr_anomaly
                
                # Add to total anomalies
                anomaly_df['total_anomalies'] += anomaly_flags[f'{col}_anomaly']
                
                # Add to anomaly score
                anomaly_df['anomaly_score'] += abs(z_corr) * anomaly_flags[f'{col}_anomaly']
    
    # -------------------------------
    # Add all anomaly flags to the dataframe
    # -------------------------------
    for flag_col, flag_values in anomaly_flags.items():
        anomaly_df[flag_col] = flag_values
    
    # Normalize anomaly score
    if anomaly_df['anomaly_score'].max() > 0:
        anomaly_df['anomaly_score'] = anomaly_df['anomaly_score'] / anomaly_df['anomaly_score'].max() * 10
    
    # Final anomaly classification
    anomaly_df['is_anomaly'] = (anomaly_df['total_anomalies'] >= min_anomalies_for_alert).astype(int)
    
    print(f"✅ Anomaly detection completed. Found {anomaly_df['is_anomaly'].sum()} anomalous windows.")
    return anomaly_df

def apply_pyod_to_profile(profile_df: pd.DataFrame, contamination: float = 0.05):
    """
    Apply PyOD anomaly detection models to profiling metrics
    
    Parameters:
    -----------
    profile_df : pd.DataFrame
        Profiling metrics
    contamination : float
        Expected proportion of anomalies
        
    Returns:
    --------
    DataFrame with PyOD anomaly scores and labels
    """
    try:
        from pyod.models.ecod import ECOD
        from pyod.models.cblof import CBLOF
    except ImportError:
        print("⚠️ PyOD not installed. Install with 'pip install pyod'")
        return profile_df
    
    print("🔍 Step 1: Preparing data for PyOD...")
    
    # Create copy to avoid modifying the original
    pyod_df = profile_df.copy()
    
    # Select numeric features, exclude timestamp and any existing anomaly columns
    features = profile_df.select_dtypes(include=['float64', 'int64']).columns
    features = [col for col in features if not any(col.startswith(prefix) for prefix in 
                                                 ['window_', 'is_anomaly', 'anomaly_', 'total_anomalies'])]
    
    if len(features) < 2:
        print("⚠️ Not enough numeric features for PyOD")
        return profile_df
    
    # Handle missing values
    X = profile_df[features].fillna(profile_df[features].mean())
    
    print(f"📊 Step 2: Applying PyOD models to {len(features)} features...")
    
    # Apply ECOD model (Empirical Cumulative Distribution Functions for Outlier Detection)
    ecod = ECOD(contamination=contamination)
    ecod.fit(X)
    
    # Apply CBLOF model (Cluster-Based Local Outlier Factor)
    n_clusters = min(8, len(X) // 10 + 1)  # Adaptive number of clusters
    cblof = CBLOF(contamination=contamination, n_clusters=max(2, n_clusters))
    cblof.fit(X)
    
    # Add results to dataframe
    pyod_df['ecod_score'] = ecod.decision_scores_
    pyod_df['ecod_label'] = ecod.labels_
    pyod_df['cblof_score'] = cblof.decision_scores_
    pyod_df['cblof_label'] = cblof.labels_
    
    # Combined anomaly flag
    pyod_df['pyod_is_anomaly'] = ((pyod_df['ecod_label'] == 1) | (pyod_df['cblof_label'] == 1)).astype(int)
    
    print(f"✅ PyOD anomaly detection complete. Found {pyod_df['pyod_is_anomaly'].sum()} anomalies.")
    return pyod_df

def apply_telemanom_to_profile(profile_df: pd.DataFrame, sequence_cols: List[str],
                             train_ratio: float = 0.7):
    """
    Apply Telemanom time-series anomaly detection to profiling metrics
    
    Parameters:
    -----------
    profile_df : pd.DataFrame
        Profiling metrics with a 'window_start' column
    sequence_cols : List[str]
        List of columns to apply time-series anomaly detection to
    train_ratio : float
        Proportion of data to use for training (0.0-1.0)
        
    Returns:
    --------
    DataFrame with Telemanom anomaly scores
    """
    try:
        from telemanom import detector
    except ImportError:
        print("⚠️ Telemanom not installed. Install with 'pip install telemanom'")
        return profile_df
    
    if 'window_start' not in profile_df.columns:
        print("⚠️ Cannot apply Telemanom without 'window_start' column")
        return profile_df
    
    # Sort by window_start
    sorted_df = profile_df.sort_values('window_start').reset_index(drop=True)
    
    # Create copy for results
    telemanom_df = sorted_df.copy()
    
    # Initialize columns for anomaly scores
    for col in sequence_cols:
        telemanom_df[f'{col}_telemanom_score'] = 0.0
    
    print(f"🔍 Step 1: Applying Telemanom to {len(sequence_cols)} sequence columns...")
    
    # Split into train/test for each column
    train_size = int(len(sorted_df) * train_ratio)
    
    for col in sequence_cols:
        # Skip if column doesn't exist
        if col not in sorted_df.columns:
            continue
            
        # Skip if too many missing values
        if sorted_df[col].isna().sum() > len(sorted_df) * 0.3:
            continue
            
        # Prepare data for Telemanom (reshape to 2D)
        series = sorted_df[col].fillna(method='ffill').fillna(method='bfill').values.reshape(-1, 1)
        
        # Ensure we have enough data
        if len(series) < 30:
            print(f"⚠️ Not enough data points for '{col}' to apply Telemanom")
            continue
            
        # Configure detector
        config = {
            'batch_size': min(64, len(series) // 4),
            'lookback': min(10, len(series) // 10),
            'error_buffer': min(5, len(series) // 20),
            'lstm_batch_size': min(64, len(series) // 4),
            'p': 0.05  # Significance level
        }
        
        try:
            print(f"  Processing column: {col}")
            # Run detection
            scores, _ = detector.detect(
                series[:train_size], 
                series[train_size:],
                config
            )
            
            # Map scores back to dataframe
            telemanom_scores = np.zeros(len(sorted_df))
            telemanom_scores[train_size:] = scores
            telemanom_df[f'{col}_telemanom_score'] = telemanom_scores
            
        except Exception as e:
            print(f"⚠️ Error applying Telemanom to '{col}': {str(e)}")
    
    # Determine overall telemanom anomaly flag
    telemanom_score_cols = [col for col in telemanom_df.columns if col.endswith('_telemanom_score')]
    
    if telemanom_score_cols:
        # Create combined score (maximum of all individual scores)
        telemanom_df['telemanom_max_score'] = telemanom_df[telemanom_score_cols].max(axis=1)
        
        # Flag as anomaly if any score is above threshold (typical threshold is around 0.5)
        telemanom_df['telemanom_is_anomaly'] = (telemanom_df['telemanom_max_score'] > 0.5).astype(int)
        
        print(f"✅ Telemanom anomaly detection complete. Found {telemanom_df['telemanom_is_anomaly'].sum()} anomalies.")
    else:
        print("⚠️ No telemanom scores were calculated.")
    
    return telemanom_df

def visualize_profiling_results(profile_df: pd.DataFrame, anomaly_df: Optional[pd.DataFrame] = None,
                               key_metrics: Optional[List[str]] = None, 
                               save_path: Optional[str] = None):
    """
    Visualize profiling metrics and highlight anomalies
    
    Parameters:
    -----------
    profile_df : pd.DataFrame
        Profiling metrics
    anomaly_df : Optional[pd.DataFrame]
        Anomaly detection results (if None, will look for anomaly columns in profile_df)
    key_metrics : Optional[List[str]]
        List of key metrics to visualize (if None, will select automatically)
    save_path : Optional[str]
        Path to save visualizations (if None, displays instead)
        
    Returns:
    --------
    None
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠️ Visualization requires matplotlib and seaborn")
        return
    
    print("📈 Creating visualizations of profiling metrics and anomalies...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'figure.figsize': (15, 10)})
    
    # If anomaly_df is None, check if profile_df has anomaly columns
    has_anomalies = False
    if anomaly_df is None:
        if 'is_anomaly' in profile_df.columns or 'anomaly_score' in profile_df.columns:
            anomaly_df = profile_df
            has_anomalies = True
    else:
        has_anomalies = True
    
    # Ensure we have a datetime index
    if 'window_start' in profile_df.columns:
        profile_df = profile_df.set_index('window_start')
        
    if has_anomalies and 'window_start' in anomaly_df.columns:
        anomaly_df = anomaly_df.set_index('window_start')
    
    # If no key metrics provided, try to identify some interesting ones
    if key_metrics is None:
        num_cols = profile_df.select_dtypes(include=['float64', 'int64']).columns
        
        # Try to find important metrics
        potential_metrics = []
        
        # Check for common important metrics
        for metric in ['row_count', 'missing_values_pct', 'duplicate_rows_pct']:
            if metric in profile_df.columns:
                potential_metrics.append(metric)
        
        # Look for metrics with high variance (may be more interesting)
        variance_metrics = []
        for col in num_cols:
            if col not in potential_metrics and not col.startswith('window_') and not 'anomaly' in col.lower():
                try:
                    variance_metrics.append((col, profile_df[col].std() / profile_df[col].mean()))
                except:
                    pass
        
        # Sort by coefficient of variation and take top metrics
        variance_metrics.sort(key=lambda x: x[1], reverse=True)
        potential_metrics.extend([m[0] for m in variance_metrics[:5]])
        
        key_metrics = potential_metrics[:6]  # Limit to 6 metrics
    
    # Create figure for overall metrics
    fig = plt.figure(figsize=(15, 10))
    
    if has_anomalies and 'total_anomalies' in anomaly_df.columns:
        # Plot 1: Total anomalies over time
        plt.subplot(3, 1, 1)
        plt.plot(anomaly_df.index, anomaly_df['total_anomalies'], 'b-', linewidth=2)
        plt.fill_between(anomaly_df.index, anomaly_df['total_anomalies'], alpha=0.3)
        plt.title('Total Anomalies Over Time')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly score over time
        if 'anomaly_score' in anomaly_df.columns:
            plt.subplot(3, 1, 2)
            plt.plot(anomaly_df.index, anomaly_df['anomaly_score'], 'r-', linewidth=2)
            plt.fill_between(anomaly_df.index, anomaly_df['anomaly_score'], alpha=0.3)
            plt.title('Anomaly Score Over Time')
            plt.ylabel('Score (0-10)')
            plt.grid(True, alpha=0.3)
            
            # Adjust for 3rd plot
            plt.subplot(3, 1, 3)
        else:
            # Adjust for only 1 more plot
            plt.subplot(2, 1, 2)
    else:
        # Just one plot for row count
        plt.subplot(1, 1, 1)
    
    # Plot row count over time if available
    if 'row_count' in profile_df.columns:
        plt.plot(profile_df.index, profile_df['row_count'], 'g-', linewidth=2)
        plt.fill_between(profile_df.index, profile_df['row_count'], alpha=0.3)
        
        # Highlight anomalous periods
        if has_anomalies and 'is_anomaly' in anomaly_df.columns:
            anomaly_periods = anomaly_df[anomaly_df['is_anomaly'] == 1].index
            for period in anomaly_periods:
                plt.axvline(x=period, color='r', linestyle='--', alpha=0.5)
        
        plt.title('Number of Records Over Time')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/overall_metrics.png", dpi=300, bbox_inches='tight')
    
    # ---------------------------------
    # Plot key metrics with anomalies
    # ---------------------------------
    # Create subplots for each key metric
    if key_metrics:
        n_plots = len(key_metrics)
        n_cols = 2
        n_rows = (n_plots + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        for i, metric in enumerate(key_metrics):
            if i < len(axes) and metric in profile_df.columns:
                ax = axes[i]
                
                # Plot the metric
                ax.plot(profile_df.index, profile_df[metric], 'b-', alpha=0.7)
                
                # Add trend line
                try:
                    from scipy.signal import savgol_filter
                    window_length = min(15, len(profile_df) // 2)
                    # Make window length odd
                    if window_length % 2 == 0:
                        window_length = max(3, window_length - 1)
                    
                    if window_length >= 3:
                        trend = savgol_filter(profile_df[metric].fillna(method='ffill').fillna(method='bfill'), 
                                             window_length, 3)
                        ax.plot(profile_df.index, trend, 'r-', alpha=0.5)
                except:
                    pass
                
                # Highlight anomalies if available
                if has_anomalies:
                    # Check for metric-specific anomaly flags
                    anomaly_flag_col = f'{metric}_zscore_anomaly'
                    if anomaly_flag_col in anomaly_df.columns:
                        anomaly_points = profile_df.index[anomaly_df[anomaly_flag_col] == 1]
                        if len(anomaly_points) > 0:
                            anomaly_values = profile_df.loc[anomaly_points, metric]
                            ax.scatter(anomaly_points, anomaly_values, color='red', s=50, zorder=5)
                    
                    # Otherwise use general anomaly flag
                    elif 'is_anomaly' in anomaly_df.columns:
                        anomaly_points = profile_df.index[anomaly_df['is_anomaly'] == 1]
                        if len(anomaly_points) > 0:
                            anomaly_values = profile_df.loc[anomaly_points, metric]
                            ax.scatter(anomaly_points, anomaly_values, color='red', s=50, zorder=5, alpha=0.5)
                
                ax.set_title(metric)
                ax.grid(True, alpha=0.3)
                
                # Rotate x labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Remove empty subplots
        for i in range(len(key_metrics), len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/key_metrics.png", dpi=300, bbox_inches='tight')
    
    # Display if not saving
    if not save_path:
        plt.show()
    
    print("✅ Visualization complete.")