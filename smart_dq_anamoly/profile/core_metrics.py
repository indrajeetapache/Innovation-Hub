import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional

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
    Generate comprehensive profiling metrics for time-series data.
    """
    start_time = time.time()
    print("\n📊 Step 1: Initializing profiling process...")

    # Validate timestamp column
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in dataframe")

    # Prepare skip columns
    if skip_columns is None:
        skip_columns = []
    if timestamp_col not in skip_columns:
        skip_columns.append(timestamp_col)

    # Convert timestamp
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df_indexed = df.set_index(timestamp_col)

    # Identify numerical and categorical columns
    num_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col not in skip_columns]
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category', 'bool']).columns if col not in skip_columns]
    print(f"🔍 Identified {len(num_cols)} numerical and {len(cat_cols)} categorical columns")

    # Global stats for drift detection
    global_stats = {}
    for col in num_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            global_stats[f'{col}_global_mean'] = col_data.mean()
            global_stats[f'{col}_global_std'] = col_data.std()
            global_stats[f'{col}_global_min'] = col_data.min()
            global_stats[f'{col}_global_max'] = col_data.max()

    profile_metrics = []
    print(f"\n📅 Step 2: Resampling and window-wise processing using '{window}' window")
    
    for i, (window_start, window_df) in enumerate(df_indexed.resample(window), 1):
        if i % 10 == 0 or i == 1:
            print(f"📦 Processing window {i} starting on {window_start.date()}...")

        if len(window_df) == 0:
            continue

        metrics = {'window_start': window_start}
        metrics['row_count'] = len(window_df)
        metrics['column_count'] = len(window_df.columns)
        metrics['missing_values_total'] = window_df.isna().sum().sum()
        metrics['missing_values_pct'] = (metrics['missing_values_total'] / (len(window_df) * len(window_df.columns))) * 100
        metrics['duplicate_rows'] = window_df.duplicated().sum()
        metrics['duplicate_rows_pct'] = (metrics['duplicate_rows'] / len(window_df)) * 100

        for col in num_cols:
            col_data = window_df[col].dropna()
            if len(col_data) == 0:
                metrics[f'{col}_missing_pct'] = 100.0
                continue

            metrics[f'{col}_count'] = len(col_data)
            metrics[f'{col}_mean'] = col_data.mean()
            metrics[f'{col}_std'] = col_data.std()
            metrics[f'{col}_min'] = col_data.min()
            metrics[f'{col}_max'] = col_data.max()
            metrics[f'{col}_p25'] = col_data.quantile(0.25)
            metrics[f'{col}_p75'] = col_data.quantile(0.75)

            # Drift detection
            if f'{col}_global_mean' in global_stats:
                global_mean = global_stats[f'{col}_global_mean']
                global_std = global_stats[f'{col}_global_std']
                metrics[f'{col}_mean_vs_global'] = (metrics[f'{col}_mean'] - global_mean) / global_mean if global_mean != 0 else 0
                metrics[f'{col}_zscore'] = (metrics[f'{col}_mean'] - global_mean) / global_std if global_std > 0 else 0

            # Outliers (IQR)
            IQR = metrics[f'{col}_p75'] - metrics[f'{col}_p25']
            lb = metrics[f'{col}_p25'] - 1.5 * IQR
            ub = metrics[f'{col}_p75'] + 1.5 * IQR
            outliers = ((col_data < lb) | (col_data > ub)).sum()
            metrics[f'{col}_outliers_pct'] = (outliers / len(col_data)) * 100

            if rate_metrics and col in rate_metrics and len(col_data) > 1:
                metrics[f'{col}_velocity'] = (col_data.iloc[-1] - col_data.iloc[0]) / len(col_data)

        for col in cat_cols:
            col_data = window_df[col].dropna()
            if len(col_data) == 0:
                metrics[f'{col}_missing_pct'] = 100.0
                continue

            metrics[f'{col}_unique_count'] = col_data.nunique()
            top_cat = col_data.value_counts().nlargest(1)
            if not top_cat.empty:
                metrics[f'{col}_most_common'] = str(top_cat.index[0])
                metrics[f'{col}_most_common_pct'] = (top_cat.iloc[0] / len(col_data)) * 100

        # Correlation metrics
        if correlation_columns and len(correlation_columns) >= 2:
            valid_corr_cols = [c for c in correlation_columns if c in window_df.columns]
            if len(valid_corr_cols) >= 2:
                corr_matrix = window_df[valid_corr_cols].corr()
                for i, col1 in enumerate(valid_corr_cols):
                    for j, col2 in enumerate(valid_corr_cols):
                        if i < j:
                            metrics[f'corr_{col1}_{col2}'] = corr_matrix.loc[col1, col2]

        # Business metrics
        if business_metrics:
            for metric_name, cols in business_metrics.items():
                for col in cols:
                    if col in window_df.columns:
                        metrics[f'total_{metric_name}_{col}'] = window_df[col].sum()
                        metrics[f'avg_{metric_name}_{col}'] = window_df[col].mean()

        profile_metrics.append(metrics)

    profile_df = pd.DataFrame(profile_metrics)
    elapsed = time.time() - start_time
    print(f"\n✅ Step 3: Profiling complete in {elapsed:.2f} seconds")
    print(f"🧾 Generated {len(profile_df)} windows with {len(profile_df.columns)} metrics")
    return profile_df
