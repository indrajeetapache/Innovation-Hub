import pandas as pd
import numpy as np

def zscore_anomaly_detection(profile_df, threshold=3.0, exclude_metrics=None):
    print("📊 Applying Z-score based anomaly detection...")
    num_cols = profile_df.select_dtypes(include=['float64', 'int64']).columns
    exclude_metrics = exclude_metrics or []
    metrics = [col for col in num_cols if not any(x in col for x in exclude_metrics)]

    z_flags = {}
    for col in metrics:
        if profile_df[col].isna().sum() > len(profile_df) * 0.3:
            continue
        mean, std = profile_df[col].mean(), profile_df[col].std()
        if std > 0:
            z = (profile_df[col] - mean) / std
            z_flags[f'{col}_z_anomaly'] = (abs(z) > threshold).astype(int)
            print(f"  ➤ Processed Z-score for: {col}")
    return z_flags


def iqr_anomaly_detection(profile_df, exclude_metrics=None):
    print("📊 Applying IQR based anomaly detection...")
    num_cols = profile_df.select_dtypes(include=['float64', 'int64']).columns
    exclude_metrics = exclude_metrics or []
    metrics = [col for col in num_cols if not any(x in col for x in exclude_metrics)]

    iqr_flags = {}
    for col in metrics:
        if profile_df[col].isna().sum() > len(profile_df) * 0.3:
            continue
        Q1, Q3 = profile_df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        if IQR > 0:
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            iqr_flags[f'{col}_iqr_anomaly'] = ((profile_df[col] < lower) | (profile_df[col] > upper)).astype(int)
            print(f"  ➤ Processed IQR for: {col}")
    return iqr_flags


def rate_change_anomaly_detection(profile_df, threshold=3.0, exclude_metrics=None):
    print("📊 Applying rate of change anomaly detection...")
    profile_df = profile_df.sort_values('window_start')
    exclude_metrics = exclude_metrics or []
    rate_flags = {}
    
    for col in profile_df.select_dtypes(include=['float64', 'int64']).columns:
        if 'window_start' in col or any(x in col for x in exclude_metrics):
            continue
        changes = profile_df[col].pct_change()
        mean, std = changes.mean(), changes.std()
        if std > 0:
            z = (changes - mean) / std
            rate_flags[f'{col}_rate_change_anomaly'] = (abs(z) > threshold).astype(int)
            print(f"  ➤ Processed rate change for: {col}")
    return rate_flags


def correlation_anomaly_detection(profile_df, threshold=3.0):
    print("📊 Applying correlation anomaly detection...")
    corr_flags = {}
    corr_cols = [col for col in profile_df.columns if col.startswith('corr_')]
    for col in corr_cols:
        if profile_df[col].isna().sum() > len(profile_df) * 0.3:
            continue
        mean, std = profile_df[col].mean(), profile_df[col].std()
        if std > 0:
            z = (profile_df[col] - mean) / std
            corr_flags[f'{col}_corr_anomaly'] = (abs(z) > threshold).astype(int)
            print(f"  ➤ Processed correlation for: {col}")
    return corr_flags


def aggregate_anomalies(profile_df, *anomaly_dicts, min_flags=3):
    print("📦 Aggregating anomaly flags and computing anomaly scores...")
    result = profile_df.copy()
    result['total_anomalies'] = 0
    result['anomaly_score'] = 0

    for anomaly_dict in anomaly_dicts:
        for col, flags in anomaly_dict.items():
            result[col] = flags
            result['total_anomalies'] += flags
            result['anomaly_score'] += flags  # Simplified scoring

    result['is_anomaly'] = (result['total_anomalies'] >= min_flags).astype(int)
    print(f"✅ Detected {result['is_anomaly'].sum()} anomalous windows.")
    return result
