"""
# Transforms profile metrics into features usable by LSTM/PYOD
# Creates temporal features from historical comparisons
# Implements customized feature sets based on column data types
"""
# feature_engineering.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union

class FeatureEngineer:
    """Transforms profile metrics into enhanced features for anomaly detection."""
    
    def __init__(self, zscore_threshold: float = 3.0):
        """
        Initialize feature engineer.
        
        Args:
            zscore_threshold: Threshold for flagging z-score outliers
        """
        self.zscore_threshold = zscore_threshold
        print(f"[INIT] FeatureEngineer initialized with zscore_threshold={zscore_threshold}")
        
    def create_features(self, 
                       profile_df: pd.DataFrame, 
                       process_date_col: str,
                       target_cols: List[str] = None) -> pd.DataFrame:
        """
        Create enhanced features from profile metrics.
        
        Args:
            profile_df: DataFrame with profile metrics
            process_date_col: Column containing process dates
            target_cols: Target columns for specialized features
            
        Returns:
            DataFrame with additional engineered features
        """
        print(f"[FEATURES] Creating enhanced features from {profile_df.shape[1]} profile metrics")
        
        # Create copy to avoid modifying original
        result_df = profile_df.copy()
        
        # Extract profile columns
        profile_cols = [col for col in profile_df.columns if any(
            suffix in col for suffix in ['_mean', '_median', '_std', '_zscore', '_diff', '_ratio']
        )]
        
        print(f"[COLUMNS] Found {len(profile_cols)} profile metric columns")
        
        # Create global anomaly features
        result_df = self._create_zscore_features(result_df, profile_cols)
        result_df = self._create_combined_anomaly_scores(result_df, profile_cols)
        
        # Create time-based features
        result_df = self._add_time_features(result_df, process_date_col)
        
        # Create target-specific features if provided
        if target_cols:
            print(f"[TARGETS] Creating specialized features for {len(target_cols)} target columns")
            result_df = self._create_target_specific_features(result_df, target_cols)
        
        print(f"[COMPLETE] Created {result_df.shape[1] - profile_df.shape[1]} new features")
        return result_df
    
    def _create_zscore_features(self, df: pd.DataFrame, profile_cols: List[str]) -> pd.DataFrame:
        """Create binary anomaly flags from z-score columns."""
        result_df = df.copy()
        
        # Find zscore columns
        zscore_cols = [col for col in profile_cols if '_zscore_' in col]
        print(f"[ZSCORE] Creating flags for {len(zscore_cols)} z-score metrics")
        
        # Create flags for each zscore column
        for col in zscore_cols:
            base_name = col.split('_zscore_')[0]
            window = col.split('_zscore_')[1]
            flag_name = f"{base_name}_anomaly_{window}"
            
            # Create flag based on threshold
            result_df[flag_name] = (result_df[col].abs() > self.zscore_threshold).astype(int)
        
        return result_df
    
    def _create_combined_anomaly_scores(self, df: pd.DataFrame, profile_cols: List[str]) -> pd.DataFrame:
        """Create combined anomaly scores across different metrics."""
        result_df = df.copy()
        
        # Group columns by base feature name
        base_features = set()
        for col in profile_cols:
            if '_mean' in col or '_median' in col or '_std' in col:
                base_name = col.split('_')[0]
                base_features.add(base_name)
        
        print(f"[COMBINED] Creating combined scores for {len(base_features)} base features")
        
        # For each base feature, create combined score
        for base in base_features:
            # Get relevant columns
            feature_cols = [col for col in profile_cols if col.startswith(f"{base}_") and 
                           ('_diff_' in col or '_ratio_' in col or '_zscore_' in col)]
            
            if len(feature_cols) > 1:
                # Calculate combined score (mean of absolute z-scores)
                combined_col = f"{base}_combined_anomaly_score"
                
                # Normalize each column and take absolute value
                normalized_cols = []
                for col in feature_cols:
                    # Skip columns with all NaN
                    if df[col].isna().all():
                        continue
                        
                    # Normalize using z-score normalization
                    mean, std = df[col].mean(), df[col].std()
                    if std > 0:
                        normalized_cols.append(abs((df[col] - mean) / std))
                
                # Calculate combined score if we have normalized columns
                if normalized_cols:
                    result_df[combined_col] = sum(normalized_cols) / len(normalized_cols)
                    
                    # Create flag for extreme values
                    result_df[f"{base}_combined_anomaly_flag"] = (
                        result_df[combined_col] > 3.0
                    ).astype(int)
        
        return result_df
    
    def _add_time_features(self, df: pd.DataFrame, process_date_col: str) -> pd.DataFrame:
        """Add time-based features."""
        result_df = df.copy()
        
        # Ensure date column is datetime
        if process_date_col in result_df.columns:
            result_df[process_date_col] = pd.to_datetime(result_df[process_date_col])
            
            # Extract date components
            result_df['day_of_week'] = result_df[process_date_col].dt.dayofweek
            result_df['day_of_month'] = result_df[process_date_col].dt.day
            result_df['month'] = result_df[process_date_col].dt.month
            result_df['is_month_end'] = result_df[process_date_col].dt.is_month_end.astype(int)
            result_df['is_quarter_end'] = result_df[process_date_col].dt.is_quarter_end.astype(int)
            
            print("[TIME] Added calendar-based time features")
        
        return result_df
    
    def _create_target_specific_features(self, df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        """Create features specific to target columns."""
        result_df = df.copy()
        
        for target in target_cols:
            # Find relevant profile columns for this target
            target_profile_cols = [col for col in df.columns if col.startswith(f"{target}_")]
            
            if not target_profile_cols:
                print(f"[WARNING] No profile columns found for target {target}")
                continue
                
            print(f"[TARGET] Creating specialized features for {target} from {len(target_profile_cols)} metrics")
            
            # Find anomaly flags for this target
            anomaly_flags = [col for col in target_profile_cols if '_anomaly_' in col]
            
            if anomaly_flags:
                # Create combined anomaly flag
                result_df[f"{target}_any_anomaly"] = result_df[anomaly_flags].max(axis=1)
                
                # Create anomaly severity (count of flags)
                result_df[f"{target}_anomaly_count"] = result_df[anomaly_flags].sum(axis=1)
            
        return result_df
