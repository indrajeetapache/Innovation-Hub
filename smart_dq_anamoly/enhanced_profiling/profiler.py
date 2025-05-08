"""
# Core profiling logic that calculates metrics per date
# Will detect outliers and pattern breaks by comparing with historical data
# Focus on numerical features with appropriate handling for other types
"""

# profiler.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

class TimeSeriesProfiler:
    """Creates statistical profiles for time series data by process date."""
    
    def __init__(self, window_sizes: List[int] = [7, 14, 30, 90]):
        """
        Initialize profiler with window sizes for historical comparisons.
        
        Args:
            window_sizes: List of lookback window sizes in days
        """
        self.window_sizes = window_sizes
        print(f"[INIT] TimeSeriesProfiler initialized with windows: {window_sizes}")
        
    def create_profiles(self, 
                      df: pd.DataFrame, 
                      process_date_col: str,
                      ignore_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Create profiles for each date in the DataFrame.
        
        Args:
            df: Input DataFrame
            process_date_col: Column containing process dates
            ignore_cols: Columns to ignore in profiling
            
        Returns:
            Tuple of (enhanced DataFrame, dictionary of individual profiles by date)
        """
        # Validate inputs
        if process_date_col not in df.columns:
            raise ValueError(f"Process date column '{process_date_col}' not found in DataFrame")
            
        # Ensure process_date is datetime
        df = df.copy()
        df[process_date_col] = pd.to_datetime(df[process_date_col])
        
        # Identify columns to profile
        if ignore_cols is None:
            ignore_cols = []
        ignore_cols.append(process_date_col)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ignore_cols]
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ignore_cols]
        
        print(f"[PROFILE] Profiling {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
        
        # Sort by process date
        df = df.sort_values(by=process_date_col)
        
        # Get unique dates
        unique_dates = df[process_date_col].unique()
        print(f"[DATES] Found {len(unique_dates)} unique process dates")
        
        # Create profiles for each date
        all_profiles = {}
        for date in unique_dates:
            print(f"[PROCESSING] Creating profile for {date}")
            date_df = df[df[process_date_col] == date]
            
            # Create historical df (all data before this date)
            hist_df = df[df[process_date_col] < date]
            
            # Skip if insufficient history
            if len(hist_df) == 0:
                print(f"  [SKIP] No historical data available for {date}")
                continue
                
            # Create profile
            profile = self._create_single_date_profile(
                date_df=date_df,
                hist_df=hist_df,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols
            )
            
            # Store profile
            all_profiles[date] = profile
        
        # Combine profiles into enhanced DataFrame
        enhanced_df = self._merge_profiles_with_original(df, all_profiles, process_date_col)
        
        print(f"[COMPLETE] Created {len(all_profiles)} profiles and enhanced DataFrame with {enhanced_df.shape[1]} columns")
        return enhanced_df, all_profiles
        
    def _create_single_date_profile(self,
                                  date_df: pd.DataFrame,
                                  hist_df: pd.DataFrame,
                                  numeric_cols: List[str],
                                  categorical_cols: List[str]) -> pd.DataFrame:
        """Create profile for a single date, comparing with historical data."""
        profiles = []
        
        # Profile each numeric column
        for col in numeric_cols:
            col_profile = self._profile_numeric_column(
                date_series=date_df[col],
                hist_series=hist_df[col],
                col_name=col
            )
            profiles.append(col_profile)
            
        # Profile each categorical column
        for col in categorical_cols:
            col_profile = self._profile_categorical_column(
                date_series=date_df[col],
                hist_series=hist_df[col],
                col_name=col
            )
            profiles.append(col_profile)
            
        # Combine all column profiles
        if profiles:
            return pd.concat(profiles, axis=1)
        else:
            return pd.DataFrame()
    
    def _profile_numeric_column(self,
                              date_series: pd.Series,
                              hist_series: pd.Series,
                              col_name: str) -> pd.DataFrame:
        """Profile a numeric column, comparing current day with history."""
        # Calculate statistics for current date
        current_stats = {
            f"{col_name}_mean": date_series.mean(),
            f"{col_name}_median": date_series.median(),
            f"{col_name}_std": date_series.std(),
            f"{col_name}_min": date_series.min(),
            f"{col_name}_max": date_series.max(),
            f"{col_name}_range": date_series.max() - date_series.min(),
            f"{col_name}_iqr": date_series.quantile(0.75) - date_series.quantile(0.25),
            f"{col_name}_skew": date_series.skew(),
            f"{col_name}_kurtosis": date_series.kurtosis(),
            f"{col_name}_zeros_pct": (date_series == 0).mean() * 100,
            f"{col_name}_nulls_pct": date_series.isna().mean() * 100
        }
        
        # Calculate historical statistics for window periods
        for window in self.window_sizes:
            # Get last N days from history
            if len(hist_series) >= window:
                last_n = hist_series.iloc[-window:]
                
                # Calculate window statistics
                current_stats.update({
                    f"{col_name}_mean_diff_{window}d": current_stats[f"{col_name}_mean"] - last_n.mean(),
                    f"{col_name}_median_diff_{window}d": current_stats[f"{col_name}_median"] - last_n.median(),
                    f"{col_name}_std_ratio_{window}d": current_stats[f"{col_name}_std"] / max(last_n.std(), 1e-8),
                    f"{col_name}_range_ratio_{window}d": current_stats[f"{col_name}_range"] / max((last_n.max() - last_n.min()), 1e-8)
                })
                
                # Z-score compared to window history
                z_score = (current_stats[f"{col_name}_mean"] - last_n.mean()) / max(last_n.std(), 1e-8)
                current_stats[f"{col_name}_zscore_{window}d"] = z_score
        
        # Create and return profile DataFrame
        profile_df = pd.DataFrame([current_stats])
        return profile_df
    
    def _profile_categorical_column(self,
                                  date_series: pd.Series,
                                  hist_series: pd.Series,
                                  col_name: str) -> pd.DataFrame:
        """Profile a categorical column, comparing current day with history."""
        # Count current values
        current_counts = date_series.value_counts(normalize=True, dropna=False)
        
        # Calculate statistics for current date
        current_stats = {
            f"{col_name}_unique_count": date_series.nunique(),
            f"{col_name}_mode": date_series.mode().iloc[0] if not date_series.mode().empty else None,
            f"{col_name}_entropy": self._calculate_entropy(current_counts),
            f"{col_name}_nulls_pct": date_series.isna().mean() * 100
        }
        
        # Calculate historical statistics
        for window in self.window_sizes:
            # Get last N days from history
            if len(hist_series) >= window:
                last_n = hist_series.iloc[-window:]
                hist_counts = last_n.value_counts(normalize=True, dropna=False)
                
                # Compare current distribution with historical
                current_stats.update({
                    f"{col_name}_unique_diff_{window}d": current_stats[f"{col_name}_unique_count"] - last_n.nunique(),
                    f"{col_name}_js_distance_{window}d": self._jensen_shannon_distance(current_counts, hist_counts),
                    f"{col_name}_new_categories_{window}d": self._count_new_categories(date_series, last_n)
                })
        
        # Create and return profile DataFrame
        profile_df = pd.DataFrame([current_stats])
        return profile_df
        
    def _calculate_entropy(self, counts: pd.Series) -> float:
        """Calculate entropy of a distribution."""
        probs = counts.values
        return -np.sum(probs * np.log2(probs + 1e-10))
        
    def _jensen_shannon_distance(self, p: pd.Series, q: pd.Series) -> float:
        """Calculate Jensen-Shannon distance between two distributions."""
        # Get common index
        common_idx = p.index.union(q.index)
        
        # Reindex with 0 for missing values
        p_full = p.reindex(common_idx, fill_value=0)
        q_full = q.reindex(common_idx, fill_value=0)
        
        # Calculate JS distance
        m = 0.5 * (p_full + q_full)
        return 0.5 * (
            np.sum(p_full * np.log2(p_full / (m + 1e-10) + 1e-10)) +
            np.sum(q_full * np.log2(q_full / (m + 1e-10) + 1e-10))
        )
    
    def _count_new_categories(self, current: pd.Series, history: pd.Series) -> int:
        """Count categories in current that weren't in history."""
        current_cats = set(current.unique())
        hist_cats = set(history.unique())
        return len(current_cats - hist_cats)
        
    def _merge_profiles_with_original(self,
                                    original_df: pd.DataFrame,
                                    profiles: Dict[str, pd.DataFrame],
                                    process_date_col: str) -> pd.DataFrame:
        """Merge profile features with original DataFrame."""
        # Copy original
        result_df = original_df.copy()
        
        # Add profiles for each date
        for date, profile in profiles.items():
            # Get indices for this date
            date_mask = result_df[process_date_col] == date
            
            # Add each profile column
            for col in profile.columns:
                result_df.loc[date_mask, col] = profile[col].iloc[0]
        
        return result_df
