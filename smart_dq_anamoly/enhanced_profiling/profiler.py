import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class TimeSeriesProfiler:
    """Creates statistical profiles for time series data by process date."""
    
    def __init__(self, window_sizes: List[int] = [7, 14, 30, 90]):
        self.window_sizes = window_sizes
        print(f"[INIT] TimeSeriesProfiler initialized with window sizes: {window_sizes}")
        
    def create_profiles(self, 
                        df: pd.DataFrame, 
                        process_date_col: str,
                        ignore_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if process_date_col not in df.columns:
            raise ValueError(f"[ERROR] Column '{process_date_col}' not found in input DataFrame.")

        df = df.copy()
        df[process_date_col] = pd.to_datetime(df[process_date_col])
        
        if ignore_cols is None:
            ignore_cols = []
        ignore_cols.append(process_date_col)

        numeric_cols = df.select_dtypes(include=['number']).columns.difference(ignore_cols).tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(ignore_cols).tolist()

        print(f"[PROFILE] Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns for profiling.")

        df.sort_values(by=process_date_col, inplace=True)
        unique_dates = df[process_date_col].unique()
        print(f"[DATES] Found {len(unique_dates)} unique process dates.")

        all_profiles = {}
        for date in unique_dates:
            print(f"\n[PROCESSING] Profile for date: {date}")
            date_df = df[df[process_date_col] == date]
            hist_df = df[df[process_date_col] < date]

            if len(hist_df) == 0:
                print(f"[SKIP] Not enough historical data for {date}.")
                continue

            profile = self._create_single_date_profile(date_df, hist_df, numeric_cols, categorical_cols)
            all_profiles[date] = profile

        enhanced_df = self._merge_profiles_with_original(df, all_profiles, process_date_col)
        print(f"[COMPLETE] Generated profiles for {len(all_profiles)} dates. Final columns: {enhanced_df.shape[1]}")
        return enhanced_df, all_profiles

    def _create_single_date_profile(self, date_df, hist_df, numeric_cols, categorical_cols):
        profiles = []
        for col in numeric_cols:
            print(f"[NUMERIC] Profiling column: {col}")
            profiles.append(self._profile_numeric_column(date_df[col], hist_df[col], col))
        for col in categorical_cols:
            print(f"[CATEGORICAL] Profiling column: {col}")
            profiles.append(self._profile_categorical_column(date_df[col], hist_df[col], col))
        return pd.concat(profiles, axis=1) if profiles else pd.DataFrame()

    def _profile_numeric_column(self, date_series, hist_series, col_name):
        stats = {
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

        for window in self.window_sizes:
            if len(hist_series) >= window:
                last_n = hist_series.iloc[-window:]
                stats.update({
                    f"{col_name}_mean_diff_{window}d": stats[f"{col_name}_mean"] - last_n.mean(),
                    f"{col_name}_median_diff_{window}d": stats[f"{col_name}_median"] - last_n.median(),
                    f"{col_name}_std_ratio_{window}d": stats[f"{col_name}_std"] / max(last_n.std(), 1e-8),
                    f"{col_name}_range_ratio_{window}d": stats[f"{col_name}_range"] / max(last_n.max() - last_n.min(), 1e-8),
                    f"{col_name}_zscore_{window}d": (stats[f"{col_name}_mean"] - last_n.mean()) / max(last_n.std(), 1e-8)
                })

        return pd.DataFrame([stats])

    def _profile_categorical_column(self, date_series, hist_series, col_name):
        current_counts = date_series.value_counts(normalize=True, dropna=False)
        stats = {
            f"{col_name}_unique_count": date_series.nunique(),
            f"{col_name}_mode": date_series.mode().iloc[0] if not date_series.mode().empty else None,
            f"{col_name}_entropy": self._calculate_entropy(current_counts),
            f"{col_name}_nulls_pct": date_series.isna().mean() * 100
        }

        for window in self.window_sizes:
            if len(hist_series) >= window:
                last_n = hist_series.iloc[-window:]
                hist_counts = last_n.value_counts(normalize=True, dropna=False)
                stats.update({
                    f"{col_name}_unique_diff_{window}d": stats[f"{col_name}_unique_count"] - last_n.nunique(),
                    f"{col_name}_js_distance_{window}d": self._jensen_shannon_distance(current_counts, hist_counts),
                    f"{col_name}_new_categories_{window}d": self._count_new_categories(date_series, last_n)
                })

        return pd.DataFrame([stats])

    def _merge_profiles_with_original(self, original_df, profiles, process_date_col):
        print(f"[MERGE] Merging profiles into original DataFrame...")
        result_df = original_df.copy()
        profile_dfs = []

        for date, profile in profiles.items():
            if profile.empty:
                continue
            temp = profile.copy()
            temp[process_date_col] = date
            profile_dfs.append(temp)

        if profile_dfs:
            all_profiles_df = pd.concat(profile_dfs, ignore_index=True)
            result_df = pd.merge(result_df, all_profiles_df, on=process_date_col, how='left')
            print(f"[MERGE] Done. Resulting DataFrame has {result_df.shape[1]} columns.")
        else:
            print("[WARNING] No profile data to merge.")

        return result_df

    def _calculate_entropy(self, counts: pd.Series) -> float:
        probs = counts.values
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _jensen_shannon_distance(self, p: pd.Series, q: pd.Series) -> float:
        idx = p.index.union(q.index)
        p_full = p.reindex(idx, fill_value=0)
        q_full = q.reindex(idx, fill_value=0)
        m = 0.5 * (p_full + q_full)
        return 0.5 * (
            np.sum(p_full * np.log2(p_full / (m + 1e-10) + 1e-10)) +
            np.sum(q_full * np.log2(q_full / (m + 1e-10) + 1e-10))
        )

    def _count_new_categories(self, current: pd.Series, history: pd.Series) -> int:
        return len(set(current.unique()) - set(history.unique()))
