"""
Manager for orchestrating the data profiling process.
"""
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from profilers.base_profiler import BaseProfiler


class ProfileManager:
    """Manager for orchestrating data profiling."""
    
    def __init__(self, profiler: BaseProfiler, config: Optional[Dict[str, Any]] = None):
        """Initialize the profile manager."""
        self.profiler = profiler
        self.config = config or {}
        self.max_workers = self.config.get("max_workers", 4)
        self.column_limit = self.config.get("column_limit", None)
        print(f"Initialized ProfileManager with {self.max_workers} workers")
        
    def profile_table(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Profile all columns in a table."""
        print(f"\n{'='*80}")
        print(f"PROFILING TABLE: {len(df)} rows × {len(df.columns)} columns")
        print(f"{'='*80}\n")
        
        columns = df.columns[:self.column_limit] if self.column_limit else df.columns
        print(f"Profiling {len(columns)} columns with {self.max_workers} parallel workers")
        
        column_metadata = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_column = {
                executor.submit(self.profiler.profile_column, df, col): col 
                for col in columns
            }
            
            total_columns = len(future_to_column)
            completed = 0
            
            for future in as_completed(future_to_column):
                column = future_to_column[future]
                try:
                    metadata = future.result()
                    column_metadata[column] = metadata
                    completed += 1
                    if completed % max(1, total_columns // 10) == 0 or completed == total_columns:
                        progress = (completed / total_columns) * 100
                        print(f"Progress: {progress:.1f}% - Completed {completed}/{total_columns} columns")
                except Exception as e:
                    print(f"ERROR profiling column '{column}': {str(e)}")
        
        pii_columns = sum(1 for meta in column_metadata.values() if meta.get('is_pii', False))
        print(f"\nProfiling complete! {len(column_metadata)} columns profiled, {pii_columns} PII columns detected")
        return column_metadata
    
    def get_profile_dataframe(self, profile: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert profile dictionary to a DataFrame.
        
        Args:
            profile: Profile data dictionary
            
        Returns:
            DataFrame with profile information
        """
        print("Converting profile to DataFrame format")
        
        # Flatten the nested dictionary structure
        rows = []
        for column_name, metadata in profile.items():
            # Create a copy of metadata with column_name as a field
            row = {"column_name": column_name}
            
            # Add all other metadata
            for key, value in metadata.items():
                if key != "column_name":  # Avoid duplication
                    # Handle non-serializable objects
                    if isinstance(value, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
                        row[key] = value.isoformat()
                    elif isinstance(value, list):
                        row[key] = str(value)
                    else:
                        row[key] = value
            
            rows.append(row)
        
        df_profile = pd.DataFrame(rows)
        print(f"Created DataFrame with {len(df_profile)} rows and {len(df_profile.columns)} columns")
        return df_profile
    
    def save_profile(self, profile: Dict[str, Dict[str, Any]], output_path: str) -> None:
        """Save the profile to a file."""
        print(f"Saving profile to {output_path}")
        
        serializable_profile = {}
        for col, metadata in profile.items():
            serializable_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
                    serializable_metadata[key] = value.isoformat()
                else:
                    serializable_metadata[key] = value
            serializable_profile[col] = serializable_metadata
        
        with open(output_path, 'w') as f:
            json.dump(serializable_profile, f, indent=2)
        
        print(f"Profile saved successfully")