"""
Column profiler for extracting metadata and statistics from data columns.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re
from datetime import datetime

from profilers.base_profiler import BaseProfiler


class ColumnProfiler(BaseProfiler):
    """Profile data columns to extract metadata and statistics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the column profiler.
        
        Args:
            config: Configuration options for profiling
        """
        self.config = config or {}
        print("Initialized column profiler")
    
    def _infer_data_type(self, series: pd.Series) -> str:
        """
        Infer the data type of a column.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            String representing the data type
        """
        # Drop nulls for type inference
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return "unknown"
        
        # Check if boolean
        if pd.api.types.is_bool_dtype(non_null):
            return "boolean"
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(non_null):
            # Check if integers only
            if pd.api.types.is_integer_dtype(non_null) or non_null.apply(lambda x: float(x).is_integer()).all():
                return "integer"
            return "number"
        
        # Check if datetime
        if pd.api.types.is_datetime64_dtype(non_null):
            return "date"
        
        # Try to convert to datetime if string
        if pd.api.types.is_string_dtype(non_null) or pd.api.types.is_object_dtype(non_null):
            # Sample values for date pattern checking
            sample = non_null.sample(min(len(non_null), 100))
            
            # Try common date formats
            try:
                pd.to_datetime(sample, errors='raise')
                return "date"
            except:
                pass
            
            # Check for common patterns in first 100 values
            patterns = {
                "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                "phone": r'^\+?[\d\s\(\)\-]{7,20}$',
                "url": r'^(http|https)://',
                "zip_code": r'^\d{5}(-\d{4})?$',
                "ip_address": r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
            }
            
            for name, pattern in patterns.items():
                # If >80% match the pattern, classify as that type
                if sample.str.match(pattern).mean() > 0.8:
                    return name
            
            return "string"
        
        return "object"
    
    def _detect_pii(self, column_name: str, data_type: str, sample_values: List[Any]) -> bool:
        """
        Detect if a column likely contains PII.
        
        Args:
            column_name: Name of the column
            data_type: Data type of the column
            sample_values: Sample values from the column
            
        Returns:
            True if column likely contains PII, False otherwise
        """
        # PII indicators in column name
        pii_name_indicators = [
            'ssn', 'social', 'tax', 'id', 'ident', 'passport', 'license', 
            'account', 'user', 'email', 'phone', 'mobile', 'address', 'zip', 
            'postal', 'name', 'first', 'last', 'birth', 'dob', 'credit', 'card',
            'password', 'secret', 'token', 'ip', 'location', 'geo'
        ]
        
        # Check column name for PII indicators
        lower_name = column_name.lower()
        for indicator in pii_name_indicators:
            if indicator in lower_name:
                print(f"PII detected in column {column_name} based on name indicator: {indicator}")
                return True
        
        # Type-based PII detection
        if data_type in ['email', 'phone', 'ip_address']:
            print(f"PII detected in column {column_name} based on data type: {data_type}")
            return True
        
        # Pattern-based detection for sample values
        if data_type == 'string' and sample_values:
            # Check for SSN pattern
            ssn_pattern = re.compile(r'^\d{3}-\d{2}-\d{4}$')
            # Check for credit card pattern (simplified)
            cc_pattern = re.compile(r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$')
            
            for value in sample_values:
                if isinstance(value, str):
                    if ssn_pattern.match(value) or cc_pattern.match(value):
                        print(f"PII detected in column {column_name} based on value pattern")
                        return True
        
        return False
    
    def profile_column(self, df: pd.DataFrame, column_name: str) -> Dict[str, Any]:
        """
        Profile a column to extract metadata and statistics.
        
        Args:
            df: DataFrame containing the column
            column_name: Name of the column to profile
            
        Returns:
            Dictionary containing column metadata and statistics
        """
        print(f"Profiling column: {column_name}")
        
        if column_name not in df.columns:
            raise ValueError(f"Column not found: {column_name}")
        
        series = df[column_name]
        
        # Basic metadata
        data_type = self._infer_data_type(series)
        print(f"Inferred data type for {column_name}: {data_type}")
        
        # Calculate statistics
        profile = {
            "column_name": column_name,
            "data_type": data_type,
            "nullable": True,  # Assume nullable by default
            "null_count": series.isna().sum(),
            "null_percentage": round(series.isna().mean() * 100, 2),
            "distinct_count": series.nunique(),
            "distinct_percentage": round(series.nunique() / len(series) * 100, 2) if len(series) > 0 else 0,
            "sample_values": series.dropna().sample(min(5, series.nunique())).tolist() if not series.empty else []
        }
        
        # Numeric statistics
        if data_type in ["number", "integer"]:
            numeric_series = pd.to_numeric(series, errors='coerce')
            profile.update({
                "min": numeric_series.min(),
                "max": numeric_series.max(),
                "mean": round(numeric_series.mean(), 2),
                "median": numeric_series.median(),
                "std": round(numeric_series.std(), 2),
                "has_zeros": (numeric_series == 0).any(),
                "has_negatives": (numeric_series < 0).any(),
                "quartile_1": numeric_series.quantile(0.25),
                "quartile_3": numeric_series.quantile(0.75)
            })
        
        # String statistics
        if data_type == "string":
            string_series = series.dropna().astype(str)
            if not string_series.empty:
                profile.update({
                    "min_length": string_series.str.len().min(),
                    "max_length": string_series.str.len().max(),
                    "avg_length": round(string_series.str.len().mean(), 2),
                    "has_numbers": string_series.str.contains('\d').any(),
                    "has_letters": string_series.str.contains('[a-zA-Z]').any(),
                    "has_special_chars": string_series.str.contains('[^a-zA-Z0-9\s]').any()
                })
        
        # Date statistics
        if data_type == "date":
            try:
                date_series = pd.to_datetime(series, errors='coerce')
                profile.update({
                    "min_date": date_series.min(),
                    "max_date": date_series.max(),
                    "date_range_days": (date_series.max() - date_series.min()).days
                })
            except Exception as e:
                print(f"Error processing date column {column_name}: {str(e)}")
        
        # PII detection
        profile["is_pii"] = self._detect_pii(column_name, data_type, profile["sample_values"])
        
        print(f"Completed profiling for column: {column_name}")
        return profile