"""
Base profiler interface for data profiling components.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any, Optional


class BaseProfiler(ABC):
    """Abstract base class for all data profilers."""
    
    @abstractmethod
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the profiler with configuration.
        
        Args:
            config: Dictionary containing profiler configuration
        """
        pass
    
    @abstractmethod
    def profile_column(self, df: pd.DataFrame, column_name: str) -> Dict[str, Any]:
        """
        Profile a column to extract metadata and statistics.
        
        Args:
            df: DataFrame containing the column
            column_name: Name of the column to profile
            
        Returns:
            Dictionary containing column metadata and statistics
        """
        pass