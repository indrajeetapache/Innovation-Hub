"""
Data processing components for time series anomaly detection.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data with sliding window sequence creation."""
    
    def __init__(self, data: np.ndarray, seq_length: int):
        """
        Initialize the dataset with time series data.
        
        Args:
            data: NumPy array of shape [n_samples, n_features]
            seq_length: Length of sequences to create
        """
        self.data = data
        self.seq_length = seq_length
        
        # Create sequences using sliding window
        self.sequences = self._create_sequences()
        
        print(f"Created dataset with {len(self.sequences)} sequences of length {seq_length}")
    
    def _create_sequences(self) -> torch.Tensor:
        """Create sequences using sliding window approach."""
        sequences = []
        n_samples = self.data.shape[0]
        
        for i in range(n_samples - self.seq_length + 1):
            # Extract sequence of appropriate length
            sequence = self.data[i:i + self.seq_length]
            sequences.append(sequence)
        
        return torch.tensor(np.array(sequences), dtype=torch.float32)
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sequence by index."""
        return self.sequences[idx]


class DataPreprocessor:
    """Handles data preprocessing for time series anomaly detection."""
    
    def __init__(self, scaler_type: str = "minmax"):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler to use ('minmax' or 'standard')
        """
        self.scaler_type = scaler_type.lower()
        
        # Initialize the appropriate scaler
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}. Use 'minmax' or 'standard'.")
        
        print(f"Initialized DataPreprocessor with {self.scaler_type} scaler")
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the scaler and transform the data.
        
        Args:
            data: Input data of shape [n_samples, n_features]
            
        Returns:
            Scaled data of the same shape
        """
        # Reshape if needed for single feature
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Fit and transform
        scaled_data = self.scaler.fit_transform(data)
        
        print(f"Fitted scaler and transformed data with shape {data.shape}")
        return scaled_data
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using the fitted scaler.
        
        Args:
            data: Input data of shape [n_samples, n_features]
            
        Returns:
            Scaled data of the same shape
        """
        # Reshape if needed for single feature
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Transform
        scaled_data = self.scaler.transform(data)
        
        print(f"Transformed data with shape {data.shape}")
        return scaled_data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data.
        
        Args:
            data: Scaled data
            
        Returns:
            Data in original scale
        """
        return self.scaler.inverse_transform(data)


class DataManager:
    """Manages data loading, preprocessing, and splitting for anomaly detection."""
    
    def __init__(self, 
                 seq_length: int = 60, 
                 test_size: float = 0.2,
                 batch_size: int = 64,
                 scaler_type: str = "minmax",
                 random_state: int = 42):
        """
        Initialize the data manager.
        
        Args:
            seq_length: Length of sequences to create
            test_size: Proportion of data to use for testing
            batch_size: Batch size for data loaders
            scaler_type: Type of scaler to use ('minmax' or 'standard')
            random_state: Random state for reproducibility
        """
        self.seq_length = seq_length
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(scaler_type=scaler_type)
        
        print(f"Initialized DataManager with sequence length={seq_length}, "
              f"test_size={test_size}, batch_size={batch_size}")
    
    def prepare_data(self, data: Union[pd.DataFrame, np.ndarray], 
                    target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare data for anomaly detection.
        
        Args:
            data: Input data (DataFrame or NumPy array)
            target_col: Target column name if data is a DataFrame
            
        Returns:
            Dictionary containing train and test data loaders and related info
        """
        # Convert DataFrame to NumPy if needed
        if isinstance(data, pd.DataFrame):
            if target_col is None:
                raise ValueError("target_col must be specified when data is a DataFrame")
            
            # Extract features and convert to numpy
            X = data[target_col].values
            print(f"Extracted column '{target_col}' from DataFrame")
        else:
            X = data
        
        # Preprocess data
        X_scaled = self.preprocessor.fit_transform(X)
        
        # Split into train and test sets
        X_train, X_test = train_test_split(
            X_scaled, test_size=self.test_size, shuffle=False, random_state=self.random_state
        )
        
        print(f"Split data into train ({X_train.shape}) and test ({X_test.shape}) sets")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, self.seq_length)
        test_dataset = TimeSeriesDataset(X_test, self.seq_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Created DataLoaders with {len(train_loader)} train batches and "
              f"{len(test_loader)} test batches")
        
        return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "X_train": X_train,
            "X_test": X_test,
            "X_scaled": X_scaled,
            "original_data": X
        }
    
    def detect_seasonality(self, data: Union[pd.DataFrame, np.ndarray], 
                          target_col: Optional[str] = None,
                          timestamp_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect seasonality patterns in the data.
        
        Args:
            data: Input data (DataFrame or NumPy array)
            target_col: Target column name if data is a DataFrame
            timestamp_col: Timestamp column name if data is a DataFrame
            
        Returns:
            Dictionary containing seasonality information
        """
        if not isinstance(data, pd.DataFrame):
            print("Seasonality detection requires a DataFrame with timestamp column")
            return {}
            
        if timestamp_col is None:
            print("Timestamp column must be specified for seasonality detection")
            return {}
            
        if target_col is None:
            print("Target column must be specified for seasonality detection")
            return {}
        
        # Ensure timestamp is datetime
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        
        # Set timestamp as index
        ts_data = data.set_index(timestamp_col)[[target_col]].copy()
        
        print(f"Analyzing seasonality patterns in column '{target_col}'")
        
        # Extract time components
        ts_data['hour'] = ts_data.index.hour
        ts_data['day'] = ts_data.index.day
        ts_data['day_of_week'] = ts_data.index.dayofweek
        ts_data['month'] = ts_data.index.month
        
        # Calculate daily patterns
        daily_pattern = ts_data.groupby('hour')[target_col].mean()
        daily_std = ts_data.groupby('hour')[target_col].std()
        
        # Calculate weekly patterns
        weekly_pattern = ts_data.groupby('day_of_week')[target_col].mean()
        weekly_std = ts_data.groupby('day_of_week')[target_col].std()
        
        # Calculate monthly patterns
        monthly_pattern = ts_data.groupby('month')[target_col].mean()
        monthly_std = ts_data.groupby('month')[target_col].std()
        
        # Calculate autocorrelation to detect seasonality
        from statsmodels.tsa.stattools import acf
        
        # Convert to regular time series if needed
        if ts_data.index.freq is None:
            ts_data = ts_data.resample('1D').mean()
        
        # Calculate autocorrelation
        try:
            autocorr = acf(ts_data[target_col].dropna(), nlags=30)
        except:
            print("Could not calculate autocorrelation")
            autocorr = None
        
        # Store results
        seasonality = {
            'daily_pattern': daily_pattern,
            'daily_std': daily_std,
            'weekly_pattern': weekly_pattern,
            'weekly_std': weekly_std, 
            'monthly_pattern': monthly_pattern,
            'monthly_std': monthly_std,
            'autocorrelation': autocorr
        }
        
        # Detect significant seasonality
        if autocorr is not None:
            # Check if any autocorrelation (after lag 0) is significant
            if np.any(np.abs(autocorr[1:]) > 1.96/np.sqrt(len(ts_data))):
                print("Detected significant seasonality in the data")
                
                # Check for specific patterns
                if np.abs(autocorr[7]) > 1.96/np.sqrt(len(ts_data)):
                    print("Weekly seasonality detected")
                    seasonality['has_weekly_seasonality'] = True
                else:
                    seasonality['has_weekly_seasonality'] = False
                    
                if np.abs(autocorr[30 if len(autocorr) > 30 else -1]) > 1.96/np.sqrt(len(ts_data)):
                    print("Monthly seasonality detected")
                    seasonality['has_monthly_seasonality'] = True
                else:
                    seasonality['has_monthly_seasonality'] = False
                    
                # Check for daily seasonality (if data has sub-daily frequency)
                hours_per_day = len(daily_pattern)
                if hours_per_day > 1 and hours_per_day <= 24:
                    daily_variation = daily_pattern.std() / daily_pattern.mean()
                    if daily_variation > 0.1:  # Threshold for significant variation
                        print("Daily seasonality detected")
                        seasonality['has_daily_seasonality'] = True
                    else:
                        seasonality['has_daily_seasonality'] = False
        
        print("Seasonality analysis complete")
        return seasonality