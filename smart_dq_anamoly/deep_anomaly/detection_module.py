"""
Anomaly detection components for time series analysis.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime

class AnomalyDetector:
    """Detects anomalies using reconstruction error from autoencoder."""
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        """
        Initialize the anomaly detector.
        
        Args:
            model: Trained autoencoder model
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set model
        self.model = model.to(self.device)
        self.model.eval()
        
        print(f"Initialized AnomalyDetector with device={self.device}")
    
    def calculate_reconstruction_error(self, data_loader: DataLoader) -> np.ndarray:
        """
        Calculate reconstruction error for each sample.
        
        Args:
            data_loader: DataLoader containing data
            
        Returns:
            Array of reconstruction errors
        """
        all_errors = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate mean squared error for each sample
                errors = torch.mean((outputs - batch)**2, dim=(1, 2)).cpu().numpy()
                all_errors.append(errors)
        
        return np.concatenate(all_errors)
    
    def find_threshold(self, 
                      reconstruction_errors: np.ndarray, 
                      method: str = "gaussian",
                      contamination: float = 0.01, 
                      sigma: float = 3.0) -> float:
        """
        Find a threshold for anomaly detection.
        
        Args:
            reconstruction_errors: Array of reconstruction errors
            method: Method to use ('gaussian', 'percentile')
            contamination: Expected proportion of anomalies (for percentile method)
            sigma: Number of standard deviations (for gaussian method)
            
        Returns:
            Threshold value
        """
        if method.lower() == "gaussian":
            # Gaussian method: mean + sigma * std
            threshold = reconstruction_errors.mean() + sigma * reconstruction_errors.std()
            print(f"Calculated threshold using gaussian method: {threshold:.6f} "
                 f"(mean={reconstruction_errors.mean():.6f}, std={reconstruction_errors.std():.6f})")
        
        elif method.lower() == "percentile":
            # Percentile method: (1-contamination) percentile
            threshold = np.percentile(reconstruction_errors, 100 * (1 - contamination))
            print(f"Calculated threshold using percentile method: {threshold:.6f} "
                 f"(percentile={100 * (1 - contamination)})")
        
        else:
            raise ValueError(f"Unsupported threshold method: {method}")
        
        return threshold
    
    def detect_anomalies(self, 
                        data_loader: DataLoader, 
                        threshold: Optional[float] = None,
                        method: str = "gaussian",
                        contamination: float = 0.01,
                        sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Detect anomalies in the data.
        
        Args:
            data_loader: DataLoader containing data
            threshold: Anomaly threshold (if None, calculated automatically)
            method: Method to use for threshold calculation ('gaussian', 'percentile')
            contamination: Expected proportion of anomalies (for percentile method)
            sigma: Number of standard deviations (for gaussian method)
            
        Returns:
            Tuple containing:
            - Boolean array indicating anomalies
            - Array of reconstruction errors
            - Threshold value
        """
        # Calculate reconstruction errors
        reconstruction_errors = self.calculate_reconstruction_error(data_loader)
        
        # Calculate threshold if not provided
        if threshold is None:
            threshold = self.find_threshold(
                reconstruction_errors, method, contamination, sigma
            )
        
        # Detect anomalies
        anomalies = reconstruction_errors > threshold
        
        num_anomalies = anomalies.sum()
        print(f"Detected {num_anomalies} anomalies out of {len(anomalies)} samples "
             f"({100*num_anomalies/len(anomalies):.2f}%)")
        
        return anomalies, reconstruction_errors, threshold
    
    def predict_anomalies(self, 
                         data: np.ndarray, 
                         seq_length: int, 
                         batch_size: int = 64,
                         threshold: Optional[float] = None,
                         method: str = "gaussian",
                         contamination: float = 0.01,
                         sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict anomalies for new data.
        
        Args:
            data: Input data of shape [n_samples, n_features]
            seq_length: Length of sequences to create
            batch_size: Batch size for data loader
            threshold: Anomaly threshold (if None, calculated automatically)
            method: Method to use for threshold calculation ('gaussian', 'percentile')
            contamination: Expected proportion of anomalies (for percentile method)
            sigma: Number of standard deviations (for gaussian method)
            
        Returns:
            Tuple containing:
            - Boolean array indicating anomalies
            - Array of reconstruction errors
            - Threshold value
        """
        # Import TimeSeriesDataset with correct package path
        from deep_anomaly.data_module import TimeSeriesDataset
        
        # Create dataset and loader
        dataset = TimeSeriesDataset(data, seq_length)
        print(f"Created dataset with {len(dataset)} sequences of length {seq_length}")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Detect anomalies
        return self.detect_anomalies(
            loader, threshold, method, contamination, sigma
        )


class MultiLayerAnomalyDetector:
    """
    Multi-layer anomaly detection system that combines statistical and ML-based approaches.
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        """
        Initialize the multi-layer anomaly detector.
        
        Args:
            model: Trained autoencoder model
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Set up ML-based detector
        self.ml_detector = AnomalyDetector(model, device)
        
        print("Initialized MultiLayerAnomalyDetector")
    
    def detect_statistical_anomalies(self, data: np.ndarray, window_size: int = 20) -> Dict[str, np.ndarray]:
        """
        Detect anomalies using statistical methods.
        
        Args:
            data: Input data of shape [n_samples, n_features]
            window_size: Size of rolling window for statistics
            
        Returns:
            Dictionary with different types of anomalies
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        
        # Initialize anomaly flags
        value_anomalies = np.zeros((n_samples, n_features), dtype=bool)
        trend_anomalies = np.zeros((n_samples, n_features), dtype=bool)
        volatility_anomalies = np.zeros((n_samples, n_features), dtype=bool)
        
        for feat in range(n_features):
            # Get feature data
            feat_data = data[:, feat]
            
            # 1. Global value anomalies (using 3-sigma rule)
            mean = np.mean(feat_data)
            std = np.std(feat_data)
            value_anomalies[:, feat] = np.abs(feat_data - mean) > 3 * std
            
            # 2. Detect trend anomalies using rolling window
            if n_samples > window_size:
                # Calculate rolling mean
                rolling_mean = np.zeros(n_samples)
                rolling_std = np.zeros(n_samples)
                
                for i in range(window_size, n_samples):
                    window_data = feat_data[i-window_size:i]
                    rolling_mean[i] = np.mean(window_data)
                    rolling_std[i] = np.std(window_data)
                
                # Detect trend breaks (large deviations from rolling mean)
                trend_anomalies[window_size:, feat] = np.abs(feat_data[window_size:] - rolling_mean[window_size:]) > 3 * rolling_std[window_size:]
                
                # 3. Detect volatility anomalies (sudden changes in variance)
                if n_samples > 2 * window_size:
                    volatility = np.zeros(n_samples)
                    for i in range(window_size, n_samples):
                        volatility[i] = np.std(feat_data[i-window_size:i])
                    
                    # Compare volatility to its rolling mean/std
                    vol_mean = np.zeros(n_samples)
                    vol_std = np.zeros(n_samples)
                    
                    for i in range(2 * window_size, n_samples):
                        vol_mean[i] = np.mean(volatility[i-window_size:i])
                        vol_std[i] = np.std(volatility[i-window_size:i])
                    
                    # Flag volatility anomalies
                    volatility_anomalies[2 * window_size:, feat] = np.abs(volatility[2 * window_size:] - vol_mean[2 * window_size:]) > 3 * vol_std[2 * window_size:]
        
        # Combine all statistical anomalies
        combined_statistical = value_anomalies | trend_anomalies | volatility_anomalies
        
        # Count anomalies
        print(f"Statistical anomalies detected:")
        print(f"  Value anomalies: {np.sum(value_anomalies)}")
        print(f"  Trend anomalies: {np.sum(trend_anomalies)}")
        print(f"  Volatility anomalies: {np.sum(volatility_anomalies)}")
        print(f"  Combined: {np.sum(combined_statistical)}")
        
        return {
            'value_anomalies': value_anomalies,
            'trend_anomalies': trend_anomalies,
            'volatility_anomalies': volatility_anomalies,
            'combined_statistical': combined_statistical
        }
    
    def detect_seasonal_anomalies(self, 
                                 data: np.ndarray, 
                                 timestamp: np.ndarray = None,
                                 period: int = 24) -> np.ndarray:
        """
        Detect anomalies in seasonal patterns.
        
        Args:
            data: Input data of shape [n_samples, n_features]
            timestamp: Timestamps for data points (can be datetime)
            period: Expected seasonality period
            
        Returns:
            Boolean array indicating seasonal anomalies
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        
        # Need at least 2 full periods to detect seasonal anomalies
        if n_samples < 2 * period:
            print(f"Warning: Not enough data points ({n_samples}) for seasonal analysis with period {period}")
            return np.zeros(n_samples, dtype=bool)
        
        seasonal_anomalies = np.zeros(n_samples, dtype=bool)
        
        # Create seasonal indices
        if timestamp is not None:
            # If timestamps provided, use them
            seasonal_idx = np.zeros(n_samples, dtype=int)
            
            # Check if timestamp contains datetime objects
            is_datetime = False
            if len(timestamp) > 0:
                first_ts = timestamp[0]
                is_datetime = isinstance(first_ts, (pd.Timestamp, np.datetime64, datetime)) or np.issubdtype(type(first_ts), np.datetime64)
                print(f"Timestamp type detected: {type(first_ts).__name__}, is_datetime={is_datetime}")
            
            # Process timestamps
            for i, ts in enumerate(timestamp):
                if is_datetime:
                    # Handle datetime timestamp based on period
                    if isinstance(ts, np.datetime64):
                        ts = pd.Timestamp(ts)
                    elif not isinstance(ts, (pd.Timestamp, datetime)):
                        try:
                            ts = pd.Timestamp(ts)
                        except:
                            print(f"Warning: Could not convert timestamp {ts} to datetime. Using index instead.")
                            seasonal_idx[i] = i % period
                            continue
                    
                    # Extract appropriate seasonality component
                    if period == 24:  # Daily (hours)
                        seasonal_idx[i] = ts.hour
                    elif period == 7:  # Weekly (day of week)
                        seasonal_idx[i] = ts.dayofweek
                    elif period == 12:  # Monthly (month of year)
                        seasonal_idx[i] = ts.month - 1
                    elif period == 30 or period == 31:  # Monthly (day of month)
                        seasonal_idx[i] = min(ts.day - 1, period - 1)
                    else:
                        # Default to day of year modulo period
                        seasonal_idx[i] = (ts.dayofyear - 1) % period
                else:
                    # For non-datetime, use modulo
                    try:
                        seasonal_idx[i] = int(ts) % period
                    except:
                        print(f"Warning: Failed to extract seasonal index from {ts}. Using index instead.")
                        seasonal_idx[i] = i % period
        else:
            # Otherwise just use modulo on indices
            seasonal_idx = np.arange(n_samples) % period
        
        print(f"Seasonal indices distribution: min={seasonal_idx.min()}, max={seasonal_idx.max()}, unique values={len(np.unique(seasonal_idx))}")
        
        # Calculate mean and std for each position in the seasonal pattern
        for pos in range(period):
            pos_indices = seasonal_idx == pos
            num_pos_samples = np.sum(pos_indices)
            print(f"Position {pos}: {num_pos_samples} samples")
            
            if num_pos_samples > 5:  # Need enough samples
                for feat in range(n_features):
                    pos_data = data[pos_indices, feat]
                    pos_mean = np.mean(pos_data)
                    pos_std = max(np.std(pos_data), 1e-8)  # Avoid division by zero
                    
                    # Mark points that deviate significantly from seasonal pattern
                    deviations = np.abs(data[pos_indices, feat] - pos_mean) / pos_std
                    anomaly_indices = pos_indices.copy()
                    anomaly_indices[anomaly_indices] = deviations > 3
                    seasonal_anomalies = seasonal_anomalies | anomaly_indices
        
        print(f"Detected {np.sum(seasonal_anomalies)} seasonal anomalies out of {n_samples} samples")
        return seasonal_anomalies
    
    def detect_anomalies(self, 
                        data: np.ndarray,
                        seq_length: int,
                        batch_size: int = 64,
                        threshold: Optional[float] = None,
                        method: str = "gaussian",
                        contamination: float = 0.01,
                        sigma: float = 3.0,
                        detect_seasonal: bool = True,
                        seasonal_period: int = 24,
                        timestamp: np.ndarray = None) -> Dict[str, Any]:
        """
        Detect anomalies using multiple layers.
        
        Args:
            data: Input data
            seq_length: Length of sequences for ML detector
            batch_size: Batch size for data loader
            threshold: Anomaly threshold for ML detector
            method: Method for threshold calculation
            contamination: Expected proportion of anomalies
            sigma: Number of standard deviations
            detect_seasonal: Whether to detect seasonal anomalies
            seasonal_period: Period for seasonal anomalies
            timestamp: Timestamps for seasonal detection
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Store original data length for proper padding later
        original_length = len(data)
        print(f"\n=== Multi-Layer Anomaly Detection on {original_length} data points ===")
        
        # 1. Detect statistical anomalies
        print("\nLayer 1: Statistical anomaly detection")
        statistical_results = self.detect_statistical_anomalies(data)
        
        # 2. Detect seasonal anomalies if requested
        seasonal_anomalies = None
        if detect_seasonal and timestamp is not None:
            print("\nLayer 2: Seasonal anomaly detection")
            try:
                seasonal_anomalies = self.detect_seasonal_anomalies(data, timestamp, seasonal_period)
            except Exception as e:
                print(f"Error in seasonal anomaly detection: {str(e)}")
                print("Continuing without seasonal anomaly detection...")
        
        # 3. Detect ML-based anomalies
        print("\nLayer 3: Deep learning anomaly detection")
        ml_anomalies, reconstruction_errors, ml_threshold = self.ml_detector.predict_anomalies(
            data, seq_length, batch_size, threshold, method, contamination, sigma
        )
        
        # 4. Combine all anomalies with proper handling of different lengths
        print("\nCombining anomaly detection results")
        # Initialize combined anomalies array with the original data length
        combined_anomalies = np.zeros(original_length, dtype=bool)
        
        # Pad ML anomalies to match original length if needed
        ml_anomalies_padded = np.zeros(original_length, dtype=bool)
        ml_length = len(ml_anomalies)
        
        # Check if ML anomalies length is different from original
        if ml_length != original_length:
            print(f"Note: ML anomalies length ({ml_length}) differs from original data length ({original_length})")
            # Place ML anomalies at appropriate position (typically starts at seq_length-1)
            offset = seq_length - 1
            # Ensure we don't exceed array bounds
            valid_length = min(ml_length, original_length - offset)
            if valid_length > 0:
                ml_anomalies_padded[offset:offset+valid_length] = ml_anomalies[:valid_length]
            ml_anomalies = ml_anomalies_padded
        else:
            # If lengths match, no padding needed
            ml_anomalies_padded = ml_anomalies
        
        # Process statistical anomalies
        if len(statistical_results['combined_statistical'].shape) > 1:
            # If multi-dimensional, flatten by taking any anomaly across features
            stat_anomalies = np.any(statistical_results['combined_statistical'], axis=1)
        else:
            stat_anomalies = statistical_results['combined_statistical']
        
        # Ensure stat anomalies match original length
        if len(stat_anomalies) != original_length:
            print(f"Warning: Statistical anomalies length ({len(stat_anomalies)}) differs from original data length ({original_length})")
            temp = np.zeros(original_length, dtype=bool)
            valid_length = min(len(stat_anomalies), original_length)
            temp[:valid_length] = stat_anomalies[:valid_length]
            stat_anomalies = temp
        
        # Combine statistical and ML anomalies safely
        combined_anomalies = stat_anomalies | ml_anomalies_padded
        
        # Add seasonal anomalies if available
        if seasonal_anomalies is not None:
            # Ensure seasonal anomalies match original length
            if len(seasonal_anomalies) != original_length:
                print(f"Warning: Seasonal anomalies length ({len(seasonal_anomalies)}) differs from original data length ({original_length})")
                temp = np.zeros(original_length, dtype=bool)
                valid_length = min(len(seasonal_anomalies), original_length)
                temp[:valid_length] = seasonal_anomalies[:valid_length]
                seasonal_anomalies = temp
            combined_anomalies = combined_anomalies | seasonal_anomalies
        
        # Count final anomalies
        total_anomalies = np.sum(combined_anomalies)
        print(f"\nFinal results: {total_anomalies} anomalies detected out of {original_length} samples "
              f"({100*total_anomalies/original_length:.2f}%)")
        
        return {
            'combined_anomalies': combined_anomalies,
            'statistical_anomalies': statistical_results,
            'ml_anomalies': ml_anomalies_padded,
            'seasonal_anomalies': seasonal_anomalies,
            'reconstruction_errors': reconstruction_errors,
            'ml_threshold': ml_threshold
        }