"""
Configuration module for the PyTorch Anomaly Detection system.
"""
import os
import logging
import torch
from typing import Dict, Any, Optional

class Config:
    """Configuration class to store all parameters for the anomaly detection system."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration with default values or provided dictionary."""
        # Data processing parameters
        self.seq_length = 60  # Length of sequences for time series
        self.test_size = 0.2  # Proportion of data for testing
        self.batch_size = 64  # Batch size for training
        self.scaler_type = "minmax"  # Type of scaler ('minmax' or 'standard')
        
        # Model parameters
        self.model_type = "lstm_ae"  # Model type ('lstm_ae' or 'cnn_lstm_ae')
        self.input_dim = 1  # Number of input features
        self.hidden_dim = 64  # Size of hidden layers
        self.layer_dim = 2  # Number of LSTM layers
        self.dropout = 0.2  # Dropout probability
        
        # Training parameters
        self.learning_rate = 0.001  # Learning rate for optimizer
        self.weight_decay = 1e-5  # L2 regularization
        self.epochs = 100  # Maximum number of training epochs
        self.patience = 10  # Patience for early stopping
        self.optimizer_name = "adam"  # Optimizer ('adam' or 'sgd')
        
        # Anomaly detection parameters
        self.threshold_method = "gaussian"  # Threshold method ('gaussian' or 'percentile')
        self.contamination = 0.01  # Expected proportion of anomalies
        self.sigma = 3.0  # Number of standard deviations for threshold
        
        # Seasonal detection parameters
        self.detect_daily_seasonality = True  # Whether to detect daily patterns
        self.detect_weekly_seasonality = True  # Whether to detect weekly patterns
        self.detect_monthly_seasonality = True  # Whether to detect monthly patterns
        
        # System parameters
        self.random_state = 42  # Random seed for reproducibility
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Device for computation
        self.log_level = logging.INFO  # Logging level
        
        # Visualization parameters
        self.visualize = True  # Whether to visualize results
        
        # Override defaults with provided values
        if config_dict is not None:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Warning: Unknown configuration parameter '{key}'")
        
        print(f"Configuration initialized (device: {self.device}, model: {self.model_type})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items()}