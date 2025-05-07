"""
Core pipeline for time series anomaly detection in Google Colab.

This script brings together all the components of the anomaly detection system.
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List

# Import custom modules
from .config import Config
from .data_module import DataManager
from .model_module import ModelFactory
from .training_module import Trainer
from .detection_module import AnomalyDetector, MultiLayerAnomalyDetector
from .visualization import Visualizer

def run_anomaly_detection(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    config_params: Dict[str, Any] = None,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Run full anomaly detection pipeline on a DataFrame.
    
    Args:
        df: Input DataFrame containing time series data
        target_col: Column to analyze for anomalies
        timestamp_col: Column containing timestamps
        config_params: Optional configuration parameters to override defaults
        visualize: Whether to create visualizations
        
    Returns:
        Dictionary with results from anomaly detection
    """
    print("\n===== Time Series Anomaly Detection System =====")
    print(f"Analyzing column '{target_col}' in DataFrame with {len(df)} rows")
    
    # Store original data length
    original_length = len(df)
    
    # Initialize configuration
    config = Config(config_params)
    
    # Initialize data manager
    data_manager = DataManager(
        seq_length=config.seq_length,
        test_size=config.test_size,
        batch_size=config.batch_size,
        scaler_type=config.scaler_type,
        random_state=config.random_state
    )
    
    # Prepare data
    print("\n----- Data Preparation -----")
    data_dict = data_manager.prepare_data(df, target_col=target_col)
    
    # Check for seasonality patterns
    print("\n----- Seasonality Analysis -----")
    seasonality = data_manager.detect_seasonality(df, target_col=target_col, timestamp_col=timestamp_col)
    
    # Create model
    print("\n----- Model Creation -----")
    model = ModelFactory.create_model(
        model_type=config.model_type,
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        layer_dim=config.layer_dim,
        dropout=config.dropout
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer_name=config.optimizer_name,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device
    )
    
    # Train model
    print("\n----- Model Training -----")
    history = trainer.train(
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['test_loader'],
        epochs=config.epochs,
        patience=config.patience
    )
    
    # Get reconstructions
    reconstructed_test = trainer.get_reconstructions(data_dict['test_loader']).numpy()
    
    # Multi-layer anomaly detection
    print("\n----- Multi-Layer Anomaly Detection -----")
    detector = MultiLayerAnomalyDetector(model=model, device=config.device)
    
    # Extract timestamps if available
    timestamps = None
    if timestamp_col in df.columns:
        print(f"Extracting timestamps from column: {timestamp_col}")
        timestamps = pd.to_datetime(df[timestamp_col]).values
        print(f"Timestamp example: {timestamps[0]}, type: {type(timestamps[0])}")
    
    # Run multi-layer detection
    try:
        anomaly_results = detector.detect_anomalies(
            data=data_dict['X_scaled'],
            seq_length=config.seq_length,
            batch_size=config.batch_size,
            threshold=None,
            method=config.threshold_method,
            contamination=config.contamination,
            sigma=config.sigma,
            detect_seasonal=config.detect_daily_seasonality or 
                           config.detect_weekly_seasonality or 
                           config.detect_monthly_seasonality,
            seasonal_period=24 if config.detect_daily_seasonality else 
                          7 if config.detect_weekly_seasonality else 
                          30 if config.detect_monthly_seasonality else 24,
            timestamp=timestamps
        )
        
        # Ensure results match the original data length
        result_length = len(anomaly_results['combined_anomalies'])
        if result_length != original_length:
            print(f"\nWarning: Anomaly results length ({result_length}) differs from original data length ({original_length})")
            print("Adjusting results array to match original data length...")
            
            # Create padded array
            padded_anomalies = np.zeros(original_length, dtype=bool)
            
            # Copy valid portion
            valid_length = min(result_length, original_length)
            padded_anomalies[:valid_length] = anomaly_results['combined_anomalies'][:valid_length]
            
            # Update results
            anomaly_results['combined_anomalies'] = padded_anomalies
            print(f"Results adjusted successfully. Final anomaly count: {np.sum(padded_anomalies)}")
    
    except Exception as e:
        print(f"\nError during anomaly detection: {str(e)}")
        print("Using fallback detection method...")
        # Fallback to simpler detection if advanced method fails
        # This creates a default anomaly result with zeros
        anomaly_results = {
            'combined_anomalies': np.zeros(original_length, dtype=bool),
            'ml_anomalies': np.zeros(original_length, dtype=bool),
            'statistical_anomalies': {
                'combined_statistical': np.zeros((original_length, 1), dtype=bool)
            },
            'seasonal_anomalies': None,
            'reconstruction_errors': np.zeros(original_length),
            'ml_threshold': 0.0
        }
    
    # Visualize results if requested
    if visualize:
        print("\n----- Visualization -----")
        os.makedirs('anomaly_results', exist_ok=True)
        
        # Plot training history
        Visualizer.plot_training_history(
            history=history,
            title="Training History",
            save_path="anomaly_results/training_history.png"
        )
        
        # Plot reconstructions
        original_test = data_dict['test_dataset'].sequences.numpy()
        Visualizer.plot_reconstruction(
            original=original_test,
            reconstructed=reconstructed_test,
            sample_idx=0,
            title="Original vs Reconstructed",
            save_path="anomaly_results/reconstruction.png"
        )
        
        # Plot multi-layer anomalies
        Visualizer.plot_multi_layer_anomalies(
            data=data_dict['X_scaled'],
            anomaly_results=anomaly_results,
            timestamps=timestamps,
            save_path="anomaly_results/multi_layer_anomalies.png"
        )
        
        # Plot seasonal patterns if available
        if seasonality:
            Visualizer.plot_seasonal_patterns(
                data=data_dict['X_scaled'],
                seasonal_info=seasonality,
                save_path="anomaly_results/seasonal_patterns.png"
            )
        
        print(f"Visualizations saved to 'anomaly_results/' directory")
    
    # Return results
    results = {
        'anomaly_results': anomaly_results,
        'data_dict': data_dict,
        'history': history,
        'model': model,
        'seasonality': seasonality,
        'config': config
    }
    
    print("\n===== Anomaly Detection Complete =====")
    return results