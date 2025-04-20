"""
Visualization components for anomaly detection results.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any


class Visualizer:
    """Provides visualization tools for anomaly detection results."""
    
    @staticmethod
    def plot_reconstruction(original: np.ndarray, 
                           reconstructed: np.ndarray, 
                           sample_idx: int = 0,
                           title: str = "Original vs Reconstructed",
                           save_path: Optional[str] = None):
        """
        Plot original vs reconstructed data for a single sample.
        
        Args:
            original: Original data of shape [n_samples, seq_length, n_features]
            reconstructed: Reconstructed data of the same shape
            sample_idx: Index of sample to plot
            title: Plot title
            save_path: Path to save the plot (if None, displays instead)
        """
        plt.figure(figsize=(12, 6))
        
        # Get data for the selected sample
        orig = original[sample_idx, :, 0]  # Assume first feature if multivariate
        recon = reconstructed[sample_idx, :, 0]
        
        # Plot
        plt.plot(orig, 'b-', label='Original')
        plt.plot(recon, 'r--', label='Reconstructed')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved reconstruction plot to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_reconstruction_error(reconstruction_error: np.ndarray, 
                                threshold: float,
                                anomalies: Optional[np.ndarray] = None,
                                title: str = "Reconstruction Error",
                                save_path: Optional[str] = None):
        """
        Plot reconstruction error with threshold and anomalies.
        
        Args:
            reconstruction_error: Array of reconstruction errors
            threshold: Anomaly threshold
            anomalies: Boolean array indicating anomalies (optional)
            title: Plot title
            save_path: Path to save the plot (if None, displays instead)
        """
        plt.figure(figsize=(15, 5))
        
        # Plot reconstruction error
        plt.plot(reconstruction_error, 'b-', alpha=0.7, label='Reconstruction Error')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
        
        # Highlight anomalies if provided
        if anomalies is not None:
            anomaly_indices = np.where(anomalies)[0]
            plt.scatter(anomaly_indices, reconstruction_error[anomaly_indices], 
                        color='red', alpha=0.7, label=f'Anomalies ({len(anomaly_indices)})')
        
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Reconstruction Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved reconstruction error plot to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_anomalies_on_data(data: np.ndarray, 
                             anomalies: np.ndarray, 
                             timestamps: Optional[np.ndarray] = None,
                             title: str = "Anomalies in Data",
                             save_path: Optional[str] = None):
        """
        Plot original data with anomalies highlighted.
        
        Args:
            data: Original data
            anomalies: Boolean array indicating anomalies
            timestamps: Timestamps for x-axis (optional)
            title: Plot title
            save_path: Path to save the plot (if None, displays instead)
        """
        plt.figure(figsize=(15, 5))
        
        # Use timestamps for x-axis if provided
        x = timestamps if timestamps is not None else np.arange(len(data))
        
        # Reshape data if needed
        if len(data.shape) > 1 and data.shape[1] == 1:
            data = data.flatten()
        
        # Plot original data
        plt.plot(x, data, 'b-', alpha=0.5, label='Original Data')
        
        # Highlight anomalies
        anomaly_indices = np.where(anomalies)[0]
        if len(anomaly_indices) > 0:
            plt.scatter(
                x[anomaly_indices] if timestamps is not None else anomaly_indices, 
                data[anomaly_indices], 
                color='red', alpha=0.7, s=50,
                label=f'Anomalies ({len(anomaly_indices)})'
            )
        
        plt.title(title)
        plt.xlabel('Time' if timestamps is not None else 'Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved anomalies plot to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]],
                            title: str = "Training History",
                            save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            history: Dictionary containing training history
            title: Plot title
            save_path: Path to save the plot (if None, displays instead)
        """
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        plt.plot(history['train_loss'], 'b-', label='Training Loss')
        
        # Plot validation loss if available
        if 'val_loss' in history and len(history['val_loss']) > 0:
            plt.plot(history['val_loss'], 'r-', label='Validation Loss')
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training history plot to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_multi_layer_anomalies(data: np.ndarray,
                                 anomaly_results: Dict[str, Any],
                                 timestamps: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None):
        """
        Plot multi-layer anomaly detection results.
        
        Args:
            data: Original data
            anomaly_results: Dictionary with multi-layer anomaly results
            timestamps: Timestamps for x-axis (optional)
            save_path: Path to save the plot (if None, displays instead)
        """
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
        
        # Use timestamps for x-axis if provided
        x = timestamps if timestamps is not None else np.arange(len(data))
        
        # Reshape data if needed
        if len(data.shape) > 1 and data.shape[1] == 1:
            data = data.flatten()
        
        # 1. Original data with combined anomalies
        axs[0].plot(x, data, 'b-', alpha=0.5, label='Original Data')
        combined_anomalies = anomaly_results['combined_anomalies']
        anomaly_indices = np.where(combined_anomalies)[0]
        if len(anomaly_indices) > 0:
            axs[0].scatter(
                x[anomaly_indices] if timestamps is not None else anomaly_indices, 
                data[anomaly_indices], 
                color='red', alpha=0.7, s=50,
                label=f'Combined Anomalies ({len(anomaly_indices)})'
            )
        axs[0].set_title('Combined Anomalies')
        axs[0].set_ylabel('Value')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # 2. Statistical anomalies
        axs[1].plot(x, data, 'b-', alpha=0.5, label='Original Data')
        if 'statistical_anomalies' in anomaly_results:
            stat_anomalies = anomaly_results['statistical_anomalies']['combined_statistical']
            if len(stat_anomalies.shape) > 1 and stat_anomalies.shape[1] > 1:
                # If multiple features, take any anomaly
                stat_anomalies = np.any(stat_anomalies, axis=1)
            stat_indices = np.where(stat_anomalies)[0]
            if len(stat_indices) > 0:
                axs[1].scatter(
                    x[stat_indices] if timestamps is not None else stat_indices, 
                    data[stat_indices], 
                    color='orange', alpha=0.7, s=50,
                    label=f'Statistical Anomalies ({len(stat_indices)})'
                )
        axs[1].set_title('Statistical Anomalies')
        axs[1].set_ylabel('Value')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        # 3. Seasonal anomalies if available
        axs[2].plot(x, data, 'b-', alpha=0.5, label='Original Data')
        if 'seasonal_anomalies' in anomaly_results and anomaly_results['seasonal_anomalies'] is not None:
            season_anomalies = anomaly_results['seasonal_anomalies']
            season_indices = np.where(season_anomalies)[0]
            if len(season_indices) > 0:
                axs[2].scatter(
                    x[season_indices] if timestamps is not None else season_indices, 
                    data[season_indices], 
                    color='green', alpha=0.7, s=50,
                    label=f'Seasonal Anomalies ({len(season_indices)})'
                )
        axs[2].set_title('Seasonal Anomalies')
        axs[2].set_ylabel('Value')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        
        # 4. ML-based anomalies
        axs[3].plot(x, data, 'b-', alpha=0.5, label='Original Data')
        if 'ml_anomalies' in anomaly_results:
            ml_anomalies = anomaly_results['ml_anomalies']
            ml_indices = np.where(ml_anomalies)[0]
            if len(ml_indices) > 0:
                axs[3].scatter(
                    x[ml_indices] if timestamps is not None else ml_indices, 
                    data[ml_indices], 
                    color='purple', alpha=0.7, s=50,
                    label=f'ML Anomalies ({len(ml_indices)})'
                )
        axs[3].set_title('ML-based Anomalies')
        axs[3].set_xlabel('Time' if timestamps is not None else 'Sample Index')
        axs[3].set_ylabel('Value')
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-layer anomalies plot to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_seasonal_patterns(data: np.ndarray,
                             seasonal_info: Dict[str, Any],
                             save_path: Optional[str] = None):
        """
        Plot seasonal patterns detected in the data.
        
        Args:
            data: Original data
            seasonal_info: Dictionary with seasonality information
            save_path: Path to save the plot (if None, displays instead)
        """
        fig, axs = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Daily pattern
        if 'daily_pattern' in seasonal_info:
            daily = seasonal_info['daily_pattern']
            daily_std = seasonal_info['daily_std']
            hours = daily.index
            
            axs[0].plot(hours, daily, 'b-', marker='o', label='Mean Value')
            axs[0].fill_between(hours, 
                              daily - daily_std, 
                              daily + daily_std, 
                              alpha=0.3, color='blue',
                              label='±1 Std Dev')
            axs[0].set_title('Daily Pattern')
            axs[0].set_xlabel('Hour of Day')
            axs[0].set_ylabel('Value')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
        else:
            axs[0].text(0.5, 0.5, 'Daily pattern not available', 
                      horizontalalignment='center', verticalalignment='center')
        
        # 2. Weekly pattern
        if 'weekly_pattern' in seasonal_info:
            weekly = seasonal_info['weekly_pattern']
            weekly_std = seasonal_info['weekly_std']
            days = weekly.index
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            axs[1].plot(days, weekly, 'g-', marker='o', label='Mean Value')
            axs[1].fill_between(days, 
                              weekly - weekly_std, 
                              weekly + weekly_std, 
                              alpha=0.3, color='green',
                              label='±1 Std Dev')
            axs[1].set_title('Weekly Pattern')
            axs[1].set_xlabel('Day of Week')
            axs[1].set_ylabel('Value')
            axs[1].set_xticks(days)
            axs[1].set_xticklabels(day_names)
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
        else:
            axs[1].text(0.5, 0.5, 'Weekly pattern not available', 
                      horizontalalignment='center', verticalalignment='center')
        
        # 3. Monthly pattern
        if 'monthly_pattern' in seasonal_info:
            monthly = seasonal_info['monthly_pattern']
            monthly_std = seasonal_info['monthly_std']
            months = monthly.index
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            axs[2].plot(months, monthly, 'r-', marker='o', label='Mean Value')
            axs[2].fill_between(months, 
                              monthly - monthly_std, 
                              monthly + monthly_std, 
                              alpha=0.3, color='red',
                              label='±1 Std Dev')
            axs[2].set_title('Monthly Pattern')
            axs[2].set_xlabel('Month')
            axs[2].set_ylabel('Value')
            axs[2].set_xticks(months)
            axs[2].set_xticklabels([month_names[i-1] for i in months])
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)
        else:
            axs[2].text(0.5, 0.5, 'Monthly pattern not available', 
                      horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved seasonal patterns plot to {save_path}")
        else:
            plt.show()