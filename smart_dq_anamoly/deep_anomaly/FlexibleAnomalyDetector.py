import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from core_pipeline import run_anomaly_detection
import os
import matplotlib.pyplot as plt

class FlexibleAnomalyDetector:
    """
    A flexible wrapper for anomaly detection that can handle:
    - Multiple group-by columns (not just account_id)
    - Multiple target columns to analyze
    - Batch processing of many time series
    - Customizable analysis parameters
    """
    
    def __init__(self, default_config: Dict[str, Any] = None):
        """
        Initialize the detector with default configuration.
        
        Args:
            default_config: Default configuration parameters for anomaly detection
        """
        self.default_config = default_config or {
            'seq_length': 14,
            'hidden_dim': 64,
            'layer_dim': 2,
            'batch_size': 32,
            'lr': 0.001,
            'epochs': 50,
            'patience': 5,
            'threshold_method': 'gaussian',
            'contamination': 0.01,
            'detect_daily_seasonality': True,
            'detect_weekly_seasonality': True
        }
        
    def detect_anomalies_by_group(
        self,
        df: pd.DataFrame,
        group_by: Union[str, List[str]],
        target_columns: Union[str, List[str]],
        timestamp_col: str,
        config: Dict[str, Any] = None,
        min_samples: int = 30,
        visualize: bool = False,
        output_dir: str = "anomaly_results"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run anomaly detection on multiple groups and multiple target columns.
        
        Args:
            df: Input DataFrame
            group_by: Column(s) to group by (e.g., 'account_id', 'customer_id', ['region', 'product'])
            target_columns: Column(s) to analyze for anomalies (e.g., 'balance', ['balance', 'transaction_amount'])
            timestamp_col: Column containing timestamps
            config: Configuration overrides for anomaly detection
            min_samples: Minimum number of samples required for a group
            visualize: Whether to create visualizations
            output_dir: Directory to save results
            
        Returns:
            Dictionary with results for each group and target column
        """
        # Ensure group_by and target_columns are lists
        if isinstance(group_by, str):
            group_by = [group_by]
        if isinstance(target_columns, str):
            target_columns = [target_columns]
            
        # Merge configs
        detection_config = self.default_config.copy()
        if config:
            detection_config.update(config)
        
        # Create output directory
        if visualize and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Results storage
        all_results = {}
        
        # Group the data
        grouped = df.groupby(group_by)
        total_groups = len(grouped)
        
        print(f"Processing {total_groups} groups with {len(target_columns)} target columns each...")
        
        for idx, (group_key, group_df) in enumerate(grouped):
            # Convert group_key to string for dictionary key
            if isinstance(group_key, tuple):
                group_key_str = "_".join([str(k) for k in group_key])
            else:
                group_key_str = str(group_key)
            
            print(f"\nProcessing group {idx+1}/{total_groups}: {group_key_str}")
            
            # Skip if insufficient data
            if len(group_df) < min_samples:
                print(f"Skipping {group_key_str} - insufficient data ({len(group_df)} samples)")
                continue
            
            # Sort by timestamp
            group_df = group_df.sort_values(timestamp_col)
            
            # Initialize results for this group
            group_results = {}
            
            # Analyze each target column
            for target_col in target_columns:
                print(f"  Analyzing column: {target_col}")
                
                try:
                    # Run anomaly detection
                    results = run_anomaly_detection(
                        df=group_df,
                        target_col=target_col,
                        timestamp_col=timestamp_col,
                        config_params=detection_config,
                        visualize=visualize
                    )
                    
                    # Extract key results
                    anomalies = results['anomaly_results']['combined_anomalies']
                    anomaly_dates = group_df.iloc[anomalies == 1][timestamp_col].tolist()
                    
                    # Store results
                    group_results[target_col] = {
                        'anomaly_count': sum(anomalies),
                        'anomaly_dates': anomaly_dates,
                        'anomaly_indices': np.where(anomalies == 1)[0].tolist(),
                        'full_results': results
                    }
                    
                    print(f"    Found {sum(anomalies)} anomalies")
                    
                    # Save visualization if requested
                    if visualize:
                        self._save_group_visualization(
                            group_df, 
                            anomalies, 
                            target_col, 
                            timestamp_col,
                            group_key_str, 
                            output_dir
                        )
                    
                except Exception as e:
                    print(f"    Error analyzing {target_col}: {str(e)}")
                    group_results[target_col] = {'error': str(e)}
            
            # Store group results
            all_results[group_key_str] = group_results
        
        # Generate summary
        summary = self._generate_summary(all_results)
        
        # Save summary if visualizations are enabled
        if visualize:
            self._save_summary(summary, output_dir)
        
        return {
            'results': all_results,
            'summary': summary
        }
    
    def detect_anomalies_multiple_columns(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        timestamp_col: str,
        config: Dict[str, Any] = None,
        visualize: bool = False,
        output_dir: str = "anomaly_results"
    ) -> Dict[str, Any]:
        """
        Run anomaly detection on multiple columns without grouping.
        
        Args:
            df: Input DataFrame
            target_columns: List of columns to analyze
            timestamp_col: Column containing timestamps
            config: Configuration overrides
            visualize: Whether to create visualizations
            output_dir: Directory to save results
            
        Returns:
            Dictionary with results for each target column
        """
        # Merge configs
        detection_config = self.default_config.copy()
        if config:
            detection_config.update(config)
        
        # Create output directory
        if visualize and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Results storage
        results = {}
        
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col)
        
        print(f"Analyzing {len(target_columns)} columns...")
        
        for target_col in target_columns:
            print(f"\nAnalyzing column: {target_col}")
            
            try:
                # Run anomaly detection
                col_results = run_anomaly_detection(
                    df=df_sorted,
                    target_col=target_col,
                    timestamp_col=timestamp_col,
                    config_params=detection_config,
                    visualize=visualize
                )
                
                # Extract key results
                anomalies = col_results['anomaly_results']['combined_anomalies']
                anomaly_dates = df_sorted.iloc[anomalies == 1][timestamp_col].tolist()
                
                # Store results
                results[target_col] = {
                    'anomaly_count': sum(anomalies),
                    'anomaly_dates': anomaly_dates,
                    'anomaly_indices': np.where(anomalies == 1)[0].tolist(),
                    'full_results': col_results
                }
                
                print(f"  Found {sum(anomalies)} anomalies")
                
            except Exception as e:
                print(f"  Error analyzing {target_col}: {str(e)}")
                results[target_col] = {'error': str(e)}
        
        return results
    
    def _save_group_visualization(
        self, 
        group_df: pd.DataFrame, 
        anomalies: np.ndarray,
        target_col: str,
        timestamp_col: str,
        group_key: str,
        output_dir: str
    ):
        """Save visualization for a specific group."""
        plt.figure(figsize=(12, 6))
        plt.plot(group_df[timestamp_col], group_df[target_col], label='Normal', alpha=0.7)
        
        # Highlight anomalies
        anomaly_mask = anomalies == 1
        if any(anomaly_mask):
            plt.scatter(
                group_df[timestamp_col].iloc[anomaly_mask],
                group_df[target_col].iloc[anomaly_mask],
                color='red',
                label='Anomaly',
                s=50,
                zorder=5
            )
        
        plt.title(f'Anomalies in {target_col} for {group_key}')
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{group_key}_{target_col}_anomalies.png')
        plt.savefig(plot_path)
        plt.close()
    
    def _generate_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        summary = {
            'total_groups': len(results),
            'groups_with_anomalies': 0,
            'total_anomalies': 0,
            'anomalies_by_column': {},
            'groups_by_anomaly_count': {}
        }
        
        for group_key, group_results in results.items():
            group_anomalies = 0
            for target_col, col_results in group_results.items():
                if 'error' not in col_results:
                    anomaly_count = col_results['anomaly_count']
                    group_anomalies += anomaly_count
                    
                    # Update column statistics
                    if target_col not in summary['anomalies_by_column']:
                        summary['anomalies_by_column'][target_col] = 0
                    summary['anomalies_by_column'][target_col] += anomaly_count
            
            # Update group statistics
            if group_anomalies > 0:
                summary['groups_with_anomalies'] += 1
            summary['total_anomalies'] += group_anomalies
            summary['groups_by_anomaly_count'][group_key] = group_anomalies
        
        return summary
    
    def _save_summary(self, summary: Dict[str, Any], output_dir: str):
        """Save summary statistics to a text file."""
        summary_path = os.path.join(output_dir, 'anomaly_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Anomaly Detection Summary\n")
            f.write("========================\n\n")
            f.write(f"Total groups analyzed: {summary['total_groups']}\n")
            f.write(f"Groups with anomalies: {summary['groups_with_anomalies']}\n")
            f.write(f"Total anomalies detected: {summary['total_anomalies']}\n\n")
            
            f.write("Anomalies by column:\n")
            for col, count in summary['anomalies_by_column'].items():
                f.write(f"  {col}: {count}\n")
            
            f.write("\nTop 10 groups by anomaly count:\n")
            sorted_groups = sorted(
                summary['groups_by_anomaly_count'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            for group, count in sorted_groups:
                f.write(f"  {group}: {count}\n")


# Example usage
if __name__ == "__main__":
    # Example 1: Detect anomalies by account_id for multiple columns
    detector = FlexibleAnomalyDetector()
    
    # Analyze by account_id
    results = detector.detect_anomalies_by_group(
        df=time_series_df,
        group_by='account_id',
        target_columns=['balance', 'transaction_amount'],
        timestamp_col='process_date',
        visualize=True
    )
    
    # Example 2: Analyze by multiple grouping columns
    results = detector.detect_anomalies_by_group(
        df=time_series_df,
        group_by=['region', 'product_type'],
        target_columns='sales',
        timestamp_col='date',
        visualize=True
    )
    
    # Example 3: Analyze multiple columns without grouping
    results = detector.detect_anomalies_multiple_columns(
        df=time_series_df,
        target_columns=['balance', 'transaction_count', 'avg_transaction_amount'],
        timestamp_col='process_date',
        visualize=True
    )