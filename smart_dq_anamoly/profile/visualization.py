import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

def visualize_profiling_summary(
    profile_df: pd.DataFrame,
    anomaly_df: Optional[pd.DataFrame] = None,
    key_metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize profiling metrics and highlight anomalies.

    Parameters:
    -----------
    profile_df : pd.DataFrame
        Dataframe with time window profiling metrics
    anomaly_df : Optional[pd.DataFrame]
        Dataframe containing anomaly flags/scores
    key_metrics : Optional[List[str]]
        Key metric names to highlight in the plot
    save_path : Optional[str]
        Folder path to save plots, if desired

    Returns:
    --------
    None
    """
    print("📈 Starting profiling metric visualization...")

    if 'window_start' in profile_df.columns:
        profile_df = profile_df.set_index('window_start')
    
    if anomaly_df is not None and 'window_start' in anomaly_df.columns:
        anomaly_df = anomaly_df.set_index('window_start')
    else:
        anomaly_df = profile_df

    sns.set(style="whitegrid")
    plt.rcParams.update({'figure.figsize': (15, 10)})

    # Plot 1: Total anomalies
    if 'total_anomalies' in anomaly_df.columns:
        plt.figure()
        plt.plot(anomaly_df.index, anomaly_df['total_anomalies'], 'b-', linewidth=2)
        plt.fill_between(anomaly_df.index, anomaly_df['total_anomalies'], alpha=0.3)
        plt.title('Total Anomalies Over Time')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(f"{save_path}/total_anomalies.png")
        else:
            plt.show()

    # Plot 2: Anomaly score
    if 'anomaly_score' in anomaly_df.columns:
        plt.figure()
        plt.plot(anomaly_df.index, anomaly_df['anomaly_score'], 'r-', linewidth=2)
        plt.fill_between(anomaly_df.index, anomaly_df['anomaly_score'], alpha=0.3)
        plt.title('Anomaly Score Over Time')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(f"{save_path}/anomaly_score.png")
        else:
            plt.show()

    # Plot 3: Row count
    if 'row_count' in profile_df.columns:
        plt.figure()
        plt.plot(profile_df.index, profile_df['row_count'], 'g-', linewidth=2)
        plt.fill_between(profile_df.index, profile_df['row_count'], alpha=0.3)
        if 'is_anomaly' in anomaly_df.columns:
            for date in anomaly_df[anomaly_df['is_anomaly'] == 1].index:
                plt.axvline(x=date, color='red', linestyle='--', alpha=0.3)
        plt.title('Number of Records Over Time')
        plt.ylabel('Row Count')
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(f"{save_path}/row_count.png")
        else:
            plt.show()

    # Plot 4: Key metrics
    if key_metrics:
        print("📊 Visualizing key metrics...")
        for metric in key_metrics:
            if metric not in profile_df.columns:
                print(f"⚠️ Skipping missing metric: {metric}")
                continue
            plt.figure()
            plt.plot(profile_df.index, profile_df[metric], label=metric, color='blue')
            if 'is_anomaly' in anomaly_df.columns:
                anomaly_points = profile_df.index[anomaly_df['is_anomaly'] == 1]
                plt.scatter(anomaly_points, profile_df.loc[anomaly_points, metric], color='red', s=50, label='Anomaly')
            plt.title(f"{metric} Over Time")
            plt.xlabel('Time')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, alpha=0.3)
            if save_path:
                plt.savefig(f"{save_path}/{metric}_plot.png")
            else:
                plt.show()

    print("✅ Visualization complete.")
