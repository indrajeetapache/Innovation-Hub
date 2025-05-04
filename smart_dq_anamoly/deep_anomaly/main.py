from wealth_time_series_converter import WealthTimeSeriesConverter
from flexible_anomaly_detection import FlexibleAnomalyDetector
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')  # Or use your data generator

# Convert to time series format
converter = WealthTimeSeriesConverter()
time_series_df = converter.prepare_for_lstm(df)

# Initialize the flexible detector
detector = FlexibleAnomalyDetector(
    default_config={
        'seq_length': 14,
        'hidden_dim': 64,
        'layer_dim': 2,
        'epochs': 50,
        'contamination': 0.01
    }
)

# Example 1: Analyze multiple accounts and multiple columns
results = detector.detect_anomalies_by_group(
    df=time_series_df,
    group_by='account_id',                    # Group by account_id
    target_columns=['balance', 'market_condition'],  # Analyze multiple columns
    timestamp_col='process_date',
    visualize=True,
    output_dir='anomaly_results_accounts'
)

# Access results
for account_id, account_results in results['results'].items():
    print(f"\nAccount: {account_id}")
    for column, column_results in account_results.items():
        if 'error' not in column_results:
            print(f"  {column}: {column_results['anomaly_count']} anomalies detected")
            print(f"  Anomaly dates: {column_results['anomaly_dates'][:5]}...")  # Show first 5

# Example 2: Group by different columns (e.g., by risk_profile)
results_by_risk = detector.detect_anomalies_by_group(
    df=time_series_df,
    group_by='risk_profile',                  # Group by risk_profile instead
    target_columns='balance',
    timestamp_col='process_date',
    config={'seq_length': 30},                # Override some config
    visualize=True,
    output_dir='anomaly_results_risk_profiles'
)

# Example 3: Multiple grouping columns
results_complex = detector.detect_anomalies_by_group(
    df=time_series_df,
    group_by=['account_type', 'risk_profile'],  # Group by multiple columns
    target_columns='balance',
    timestamp_col='process_date',
    visualize=True,
    output_dir='anomaly_results_complex'
)

# Example 4: Analyze entire dataset without grouping
results_all = detector.detect_anomalies_multiple_columns(
    df=time_series_df,
    target_columns=['balance', 'market_condition', 'interest_rate'],
    timestamp_col='process_date',
    visualize=True,
    output_dir='anomaly_results_all'
)

# Print summary
print("\n=== ANOMALY DETECTION SUMMARY ===")
print(f"Total anomalies found: {results['summary']['total_anomalies']}")
print(f"Groups with anomalies: {results['summary']['groups_with_anomalies']}")
print("\nAnomalies by column:")
for col, count in results['summary']['anomalies_by_column'].items():
    print(f"  {col}: {count}")