import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WealthTimeSeriesConverter:
    """
    Converts wealth management tabular data to proper time series format for anomaly detection.
    """

    def __init__(self):
        print("✅ WealthTimeSeriesConverter initialized")

    def convert_to_time_series(self, df, timestamp_col='process_date'):
        print("\n🔹 Step 1: Converting to continuous time series format...")

        # Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        accounts = df['account_id'].unique()
        min_date = df[timestamp_col].min()
        max_date = df[timestamp_col].max()

        print(f"Found {len(accounts)} unique accounts.")
        print(f"Data spans from {min_date.date()} to {max_date.date()}.")

        all_series = []

        for idx, account_id in enumerate(accounts, 1):
            account_data = df[df['account_id'] == account_id].copy()

            if len(account_data) <= 1:
                print(f"⚠️ Skipping account {account_id} (only {len(account_data)} record)")
                continue

            account_data = account_data.sort_values(timestamp_col)

            customer_id = account_data['customer_id'].iloc[0]
            account_type = account_data['account_type'].iloc[0]
            risk_profile = account_data['risk_profile'].iloc[0]
            relationship_manager = account_data['relationship_manager'].iloc[0]

            date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            template = pd.DataFrame({
                timestamp_col: date_range,
                'account_id': account_id,
                'customer_id': customer_id,
                'account_type': account_type,
                'risk_profile': risk_profile,
                'relationship_manager': relationship_manager
            })

            merged = template.merge(account_data, on=[timestamp_col, 'account_id', 'customer_id', 
                                                      'account_type', 'risk_profile', 'relationship_manager'], 
                                    how='left')

            numeric_cols = ['balance', 'market_condition', 'trend_factor', 'risk_adjusted_factor', 
                            'account_type_factor', 'base_balance']

            for col in numeric_cols:
                if col in merged.columns:
                    merged[col] = merged[col].ffill().bfill()

            all_series.append(merged)

            if idx % 10 == 0 or idx == len(accounts):
                print(f"Processed {idx}/{len(accounts)} accounts")

        result = pd.concat(all_series, ignore_index=True)
        print(f"✅ Completed time series conversion: {len(result)} rows created.")
        return result

    def filter_accounts_with_sufficient_data(self, df, timestamp_col='process_date', min_days=30):
        print("\n🔹 Step 2: Filtering accounts with sufficient history...")

        account_days = df.groupby('account_id')[timestamp_col].nunique()
        sufficient_accounts = account_days[account_days >= min_days].index.tolist()

        filtered_df = df[df['account_id'].isin(sufficient_accounts)].copy()

        print(f"From {len(account_days)} accounts, retained {len(sufficient_accounts)} accounts having >= {min_days} days.")
        print(f"Final row count after filtering: {len(filtered_df)}")
        return filtered_df

    def add_derived_features(self, df, timestamp_col='process_date'):
        print("\n🔹 Step 3: Adding derived features for anomaly detection...")

        result = df.copy()
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])

        result['day_of_week'] = result[timestamp_col].dt.dayofweek
        result['day_of_month'] = result[timestamp_col].dt.day
        result['month'] = result[timestamp_col].dt.month
        result['is_month_end'] = result[timestamp_col].dt.is_month_end.astype(int)

        accounts = result['account_id'].unique()
        all_processed = []

        for idx, account_id in enumerate(accounts, 1):
            account_data = result[result['account_id'] == account_id].copy()
            account_data = account_data.sort_values(timestamp_col)

            if len(account_data) >= 7:
                account_data['balance_7d_mean'] = account_data['balance'].rolling(window=7, min_periods=1).mean()
                account_data['balance_7d_std'] = account_data['balance'].rolling(window=7, min_periods=1).std()
                account_data['balance_change'] = account_data['balance'].diff().fillna(0)
                account_data['balance_pct_change'] = account_data['balance'].pct_change().fillna(0)
                account_data['balance_zscore'] = (account_data['balance'] - account_data['balance_7d_mean']) / account_data['balance_7d_std'].replace(0, 1)

            all_processed.append(account_data)

            if idx % 10 == 0 or idx == len(accounts):
                print(f"Feature engineered {idx}/{len(accounts)} accounts")

        result = pd.concat(all_processed, ignore_index=True)
        print(f"✅ Derived feature addition complete. Final shape: {result.shape}")
        return result

    def prepare_for_lstm(self, tabular_df, timestamp_col='process_date'):
        print("\n🚀 Starting full time series preparation pipeline for LSTM model...")

        ts_data = self.convert_to_time_series(tabular_df, timestamp_col)
        filtered_data = self.filter_accounts_with_sufficient_data(ts_data, timestamp_col, min_days=30)
        prepared_data = self.add_derived_features(filtered_data, timestamp_col)

        print("\n🎯 Time series preparation pipeline completed!")
        return prepared_data
