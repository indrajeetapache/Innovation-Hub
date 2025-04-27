# WealthTimeSeriesConverter Documentation

This document explains the design, purpose, and transformation pipeline of the `WealthTimeSeriesConverter` class.
It helps understand why each step is necessary when preparing wealth management data for time series anomaly detection and LSTM modeling.

---

# 📋 Summary: How WealthTimeSeriesConverter Works

---

## 1. `convert_to_time_series(df)`

🔸 **Goal:**
Ensure each account has **continuous daily records** between the earliest and latest dates — even if some days are missing in the original data.

🔸 **Operations:**
- Converts the `process_date` to datetime format.
- Finds all unique `account_id`s.
- Identifies the minimum and maximum date range.
- For each account:
  - Creates a **full daily date range** (even for missing days).
  - Merges the actual data into the full range.
  - **Forward-fills** and **backward-fills** missing numeric fields (`balance`, `market_condition`, etc.).
- Combines all account data into a single DataFrame.

🔸 **Why:**
LSTM models expect continuous time series without gaps. Missing dates can confuse the model during training.

---

## 2. `filter_accounts_with_sufficient_data(df, min_days=30)`

🔸 **Goal:**
Remove accounts that have **too little history** (default: less than 30 days).

🔸 **Operations:**
- Groups data by `account_id`.
- Counts the number of unique dates for each account.
- Retains only accounts that meet or exceed the `min_days` threshold.

🔸 **Why:**
Accounts with very few data points do not provide enough sequence information for training and increase the risk of poor model performance.

---

## 3. `add_derived_features(df)`

🔸 **Goal:**
Add **time-aware features** to improve model capability in detecting patterns and anomalies.

🔸 **Operations:**
- Adds **calendar features**:
  - `day_of_week` (0=Monday, 6=Sunday)
  - `day_of_month`
  - `month`
  - `is_month_end` (1 if end of month, else 0)
- For each account:
  - Computes **7-day moving average** and **7-day standard deviation** of `balance`.
  - Calculates **daily balance change** and **percentage change**.
  - Computes **Z-score** of balance using 7-day rolling statistics.

🔸 **Why:**
These features help capture **seasonality**, **trends**, **sudden jumps**, and **outliers**, making it easier for anomaly detection models to learn.

---

## 4. `prepare_for_lstm(df)`

🔸 **Goal:**
Provide a **single function** to fully prepare wealth management tabular data for LSTM input.

🔸 **Pipeline:**
- Step 1: `convert_to_time_series()`
- Step 2: `filter_accounts_with_sufficient_data()`
- Step 3: `add_derived_features()`

🔸 **Why:**
Simplifies the workflow — users don't need to manually call each step individually.

---

# 🚀 One-liner Explanation:
> WealthTimeSeriesConverter **fills missing dates**, **filters weak accounts**, **engineers time-based features**, and **prepares clean input** for anomaly detection and LSTM modeling.

---

# 🏠 Visual Data Transformation Flow

```plaintext
Original Tabular Wealth Data
    ↓
Convert to Continuous Time Series (fills missing days)
    ↓
Filter Short Accounts (keep only stable accounts)
    ↓
Add Rolling Features (mean, std, z-score, etc.)
    ↓
✅ Final DataFrame Ready for LSTM
```

---

# Notes:
- Always ensure numeric columns are present when merging or rolling statistics.
- Accounts with very few records should be discarded to avoid training instability.
- Derived features can be easily extended if needed (e.g., monthly returns, cumulative sums).

---

This documentation should be placed inside the same `datagenerator` package under a file like:

```bash
Innovation-Hub/smart_dq_anamoly/datagenerator/wealth_time_series_converter_README.md
```


