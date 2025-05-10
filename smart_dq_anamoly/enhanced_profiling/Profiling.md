# 📊 Time Series Profiling Report

This report documents the profiling logic used in the `TimeSeriesProfiler` class for generating statistical summaries over time, grouped by `process_date` or any other datetime column. The profiler automatically handles numeric and categorical columns separately and computes a variety of metrics useful for anomaly detection and data quality validation.

---

## 🔢 Numeric Column Profiling

These features are profiled **per day**, and compared with historical windows of `[7, 14, 30, 90]` days.

### 📌 Basic Descriptive Statistics

| Metric       | Description                   | Purpose                          |
| ------------ | ----------------------------- | -------------------------------- |
| `mean`       | Average of current values     | Detect central tendency shifts   |
| `median`     | 50th percentile (robust)      | Handle skewed distributions      |
| `std`        | Standard deviation            | Detect volatility                |
| `min`, `max` | Smallest / largest value      | Identify outliers                |
| `range`      | `max - min`                   | Track daily dispersion           |
| `iqr`        | Interquartile range (Q3 - Q1) | Robust variability check         |
| `skew`       | Distribution asymmetry        | Detect left/right tail heaviness |
| `kurtosis`   | Distribution tailedness       | Spot extreme values              |
| `zeros_pct`  | % of values that are 0        | Spot inactive/broken features    |
| `nulls_pct`  | % of null/missing values      | Evaluate data completeness       |

### 🔁 Rolling Window Statistics

| Metric             | Description                                          | Use Case                      |
| ------------------ | ---------------------------------------------------- | ----------------------------- |
| `mean_diff_{N}d`   | Change from rolling historical mean                  | Detect level shift            |
| `median_diff_{N}d` | Change from rolling historical median                | Detect robust central shift   |
| `std_ratio_{N}d`   | Std deviation relative to rolling std                | Detect volatility shift       |
| `range_ratio_{N}d` | Daily range vs historical range                      | Detect sudden spikes or drops |
| `zscore_{N}d`      | Distance from historical mean in standard deviations | Flag statistical anomalies    |

---

## 🔠 Categorical Column Profiling

Categorical columns (e.g., `account_type`, `risk_profile`) are evaluated using entropy, frequency patterns, and distribution drift.

### 📌 Single-Day Summary

| Metric         | Description               | Purpose                           |
| -------------- | ------------------------- | --------------------------------- |
| `unique_count` | Number of distinct values | Detect collapse in diversity      |
| `mode`         | Most common value         | Flag domination/imbalance         |
| `entropy`      | Disorder in distribution  | Detect concentrated distributions |
| `nulls_pct`    | % of missing values       | Quality check                     |

### 🔁 Rolling Window Distribution Drift

| Metric                | Description                                             | Use Case                          |
| --------------------- | ------------------------------------------------------- | --------------------------------- |
| `unique_diff_{N}d`    | Difference in unique values compared to rolling history | Identify diversity drops/spikes   |
| `js_distance_{N}d`    | Jensen-Shannon distance from historical distribution    | Spot gradual or sudden data drift |
| `new_categories_{N}d` | # of new values not seen in rolling window              | Catch emerging patterns or errors |

---

