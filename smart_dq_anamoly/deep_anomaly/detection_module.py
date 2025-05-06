def detect_seasonal_anomalies(self, 
                             data: np.ndarray, 
                             timestamp: np.ndarray = None,
                             period: int = 24) -> np.ndarray:
    """
    Detect anomalies in seasonal patterns.
    
    Args:
        data: Input data of shape [n_samples, n_features]
        timestamp: Timestamps for data points
        period: Expected seasonality period
        
    Returns:
        Boolean array indicating seasonal anomalies
    """
    print("\n=== Starting Seasonal Anomaly Detection ===")
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        print("Reshaped 1D data to 2D")

    n_samples, n_features = data.shape
    print(f"Data shape: {n_samples} samples, {n_features} features")
    print(f"Expected seasonality period: {period}")

    if n_samples < 2 * period:
        print(f"⚠️ Warning: Not enough data points ({n_samples}) for seasonal analysis with period {period}")
        return np.zeros(n_samples, dtype=bool)

    seasonal_anomalies = np.zeros(n_samples, dtype=bool)

    print("Generating seasonal indices...")
    if timestamp is not None:
        seasonal_idx = np.zeros(n_samples, dtype=int)
        for i, ts in enumerate(timestamp):
            try:
                if hasattr(ts, 'hour'):  # pandas.Timestamp or datetime
                    seasonal_idx[i] = ts.hour
                    print(f"Index {i}: timestamp={ts}, extracted hour={ts.hour}")
                elif pd.api.types.is_datetime64_any_dtype(type(ts)):
                    dt = pd.Timestamp(ts).to_pydatetime()
                    seasonal_idx[i] = dt.hour
                    print(f"Index {i}: timestamp={ts}, extracted hour={dt.hour}")
                else:
                    seasonal_idx[i] = ts % period
                    print(f"Index {i}: numeric timestamp={ts}, modulo index={seasonal_idx[i]}")
            except Exception as e:
                print(f"⚠️ Error processing timestamp at index {i}: {e}. Falling back to i % period")
                seasonal_idx[i] = i % period
    else:
        print("No timestamps provided — using index-based modulo for seasonal grouping")
        seasonal_idx = np.arange(n_samples) % period

    print("\nAnalyzing each seasonal position...")
    for pos in range(period):
        pos_indices = seasonal_idx == pos
        count = np.sum(pos_indices)
        print(f"Seasonal position {pos}: {count} matching samples")

        if count > 5:
            for feat in range(n_features):
                pos_data = data[pos_indices, feat]
                pos_mean = np.mean(pos_data)
                pos_std = np.std(pos_data)
                print(f"  Feature {feat}: mean={pos_mean:.4f}, std={pos_std:.4f}")

                # Compute anomaly flags
                deviation = np.abs(data[pos_indices, feat] - pos_mean)
                anomaly_flags = deviation > 3 * pos_std
                seasonal_anomalies[pos_indices] |= anomaly_flags
                num_anomalies = np.sum(anomaly_flags)
                if num_anomalies > 0:
                    print(f"  ➤ {num_anomalies} anomalies detected in feature {feat} at seasonal position {pos}")
        else:
            print(f"  ⚠️ Skipping position {pos}: not enough samples ({count})")

    total_anomalies = np.sum(seasonal_anomalies)
    print(f"\n✅ Detected {total_anomalies} seasonal anomalies out of {n_samples} samples")
    print("=== Seasonal Anomaly Detection Complete ===\n")
    return seasonal_anomalies
