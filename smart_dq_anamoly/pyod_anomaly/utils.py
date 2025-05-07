# pyod_anomaly/utils.py
def detect_multiple_columns_pyod(detector, df, target_columns, feature_cols=None):
    """Run PyodDetector on multiple columns."""
    results = {}
    
    for target_col in target_columns:
        print(f"\n======= Processing '{target_col}' with PYOD =======")
        
        # Use other columns as features if none specified
        col_features = feature_cols
        if col_features is None:
            # Use other target columns as features
            col_features = [col for col in target_columns if col != target_col]
            if col_features:
                print(f"[AUTO] Using other target columns as features: {col_features}")
        
        # Run detection
        col_result = detector.detect_anomalies(
            df=df,
            target_col=target_col,
            feature_cols=col_features
        )
        
        results[target_col] = col_result
    
    return results