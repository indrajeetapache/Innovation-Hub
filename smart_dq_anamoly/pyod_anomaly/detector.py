from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class PyodDetector:
    """PYOD-based anomaly detection for financial time series with verbose logging."""
    
    def __init__(self, contamination=0.01, method='ecod', n_folds=5):
        self.contamination = contamination
        self.method = method
        self.model = None
        self.n_folds = n_folds
        print(f"[INIT] PyodDetector initialized with method='{method}', contamination={contamination}, n_folds={n_folds}")
        
    def detect_anomalies(self, df, target_col, feature_cols=None):
        print(f"\n[START] Starting anomaly detection on column: '{target_col}'")

        # Prepare features
        if feature_cols:
            print(f"[INFO] Using additional features: {feature_cols}")
            X = df[feature_cols + [target_col]].values
        else:
            print(f"[INFO] Using only target column '{target_col}'")
            X = df[target_col].values.reshape(-1, 1)
        
        print(f"[DATA] Shape of input matrix X: {X.shape}")
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        all_scores = np.zeros(len(X))
        all_flags = np.zeros(len(X), dtype=int)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"\n[CV] Fold {fold + 1}/{self.n_folds}")
            print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")

            if self.method == 'ecod':
                model = ECOD(contamination=self.contamination)
                print("  [MODEL] Using ECOD")
            elif self.method == 'copod':
                model = COPOD(contamination=self.contamination)
                print("  [MODEL] Using COPOD")
            elif self.method == 'ensemble':
                print("  [MODEL] Running ensemble of ECOD + COPOD")
                fold_result = self._run_ensemble_cv(X, train_idx, test_idx, df, target_col)
                fold_results.append(fold_result)
                all_scores[test_idx] = fold_result['anomaly_scores']
                all_flags[test_idx] = fold_result['anomaly_flags']
                continue
            
            # Train and predict
            model.fit(X[train_idx])
            print(f"  [TRAIN] Model trained on fold {fold}")

            test_scores = model.decision_function(X[test_idx])
            threshold = model.threshold_
            print(f"  [THRESHOLD] Fold {fold} threshold = {threshold:.6f}")
            test_flags = (test_scores > threshold).astype(int)
            anomaly_count = np.sum(test_flags)
            print(f"  [DETECT] Detected {anomaly_count} anomalies in fold {fold}")

            # Store results
            all_scores[test_idx] = test_scores
            all_flags[test_idx] = test_flags
            fold_results.append({
                'fold': fold,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'threshold': threshold
            })
        
        print(f"\n[COMPLETE] Total anomalies detected: {np.sum(all_flags)} out of {len(X)} samples")
        return {
            'anomaly_flags': all_flags,
            'anomaly_indices': np.where(all_flags == 1)[0],
            'anomaly_scores': all_scores,
            'method': self.method,
            'fold_results': fold_results
        }
    
    def _run_ensemble_cv(self, X, train_idx, test_idx, df, target_col):
        X_train, X_test = X[train_idx], X[test_idx]
        print("  [ENSEMBLE] Training ECOD and COPOD")

        ecod = ECOD(contamination=self.contamination)
        copod = COPOD(contamination=self.contamination)

        ecod.fit(X_train)
        copod.fit(X_train)
        print("  [FIT] Both models fitted")

        ecod_scores = ecod.decision_function(X_test)
        copod_scores = copod.decision_function(X_test)

        ecod_norm = (ecod_scores - ecod_scores.min()) / max(ecod_scores.max() - ecod_scores.min(), 1e-10)
        copod_norm = (copod_scores - copod_scores.min()) / max(copod_scores.max() - copod_scores.min(), 1e-10)

        combined_scores = (ecod_norm + copod_norm) / 2
        threshold = np.percentile(combined_scores, 100 * (1 - self.contamination))
        combined_flags = (combined_scores > threshold).astype(int)
        anomaly_count = np.sum(combined_flags)

        print(f"  [ENSEMBLE THRESHOLD] Combined score threshold: {threshold:.6f}")
        print(f"  [ENSEMBLE DETECT] Detected {anomaly_count} anomalies in ensemble")

        return {
            'anomaly_flags': combined_flags,
            'anomaly_scores': combined_scores,
            'ecod_scores': ecod_scores,
            'copod_scores': copod_scores,
            'threshold': threshold
        }
