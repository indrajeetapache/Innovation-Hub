import numpy as np
import pandas as pd

class AnomalyFusion:
    """Combine results from multiple anomaly detection methods."""

    @staticmethod
    def combine_results(lstm_results, pyod_results, method='union'):
        """
        Combine LSTM and PYOD anomaly detection results.

        Args:
            lstm_results: Results from LSTM detector
            pyod_results: Results from PYOD detector
            method: Combination method ('union', 'intersection', 'weighted')

        Returns:
            Dictionary with combined results
        """
        print(f"\n[AnomalyFusion] Starting combination using method: '{method}'")

        # Extract indices
        lstm_indices = set(lstm_results['anomaly_indices'])
        pyod_indices = set(pyod_results['anomaly_indices'])
        print(f"[AnomalyFusion] LSTM anomaly count: {len(lstm_indices)}")
        print(f"[AnomalyFusion] PYOD anomaly count: {len(pyod_indices)}")

        # Combine based on method
        if method == 'union':
            combined_indices = lstm_indices.union(pyod_indices)
            print(f"[AnomalyFusion] Union of anomalies → Total: {len(combined_indices)}")

        elif method == 'intersection':
            combined_indices = lstm_indices.intersection(pyod_indices)
            print(f"[AnomalyFusion] Intersection of anomalies → Total: {len(combined_indices)}")

        elif method == 'weighted':
            print("[AnomalyFusion] Performing weighted score fusion...")

            # Need scores from both methods
            if 'anomaly_scores' not in lstm_results or 'anomaly_scores' not in pyod_results:
                raise ValueError("Weighted combination requires anomaly scores from both methods")

            lstm_scores = lstm_results['anomaly_scores']
            pyod_scores = pyod_results['anomaly_scores']

            print(f"[AnomalyFusion] Raw LSTM scores shape: {lstm_scores.shape}")
            print(f"[AnomalyFusion] Raw PYOD scores shape: {pyod_scores.shape}")

            # Normalize scores
            lstm_norm = (lstm_scores - lstm_scores.min()) / (lstm_scores.max() - lstm_scores.min() + 1e-10)
            pyod_norm = (pyod_scores - pyod_scores.min()) / (pyod_scores.max() - pyod_scores.min() + 1e-10)

            combined_scores = (lstm_norm + pyod_norm) / 2
            contamination = lstm_results.get('contamination', 0.01)
            threshold = np.percentile(combined_scores, 100 * (1 - contamination))

            print(f"[AnomalyFusion] Contamination level: {contamination}")
            print(f"[AnomalyFusion] Threshold calculated: {threshold:.6f}")

            combined_indices = np.where(combined_scores > threshold)[0]
            print(f"[AnomalyFusion] Weighted fusion → Total anomalies: {len(combined_indices)}")

        else:
            raise ValueError(f"Unsupported combination method: {method}")

        result = {
            'combined_indices': sorted(combined_indices),
            'lstm_indices': sorted(lstm_indices),
            'pyod_indices': sorted(pyod_indices),
            'combination_method': method,
            'total_anomalies': len(combined_indices)
        }

        print(f"[AnomalyFusion] Combination complete. Returning result dictionary.\n")
        return result

    @staticmethod
    def extract_lstm_result_for_fusion(profile_output: dict) -> dict:
        """
        Prepare LSTM profile result dict for fusion by extracting required keys.

        Args:
            profile_output: Dict for a target from lstm_profile_results

        Returns:
            Dict with 'anomaly_indices' and 'anomaly_scores'
        """
        if 'anomaly_indices' not in profile_output:
            raise ValueError("Missing 'anomaly_indices' in LSTM profile result.")
        if 'full_results' not in profile_output or 'reconstruction_errors' not in profile_output['full_results']:
            raise ValueError("Missing 'reconstruction_errors' in LSTM profile result.")
        
        return {
            'anomaly_indices': profile_output['anomaly_indices'],
            'anomaly_scores': profile_output['full_results']['reconstruction_errors'],
            'contamination': 0.01  # Optional: customize per config
        }
