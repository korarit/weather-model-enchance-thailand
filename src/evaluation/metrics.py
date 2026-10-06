"""
Hydrological Evaluation Metrics Suite
Implements:
1. Continuous Quantitative Metrics:
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - Error Variance Ratio (RMSE / MAE)
   - Skill Scores (SS_RMSE, SS_MAE relative to Raw NWP Baseline B0)
   - Mean Bias Error (MBE / Systematic Bias)
   - Pearson Correlation (r) & Spearman Rank Correlation (rho)

2. Categorical Rain Thresholds & Contingency Metrics:
   - Thresholds: >=0.1 mm/h (Rain), >=2.0 mm/h (Moderate), >=10.0 mm/h (Heavy), >=20.0 mm/h (Very Heavy)
   - Hits (H), False Alarms (F), Misses (M), Correct Rejections (C)
   - Probability of Detection (POD / Hit Rate)
   - False Alarm Ratio (FAR)
   - Critical Miss Rate (Miss Rate = 100% - POD)
   - Critical Success Index (CSI / Threat Score)
   - F1-Score & Success Ratio (SR = 1 - FAR)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, List, Union


def calculate_continuous_metrics(
    obs: Union[np.ndarray, pd.Series, List[float]],
    pred: Union[np.ndarray, pd.Series, List[float]],
    raw_pred: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
) -> Dict[str, float]:
    """
    Computes continuous hydrological verification metrics.
    
    Args:
        obs: Observed precipitation (mm/h)
        pred: Model predicted precipitation (mm/h)
        raw_pred: Optional raw NWP precipitation (mm/h) for Skill Score calculation
        
    Returns:
        Dict with keys: rmse, mae, ratio, mbe, pearson_r, spearman_rho, ss_rmse, ss_mae, n_samples
    """
    obs_arr = np.asarray(obs, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    
    # Filter valid finite pairs
    valid_mask = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    if raw_pred is not None:
        raw_arr = np.asarray(raw_pred, dtype=float)
        valid_mask = valid_mask & np.isfinite(raw_arr)
    else:
        raw_arr = None

    obs_valid = obs_arr[valid_mask]
    pred_valid = pred_arr[valid_mask]
    n_samples = int(len(obs_valid))
    
    if n_samples == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "ratio": np.nan,
            "mbe": np.nan,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "ss_rmse": np.nan,
            "ss_mae": np.nan,
            "n_samples": 0,
        }

    diff = pred_valid - obs_valid
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    ratio = float(rmse / mae) if mae > 1e-9 else 1.0
    mbe = float(np.mean(diff))

    # Pearson & Spearman
    if np.std(obs_valid) > 1e-9 and np.std(pred_valid) > 1e-9:
        pearson_r = float(np.corrcoef(obs_valid, pred_valid)[0, 1])
        spearman_rho = float(stats.spearmanr(obs_valid, pred_valid)[0])
    else:
        pearson_r = 0.0
        spearman_rho = 0.0

    # Skill scores relative to raw NWP baseline
    if raw_arr is not None:
        raw_valid = raw_arr[valid_mask]
        raw_diff = raw_valid - obs_valid
        raw_rmse = float(np.sqrt(np.mean(raw_diff ** 2)))
        raw_mae = float(np.mean(np.abs(raw_diff)))
        
        ss_rmse = float((1.0 - (rmse / raw_rmse)) * 100.0) if raw_rmse > 1e-9 else 0.0
        ss_mae = float((1.0 - (mae / raw_mae)) * 100.0) if raw_mae > 1e-9 else 0.0
    else:
        ss_rmse = np.nan
        ss_mae = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "ratio": ratio,
        "mbe": mbe,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "ss_rmse": ss_rmse,
        "ss_mae": ss_mae,
        "n_samples": n_samples,
    }


def calculate_contingency_metrics(
    obs: Union[np.ndarray, pd.Series, List[float]],
    pred: Union[np.ndarray, pd.Series, List[float]],
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Computes 2x2 contingency table and categorical verification metrics for a rain threshold.
    
    Args:
        obs: Observed precipitation (mm/h)
        pred: Model predicted precipitation (mm/h)
        threshold: Rain rate threshold (mm/h), default 0.1
        
    Returns:
        Dict with keys: threshold, hits, misses, false_alarms, correct_negatives,
                        pod, far, miss_rate, csi, f1, success_ratio, bias_score
    """
    obs_arr = np.asarray(obs, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    
    valid_mask = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    obs_v = obs_arr[valid_mask]
    pred_v = pred_arr[valid_mask]

    obs_rain = obs_v >= threshold
    pred_rain = pred_v >= threshold

    hits = int(np.sum(obs_rain & pred_rain))          # H
    misses = int(np.sum(obs_rain & ~pred_rain))        # M
    false_alarms = int(np.sum(~obs_rain & pred_rain))  # F
    correct_negatives = int(np.sum(~obs_rain & ~pred_rain)) # C

    # POD (Hit Rate): H / (H + M)
    pod = float((hits / (hits + misses)) * 100.0) if (hits + misses) > 0 else 0.0
    
    # FAR: F / (H + F)
    far = float((false_alarms / (hits + false_alarms)) * 100.0) if (hits + false_alarms) > 0 else 0.0
    
    # Critical Miss Rate: M / (H + M) = 100 - POD
    miss_rate = float(100.0 - pod) if (hits + misses) > 0 else 0.0
    
    # CSI (Threat Score): H / (H + M + F)
    csi_denom = hits + misses + false_alarms
    csi = float(hits / csi_denom) if csi_denom > 0 else 0.0
    
    # Precision & Recall for F1
    precision = float(hits / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
    recall = float(hits / (hits + misses)) if (hits + misses) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Success Ratio (SR = 1 - FAR / 100)
    success_ratio = float(1.0 - (far / 100.0))

    # Frequency Bias: (H + F) / (H + M)
    frequency_bias = float((hits + false_alarms) / (hits + misses)) if (hits + misses) > 0 else 1.0

    return {
        "threshold": threshold,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "pod": pod,
        "far": far,
        "miss_rate": miss_rate,
        "csi": csi,
        "f1": f1,
        "success_ratio": success_ratio,
        "frequency_bias": frequency_bias,
        "total_cases": int(len(obs_v)),
    }


def evaluate_all_thresholds(
    obs: Union[np.ndarray, pd.Series, List[float]],
    pred: Union[np.ndarray, pd.Series, List[float]],
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Evaluates contingency metrics across standardized meteorological rainfall thresholds:
    0.1 mm/h (Rain Occurrence), 2.0 mm/h (Moderate Rain),
    10.0 mm/h (Heavy Rain), 20.0 mm/h (Very Heavy Rain).
    """
    if thresholds is None:
        thresholds = [0.1, 2.0, 10.0, 20.0]
    
    results = []
    for th in thresholds:
        res = calculate_contingency_metrics(obs, pred, threshold=th)
        results.append(res)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Self-test unit verification
    np.random.seed(42)
    obs = np.array([0.0, 0.0, 1.2, 5.0, 12.0, 25.0, 0.0, 0.5])
    raw = np.array([0.0, 1.5, 0.0, 3.0, 8.0, 15.0, 0.2, 1.0])
    pred = np.array([0.0, 0.1, 1.0, 4.8, 11.5, 23.0, 0.0, 0.4])

    cont = calculate_continuous_metrics(obs, pred, raw)
    print("Continuous Metrics Test:")
    for k, v in cont.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    df_th = evaluate_all_thresholds(obs, pred)
    print("\nCategorical Metrics Test:")
    print(df_th[["threshold", "hits", "misses", "false_alarms", "pod", "far", "miss_rate", "csi"]])
