"""Evaluation metrics for electricity price forecasting.

Primary evaluation relies on Mean Absolute Error (MAE),
complemented by Root Mean Squared Error (RMSE) to capture extreme spike penalties.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE).
    
    Args:
        y_true: Array of observed actual values.
        y_pred: Array of predicted forecast values.
        
    Returns:
        float: Mean absolute error in $/MWh.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_t - y_p)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def evaluate_forecasts_by_node(
    df: pd.DataFrame,
    y_true_col: str,
    pred_cols: Dict[str, str],
    node_col: str = "clv_nodo"
) -> pd.DataFrame:
    """Evaluate multiple prediction models per node and compute global average.
    
    Args:
        df: DataFrame containing actual and forecast columns.
        y_true_col: Name of column containing ground-truth prices.
        pred_cols: Mapping of model label to prediction column name.
                   e.g., {"Daily Persistence": "y_hat_daily", ...}
        node_col: Name of column identifying electric nodes.
        
    Returns:
        pd.DataFrame: Table with MAE and RMSE per node, plus OVERALL_AVERAGE row.
    """
    records = []
    nodes = sorted(df[node_col].unique())
    
    for node in nodes:
        node_df = df[df[node_col] == node]
        y_true = node_df[y_true_col].values
        
        row_dict = {node_col: node}
        for model_name, pred_col in pred_cols.items():
            y_pred = node_df[pred_col].values
            row_dict[f"{model_name}_MAE"] = compute_mae(y_true, y_pred)
            row_dict[f"{model_name}_RMSE"] = compute_rmse(y_true, y_pred)
        records.append(row_dict)
        
    result_df = pd.DataFrame(records)
    
    # Compute overall average across nodes
    avg_dict = {node_col: "OVERALL_AVERAGE"}
    for col in result_df.columns:
        if col != node_col:
            avg_dict[col] = float(result_df[col].mean())
            
    avg_df = pd.DataFrame([avg_dict])
    return pd.concat([result_df, avg_df], ignore_index=True)
