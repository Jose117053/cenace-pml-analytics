"""XGBoost Global Multi-Node forecasting pipeline for day-ahead electricity prices.

Implements a non-linear tabular gradient boosted tree model trained across all 9 nodes:
- Target (y_t): Hourly Local Marginal Price (PML, $/MWh).
- Autoregressive Price Lags:
  1. pml_lag_24: Price at the same hour yesterday (strictly known on day D).
  2. pml_lag_48: Price at the same hour 2 days prior (day D-1).
  3. pml_lag_168: Price at the same hour 7 days prior (same weekday of previous week).
- Rolling Volatility & Regime Statistics:
  1. pml_roll_mean_24: 24-hour rolling mean computed on lag_24 (zero leakage for D+1).
  2. pml_roll_std_24: 24-hour rolling standard deviation computed on lag_24.
- Calendar & Spatial Features:
  1. hora (0-23): Hour of day capturing diurnal valley/peak cycles.
  2. dayofweek (0-6): Day of week capturing weekly cycles.
  3. is_weekend (0/1): Binary indicator for Saturday/Sunday.
  4. clv_nodo: Categorical node identifier enabling unified Global Panel training.
"""

from typing import List
import pandas as pd
import xgboost as xgb

FEATURE_COLS: List[str] = [
    "pml_lag_24",
    "pml_lag_48",
    "pml_lag_168",
    "pml_roll_mean_24",
    "pml_roll_std_24",
    "hora",
    "dayofweek",
    "is_weekend",
    "clv_nodo",
]


def build_xgboost_features(
    df: pd.DataFrame,
    target_col: str = "pml",
    node_col: str = "clv_nodo",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Engineer leak-free multi-lag, rolling memory, and calendar features grouped by node.
    
    Args:
        df: Input DataFrame containing clean PML records across all nodes.
        target_col: Name of price target column.
        node_col: Name of node identifier column.
        time_col: Name of datetime timestamp column.
        
    Returns:
        pd.DataFrame: Sorted DataFrame with all engineered feature columns.
    """
    df_feat = df.copy()
    df_feat[time_col] = pd.to_datetime(df_feat[time_col])
    df_feat = df_feat.sort_values([node_col, time_col]).reset_index(drop=True)

    # Autoregressive price lags (strictly known before 10:00 AM on day D)
    df_feat["pml_lag_24"] = df_feat.groupby(node_col)[target_col].shift(24)
    df_feat["pml_lag_48"] = df_feat.groupby(node_col)[target_col].shift(48)
    df_feat["pml_lag_168"] = df_feat.groupby(node_col)[target_col].shift(168)

    # Rolling memory window computed on lag_24 to guarantee zero leakage for all 24 hours of D+1
    df_feat["pml_roll_mean_24"] = df_feat.groupby(node_col)["pml_lag_24"].transform(
        lambda s: s.rolling(24, min_periods=24).mean()
    )
    df_feat["pml_roll_std_24"] = df_feat.groupby(node_col)["pml_lag_24"].transform(
        lambda s: s.rolling(24, min_periods=24).std()
    )

    # Calendar features
    df_feat["hora"] = df_feat[time_col].dt.hour
    df_feat["dayofweek"] = df_feat[time_col].dt.dayofweek
    df_feat["is_weekend"] = df_feat["dayofweek"].isin([5, 6]).astype(int)

    # Spatial categorical feature
    df_feat[node_col] = df_feat[node_col].astype("category")

    return df_feat


def train_global_xgboost(
    train_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    target_col: str = "pml",
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> xgb.XGBRegressor:
    """Train unified Global Multi-Node XGBoost Regressor optimizing MAE (L1 loss).
    
    Args:
        train_df: Training DataFrame containing features and target.
        feature_cols: List of feature column names.
        target_col: Name of target price column.
        n_estimators: Number of gradient boosted trees.
        learning_rate: Step size shrinkage parameter.
        max_depth: Maximum tree depth for base learners.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns when constructing each tree.
        random_state: Reproducible random seed.
        
    Returns:
        xgb.XGBRegressor: Fitted model object.
    """
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:absoluteerror",
        random_state=random_state,
        enable_categorical=True,
        n_jobs=-1,
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    return model


def predict_global_xgboost(
    model: xgb.XGBRegressor,
    val_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
) -> pd.Series:
    """Generate day-ahead predictions using the trained Global XGBoost model.
    
    Args:
        model: Trained XGBRegressor.
        val_df: Evaluation DataFrame containing feature columns.
        feature_cols: List of feature column names.
        
    Returns:
        pd.Series: Forecasted PML values indexed to match val_df.
    """
    preds = model.predict(val_df[feature_cols])
    return pd.Series(preds, index=val_df.index, name="pml_pred_xgb")
