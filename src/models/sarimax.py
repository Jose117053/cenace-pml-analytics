"""SARIMAX forecasting pipeline for day-ahead electricity prices.

Implements a seasonal ARIMA model with exogenous calendar regressors:
- Target (y_t): Hourly Local Marginal Price (PML, $/MWh).
- Seasonality (s=24): Daily diurnal cycle captured via seasonal AR and differencing terms.
- Exogenous Regressors (X_t):
  1. 6 Day-of-Week dummy variables (Monday omitted as reference base).
  2. Continuous lag of 168 hours (pml_lag_168, known before 10:00 AM on day D).
"""

from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


def prepare_sarimax_features(
    df: pd.DataFrame,
    node: str,
    target_col: str = "pml",
    node_col: str = "clv_nodo",
    time_col: str = "timestamp",
) -> Tuple[pd.Series, pd.DataFrame]:
    """Prepare hourly target series and exogenous regressors for a specific node."""
    node_df = df[df[node_col] == node].copy()
    node_df[time_col] = pd.to_datetime(node_df[time_col])
    node_df = node_df.sort_values(time_col).reset_index(drop=True)
    node_df.set_index(time_col, inplace=True)
    node_df.index = pd.DatetimeIndex(node_df.index, freq="h")

    # Weekly lag (168 hours = 7 days prior, strictly known before 10:00 AM on day D)
    node_df["pml_lag_168"] = node_df[target_col].shift(168)

    # 6 day-of-week dummies (drop_first=True avoids the dummy variable trap)
    dow_dummies = pd.get_dummies(
        node_df.index.dayofweek,
        prefix="dow",
        drop_first=True,
        dtype=float,
    )
    dow_dummies.index = node_df.index

    exog_df = pd.concat([dow_dummies, node_df[["pml_lag_168"]]], axis=1)
    return node_df[target_col], exog_df


def fit_sarimax_model(
    train_y: pd.Series,
    train_exog: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 24),
) -> SARIMAXResults:
    """Fit SARIMAX model on training partition using maximum likelihood."""
    model = SARIMAX(
        train_y,
        exog=train_exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        freq="h",
    )
    return model.fit(disp=False)


def rolling_day_ahead_forecast(
    fitted_model: SARIMAXResults,
    train_y: pd.Series,
    train_exog: pd.DataFrame,
    val_y: pd.Series,
    val_exog: pd.DataFrame,
) -> pd.Series:
    """Generate 24-hour day-ahead rolling forecasts over the validation partition.

    Simulates real-world MDA operation: each day D, the Kalman filter state is updated
    with observed prices up to D 23:00, and a dynamic 24-step forecast for day D+1 is emitted.
    """
    full_y = pd.concat([train_y, val_y])
    full_y.index = pd.DatetimeIndex(full_y.index, freq="h")

    full_exog = pd.concat([train_exog, val_exog])
    full_exog.index = pd.DatetimeIndex(full_exog.index, freq="h")

    # Filter full series through Kalman state-space using fixed parameters learned on Train
    extended_results = fitted_model.apply(full_y, exog=full_exog)

    val_dates = pd.date_range(
        val_y.index.min().floor("D"),
        val_y.index.max().floor("D"),
        freq="D",
    )

    daily_forecasts = []
    for day in val_dates:
        start_hour = day
        end_hour = day + pd.Timedelta(hours=23)

        pred_24 = extended_results.get_prediction(
            start=start_hour,
            end=end_hour,
            dynamic=start_hour,
        ).predicted_mean
        daily_forecasts.append(pred_24)

    forecasts = pd.concat(daily_forecasts)
    forecasts.index = pd.DatetimeIndex(forecasts.index, freq="h")
    return forecasts
