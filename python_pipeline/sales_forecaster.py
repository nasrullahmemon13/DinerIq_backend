"""
DineIQ Analytics - Time-Series Sales Forecasting & Decomposition (SRS Step 13)
Implemented using Statsmodels and NumPy:
 1. 7-Day Day-of-Week Seasonal Decomposition (Trend, Seasonal, Residual)
 2. Autoregressive Integrated Moving Average (ARIMA) Sales Forecaster
 3. 30-Day Forward Revenue Projections with 95% Confidence Intervals
 4. Forecast Error Diagnostics (MAE, RMSE, MAPE)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

def build_sales_forecast_model(orders_df: pd.DataFrame) -> dict:
    """
    Aggregates daily sales, fits an ARIMA time-series model using Statsmodels,
    and produces a 30-day forward demand forecast with confidence intervals.
    """
    print("[Statsmodels] Aggregating daily time-series and fitting ARIMA model...")
    orders = orders_df.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    daily_sales = orders.groupby("order_date").agg(
        daily_revenue=("total_amount", "sum"),
        order_count=("order_id", "count")
    ).asfreq("D").ffill().fillna(0)

    decomp = seasonal_decompose(daily_sales["daily_revenue"], model="additive", period=7)
    seasonal_strength = float(np.var(decomp.seasonal) / (np.var(decomp.seasonal) + np.var(decomp.resid)))

    train_series = daily_sales["daily_revenue"].iloc[:-30]
    test_series = daily_sales["daily_revenue"].iloc[-30:]

    arima_model = ARIMA(train_series, order=(1, 1, 1)).fit()

    test_preds = arima_model.forecast(steps=len(test_series))
    mae = float(np.mean(np.abs(test_series.values - test_preds.values)))
    rmse = float(np.sqrt(np.mean((test_series.values - test_preds.values) ** 2)))
    mape = float(np.mean(np.abs((test_series.values - test_preds.values) / np.maximum(test_series.values, 1e-5))) * 100)

    full_arima = ARIMA(daily_sales["daily_revenue"], order=(1, 1, 1)).fit()
    future_forecast = full_arima.get_forecast(steps=30)
    forecast_df = future_forecast.summary_frame(alpha=0.05)         

    forecast_output = pd.DataFrame({
        "forecast_date": [str(d.date()) for d in forecast_df.index],
        "projected_revenue": forecast_df["mean"].round(2),
        "ci_lower_95": forecast_df["mean_ci_lower"].round(2),
        "ci_upper_95": forecast_df["mean_ci_upper"].round(2)
    })

    return {
        "model_type": "ARIMA(1, 1, 1) Time-Series Forecaster (Statsmodels)",
        "total_historical_days": len(daily_sales),
        "seasonal_strength_index": round(seasonal_strength, 4),
        "holdout_evaluation_30d": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape_pct": round(mape, 2)
        },
        "forecast_30d_summary": {
            "mean_projected_daily_revenue": round(float(forecast_output["projected_revenue"].mean()), 2),
            "total_projected_30d_revenue": round(float(forecast_output["projected_revenue"].sum()), 2)
        },
        "forecast_df": forecast_output
    }
