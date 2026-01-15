
# GA4 client + modeling libs
pip install google-analytics-data pandas numpy matplotlib prophet pmdarima statsmodels google-auth
export GOOGLE_APPLICATION_CREDENTIALS=/secure/path/ga4-sa.json


# -*- coding: utf-8 -*-
"""
Generic GA4 → Time Series Forecasting
Author: You
Notes:
 - Pulls GA4 data via Analytics Data API
 - Builds daily series, cleans, backtests, and fits Prophet & SARIMAX
 - Picks champion by MAPE on rolling-origin backtest
"""

import os
from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Modeling
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# GA4
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Metric, Dimension
from google.oauth2 import service_account


# -----------------------------
# 1) GA4 data pull
# -----------------------------
def ga4_client(credentials_path: str = None):
    """
    Returns an authenticated GA4 client using ADC or explicit service-account key.
    """
    if credentials_path and os.path.exists(credentials_path):
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        return BetaAnalyticsDataClient(credentials=creds)
    # Fallback to ADC
    return BetaAnalyticsDataClient()

def fetch_ga4_timeseries(
    property_id: str,
    start_date: str = "2023-01-01",
    end_date: str = None,
    metrics: list = None,
    dimensions: list = None,
    credentials_path: str = None
) -> pd.DataFrame:
    """
    Runs a GA4 report and returns a tidy DataFrame.
    `metrics` should be GA4 metric API names (e.g., 'sessions', 'totalUsers').
    `dimensions` can include 'date' or 'dateHour' for time granularity.
    """
    if metrics is None:
        metrics = ["sessions"]
    if dimensions is None:
        dimensions = ["date"]

    if end_date is None:
        end_date = date.today().isoformat()

    client = ga4_client(credentials_path)

    request = RunReportRequest(
        property=f"properties/{property_id}",
        metrics=[Metric(name=m) for m in metrics],
        dimensions=[Dimension(name=d) for d in dimensions],
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
    )

    response = client.run_report(request)
    # Build DataFrame
    headers = [hdr.name for hdr in response.dimension_headers] + \
              [hdr.name for hdr in response.metric_headers]

    rows = []
    for row in response.rows:
        dim_vals = [d.value for d in row.dimension_values]
        met_vals = [float(m.value.replace(',', '')) for m in row.metric_values]
        rows.append(dim_vals + met_vals)

    df = pd.DataFrame(rows, columns=headers)

    # Parse date or dateHour into pandas datetime
    if "dateHour" in df.columns:
        # GA4 returns YYYYMMDDHH
        df["dateHour"] = pd.to_datetime(df["dateHour"], format="%Y%m%d%H")
        df.rename(columns={"dateHour": "timestamp"}, inplace=True)
        df.set_index("timestamp", inplace=True)
    elif "date" in df.columns:
        # GA4 returns YYYYMMDD
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df.rename(columns={"date": "timestamp"}, inplace=True)
        df.set_index("timestamp", inplace=True)

    # Convert metrics to numeric
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    return df.sort_index()


# -----------------------------
# 2) TS utilities
# -----------------------------
def build_daily_series(df: pd.DataFrame, target_col: str, agg: str = "sum") -> pd.DataFrame:
    """
    Ensures daily frequency and fills small gaps.
    """
    daily = df[[target_col]].resample("D").agg(agg)
    # Fill tiny gaps with forward-fill then 0 as last resort
    daily[target_col] = daily[target_col].interpolate("time").fillna(method="bfill").fillna(0.0)
    return daily

def train_test_split_ts(df: pd.DataFrame, test_days: int = 28):
    """
    Split at the end for holdout evaluation.
    """
    split_point = df.index.max() - timedelta(days=test_days)
    train = df.loc[:split_point]
    test = df.loc[split_point + timedelta(days=1):]
    return train, test

def mape(y_true, y_pred):
