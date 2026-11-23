"""
Feature engineering module for Spotter Freight Rate Prediction.
Implements out-of-fold target encoding, spatial transformations, and calendar seasonality.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.config import US_HOLIDAYS_2025


FEATURE_COLUMNS = [
    "distance",
    "log_distance",
    "weight_cleaned",
    "log_weight",
    "weight_na",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "te_lane_rpm",
    "te_eq_rpm",
    "te_pickup_rpm",
    "te_delivery_rpm",
    "dayofweek",
    "day",
    "month",
    "is_weekend",
    "days_to_holiday",
    "is_holiday_week",
    "dow_sin",
    "dow_cos",
    "pickup",
    "delivery",
    "equipment"
]

CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]


class TargetEncoder:
    """
    Bayesian Smoothed Target Encoder for Lanes and Categories.
    Derives rate-per-mile benchmarks strictly from training data with fallbacks.
    """
    def __init__(self, smoothing: float = 5.0):
        self.smoothing = smoothing
        self.global_rpm: float = 2.15
        self.eq_rpm: Dict[str, float] = {}
        self.pickup_rpm: Dict[str, float] = {}
        self.delivery_rpm: Dict[str, float] = {}
        self.lane_rpm: Dict[Tuple[str, str], float] = {}

    def fit(self, df: pd.DataFrame) -> "TargetEncoder":
        """
        Compute empirical hierarchical priors from training data.
        """
        rpm = df["rate_per_mile"]
        self.global_rpm = float(rpm.median())

        # Equipment median
        self.eq_rpm = df.groupby("equipment")["rate_per_mile"].median().to_dict()

        # Pickup & delivery medians
        self.pickup_rpm = df.groupby("pickup")["rate_per_mile"].median().to_dict()
        self.delivery_rpm = df.groupby("delivery")["rate_per_mile"].median().to_dict()

        # Lane Bayesian shrinkage: (n * median + smoothing * prior) / (n + smoothing)
        lane_cnt = df.groupby(["pickup", "delivery"])["rate_per_mile"].count()
        lane_med = df.groupby(["pickup", "delivery"])["rate_per_mile"].median()
        shrunk = (lane_cnt * lane_med + self.smoothing * self.global_rpm) / (lane_cnt + self.smoothing)
        self.lane_rpm = shrunk.to_dict()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply target encodings with hierarchical fallback for unseen categories.
        """
        res = pd.DataFrame(index=df.index)

        # Equipment encoding with global fallback
        res["te_eq_rpm"] = df["equipment"].map(self.eq_rpm).fillna(self.global_rpm)

        # Pickup and delivery encodings with equipment fallback
        res["te_pickup_rpm"] = df["pickup"].map(self.pickup_rpm).fillna(res["te_eq_rpm"])
        res["te_delivery_rpm"] = df["delivery"].map(self.delivery_rpm).fillna(res["te_eq_rpm"])

        # Lane encoding: if unseen lane, blend pickup and delivery estimates
        lane_keys = list(zip(df["pickup"], df["delivery"]))
        res["te_lane_rpm"] = [self.lane_rpm.get(k, np.nan) for k in lane_keys]
        res["te_lane_rpm"] = res["te_lane_rpm"].fillna(
            0.5 * (res["te_pickup_rpm"] + res["te_delivery_rpm"])
        )

        return res


def compute_holiday_features(dt_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized computation of distance (days) to nearest US Federal holiday and holiday week flag.
    """
    dt_arr = dt_series.dt.date.values.astype("datetime64[D]")
    diffs = np.abs(dt_arr[:, None] - US_HOLIDAYS_2025[None, :]).astype(int)
    min_dist = np.min(diffs, axis=1)
    is_holiday_week = (min_dist <= 3).astype(int)
    return min_dist, is_holiday_week


def extract_features(
    df: pd.DataFrame,
    encoder: TargetEncoder,
    fit_encoder: bool = False
) -> pd.DataFrame:
    """
    Transform raw cleaned dataframe into modeling feature matrix.
    """
    features = pd.DataFrame(index=df.index)

    # 1. Physical shipment features
    features["distance"] = df["distance"].values
    features["log_distance"] = np.log(df["distance"].values)
    features["weight_cleaned"] = df["weight_cleaned"].values
    features["log_weight"] = np.log(df["weight_cleaned"].values)
    features["weight_na"] = df["weight_na"].values

    # 2. Coordinates
    features["pickup_lat"] = df["pickup_lat"].values
    features["pickup_lon"] = df["pickup_lon"].values
    features["delivery_lat"] = df["delivery_lat"].values
    features["delivery_lon"] = df["delivery_lon"].values

    # 3. Target encodings
    if fit_encoder:
        encoder.fit(df)
    encoded_df = encoder.transform(df)
    for col in encoded_df.columns:
        features[col] = encoded_df[col].values

    # 4. Temporal calendar & seasonality features
    features["dayofweek"] = df["date"].dt.dayofweek.values
    features["day"] = df["date"].dt.day.values
    features["month"] = df["date"].dt.month.values
    features["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int).values

    days_to_holiday, is_holiday_week = compute_holiday_features(df["date"])
    features["days_to_holiday"] = days_to_holiday
    features["is_holiday_week"] = is_holiday_week

    # Cyclical day of week encoding (period = 7)
    features["dow_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7.0)

    # 5. Categoricals for tree models
    for cat in CATEGORICAL_COLUMNS:
        features[cat] = df[cat].astype("category").values

    return features[FEATURE_COLUMNS]
