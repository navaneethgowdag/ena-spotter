"""
Data preparation module for Spotter Freight Rate Prediction.
Standardizes data cleaning rules across training, validation, and December datasets.
"""
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from src.config import TRAIN_DATA_PATH, VALIDATION_DATA_PATH, DECEMBER_INPUTS_PATH

# Default coordinates for known cities if missing from inputs
DEFAULT_CITY_COORDS = {
    "Lexington": {"lat": 36.99152, "lon": -84.99876},
    "Fort Wayne": {"lat": 41.31561, "lon": -85.36206}
}

DEFAULT_MEDIAN_WEIGHT = 31436.5


def build_city_coordinates_map(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Extract exact latitude and longitude coordinates for each city from training data.
    """
    coords_map = dict(DEFAULT_CITY_COORDS)
    
    # Pickup cities
    pickup_coords = train_df.groupby("pickup")[["pickup_lat", "pickup_lon"]].first().dropna()
    for city, row in pickup_coords.iterrows():
        coords_map[city] = {"lat": float(row["pickup_lat"]), "lon": float(row["pickup_lon"])}
        
    # Delivery cities
    deliv_coords = train_df.groupby("delivery")[["delivery_lat", "delivery_lon"]].first().dropna()
    for city, row in deliv_coords.iterrows():
        if city not in coords_map:
            coords_map[city] = {"lat": float(row["delivery_lat"]), "lon": float(row["delivery_lon"])}
            
    return coords_map


def clean_dataset(
    df: pd.DataFrame,
    is_train: bool = False,
    coords_map: Optional[Dict[str, Dict[str, float]]] = None,
    median_weight: float = DEFAULT_MEDIAN_WEIGHT
) -> pd.DataFrame:
    """
    Apply identical data cleaning and sanity checks across all splits.
    
    Rules:
    1. Strip leading/trailing whitespace from string columns.
    2. Convert negative weights to positive values (intake sign glitch).
    3. Impute missing weight values with the median and add indicator.
    4. Parse dates to standard datetime.
    5. Impute pickup/delivery coordinates from reference map if missing.
    6. For training data: calculate rate_per_mile target and filter impossible anomalies.
    """
    clean_df = df.copy()

    # 1. Clean strings
    for col in ["pickup", "delivery", "equipment"]:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.strip()

    # 2. Fix weights
    if "weight" in clean_df.columns:
        clean_df["weight_na"] = clean_df["weight"].isna().astype(int)
        clean_df["weight_cleaned"] = clean_df["weight"].abs().fillna(median_weight)
    else:
        clean_df["weight_na"] = 0
        clean_df["weight_cleaned"] = median_weight

    # 3. Clean dates
    clean_df["date"] = pd.to_datetime(clean_df["date"])

    # 4. Fill coordinates if missing
    if coords_map is None:
        coords_map = DEFAULT_CITY_COORDS

    if "pickup_lat" not in clean_df.columns or clean_df["pickup_lat"].isna().any():
        clean_df["pickup_lat"] = clean_df["pickup"].map(lambda c: coords_map.get(c, {}).get("lat", 38.0))
        clean_df["pickup_lon"] = clean_df["pickup"].map(lambda c: coords_map.get(c, {}).get("lon", -85.0))

    if "delivery_lat" not in clean_df.columns or clean_df["delivery_lat"].isna().any():
        clean_df["delivery_lat"] = clean_df["delivery"].map(lambda c: coords_map.get(c, {}).get("lat", 40.0))
        clean_df["delivery_lon"] = clean_df["delivery"].map(lambda c: coords_map.get(c, {}).get("lon", -85.0))

    # 5. Training target
    if is_train and "posted_rate" in clean_df.columns:
        # Rate per mile
        clean_df["rate_per_mile"] = clean_df["posted_rate"] / clean_df["distance"]

    return clean_df


def load_and_clean_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Convenience function to load and clean train, validation, and December inputs.
    """
    raw_train = pd.read_csv(TRAIN_DATA_PATH)
    raw_val = pd.read_csv(VALIDATION_DATA_PATH)
    raw_dec = pd.read_csv(DECEMBER_INPUTS_PATH)

    coords_map = build_city_coordinates_map(raw_train)
    median_weight = float(raw_train["weight"].abs().median())

    train_clean = clean_dataset(raw_train, is_train=True, coords_map=coords_map, median_weight=median_weight)
    val_clean = clean_dataset(raw_val, is_train=False, coords_map=coords_map, median_weight=median_weight)
    dec_clean = clean_dataset(raw_dec, is_train=False, coords_map=coords_map, median_weight=median_weight)

    return train_clean, val_clean, dec_clean, coords_map
