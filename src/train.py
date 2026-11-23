"""
Training module for Spotter Freight Rate Prediction.
Performs time-series validation and trains the production LightGBM model.
"""
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    MODELS_DIR,
    MODEL_ARTIFACT_PATH,
    ENCODINGS_ARTIFACT_PATH,
    LGBM_PARAMS,
    RANDOM_SEED
)
from src.data_prep import load_and_clean_data
from src.features import TargetEncoder, extract_features, FEATURE_COLUMNS, CATEGORICAL_COLUMNS


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate MAE, RMSE, MAPE, SMAPE, and R2 on dollar rates.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 1.0, None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "SMAPE": smape,
        "R2": r2
    }


def run_time_series_cv(train_df: pd.DataFrame, num_folds: int = 4) -> None:
    """
    Evaluate expanding-window / forward-chaining cross-validation across months.
    """
    print("\n--- Running Expanding-Window Time-Series Cross-Validation ---")
    
    # Months 5, 6, 7, 8 as validation folds
    val_months = [5, 6, 7, 8]
    fold_metrics = []

    for fold, v_month in enumerate(val_months, 1):
        tr_mask = train_df["month"] < v_month
        val_mask = train_df["month"] == v_month

        tr_data = train_df[tr_mask].copy().reset_index(drop=True)
        val_data = train_df[val_mask].copy().reset_index(drop=True)

        encoder = TargetEncoder(smoothing=5.0)
        X_tr = extract_features(tr_data, encoder=encoder, fit_encoder=True)
        X_val = extract_features(val_data, encoder=encoder, fit_encoder=False)

        dtrain = lgb.Dataset(X_tr, label=tr_data["rate_per_mile"], categorical_feature=CATEGORICAL_COLUMNS)
        dval = lgb.Dataset(X_val, label=val_data["rate_per_mile"], reference=dtrain, categorical_feature=CATEGORICAL_COLUMNS)

        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            num_boost_round=800,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        pred_rpm = model.predict(X_val)
        pred_rate = pred_rpm * val_data["distance"].values

        m = calculate_metrics(val_data["posted_rate"].values, pred_rate)
        fold_metrics.append(m)
        print(f"Fold {fold} (Train months 1-{v_month-1} -> Validate month {v_month}): "
              f"MAE=${m['MAE']:.2f}, RMSE=${m['RMSE']:.2f}, MAPE={m['MAPE']:.2f}%, R2={m['R2']:.4f}")

    cv_df = pd.DataFrame(fold_metrics)
    print(f"\nTime-Series CV Average: MAE=${cv_df['MAE'].mean():.2f}, RMSE=${cv_df['RMSE'].mean():.2f}, "
          f"MAPE={cv_df['MAPE'].mean():.2f}%, R2={cv_df['R2'].mean():.4f}")
    print("------------------------------------------------------------\n")


def train_production_model() -> None:
    """
    Train final LightGBM model on all 48,000 development loads and persist artifacts.
    """
    print("Loading and preparing training data...")
    train_clean, _, _, coords_map = load_and_clean_data()
    train_clean["month"] = train_clean["date"].dt.month

    # 1. Run rigorous expanding-window cross validation
    run_time_series_cv(train_clean)

    # 2. Fit production encoder on ALL 48,000 loads
    print("Fitting production TargetEncoder on all 48,000 training loads...")
    encoder = TargetEncoder(smoothing=5.0)
    X_train = extract_features(train_clean, encoder=encoder, fit_encoder=True)
    y_train_rpm = train_clean["rate_per_mile"].values

    print(f"Training LightGBM on {len(X_train):,} loads with {len(FEATURE_COLUMNS)} features...")
    dtrain_full = lgb.Dataset(
        X_train,
        label=y_train_rpm,
        categorical_feature=CATEGORICAL_COLUMNS
    )

    # Final model fits on full dataset with optimized boost rounds
    final_model = lgb.train(
        LGBM_PARAMS,
        dtrain_full,
        num_boost_round=900
    )

    # 3. Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(final_model, MODEL_ARTIFACT_PATH)
    
    # Save encoder and coordinates metadata
    encoder_bundle = {
        "encoder": encoder,
        "coords_map": coords_map,
        "features": FEATURE_COLUMNS,
        "categoricals": CATEGORICAL_COLUMNS
    }
    joblib.dump(encoder_bundle, ENCODINGS_ARTIFACT_PATH)

    print(f"Model saved to: {MODEL_ARTIFACT_PATH}")
    print(f"Feature encoders saved to: {ENCODINGS_ARTIFACT_PATH}")
    print("Training complete successfully!")


if __name__ == "__main__":
    train_production_model()
