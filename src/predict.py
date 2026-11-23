"""
Prediction module for Spotter Freight Rate Prediction.
Generates predictions for validation.csv and december_chart_inputs.csv,
and executes score.py validation checks.
"""
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_ARTIFACT_PATH,
    ENCODINGS_ARTIFACT_PATH,
    VALIDATION_DATA_PATH,
    VALIDATION_TEMPLATE_PATH,
    DECEMBER_INPUTS_PATH,
    OUTPUT_PREDICTIONS_PATH,
    BASE_DIR
)
from src.data_prep import clean_dataset
from src.features import extract_features


def predict_all() -> None:
    """
    Generate predictions for validation.csv and december_chart_inputs.csv.
    """
    print("Loading model and feature encoders...")
    if not MODEL_ARTIFACT_PATH.is_file() or not ENCODINGS_ARTIFACT_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifacts not found in {MODEL_ARTIFACT_PATH.parent}. Run 'python -m src.train' first."
        )

    model = joblib.load(MODEL_ARTIFACT_PATH)
    bundle = joblib.load(ENCODINGS_ARTIFACT_PATH)
    encoder = bundle["encoder"]
    coords_map = bundle["coords_map"]

    # -------------------------------------------------------------
    # 1. Validation Set Predictions (12,000 loads)
    # -------------------------------------------------------------
    print(f"\nProcessing validation dataset from {VALIDATION_DATA_PATH}...")
    raw_val = pd.read_csv(VALIDATION_DATA_PATH)
    val_clean = clean_dataset(raw_val, is_train=False, coords_map=coords_map)

    X_val = extract_features(val_clean, encoder=encoder, fit_encoder=False)
    pred_val_rpm = model.predict(X_val)
    pred_val_rate = np.clip(pred_val_rpm * val_clean["distance"].values, 1.0, None)

    # Format predictions to match template exactly
    val_template = pd.read_csv(VALIDATION_TEMPLATE_PATH)
    pred_map = dict(zip(val_clean["load_id"], pred_val_rate))
    
    val_preds_df = pd.DataFrame({
        "load_id": val_template["load_id"],
        "predicted_rate": val_template["load_id"].map(pred_map).round(2)
    })

    # Save validation_predictions.csv
    val_preds_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
    print(f"Saved {len(val_preds_df):,} validation predictions to: {OUTPUT_PREDICTIONS_PATH}")

    # -------------------------------------------------------------
    # 2. December Scenario Predictions (31 days)
    # -------------------------------------------------------------
    print(f"\nProcessing December chart scenario from {DECEMBER_INPUTS_PATH}...")
    raw_dec = pd.read_csv(DECEMBER_INPUTS_PATH)
    dec_clean = clean_dataset(raw_dec, is_train=False, coords_map=coords_map)

    X_dec = extract_features(dec_clean, encoder=encoder, fit_encoder=False)
    pred_dec_rpm = model.predict(X_dec)
    pred_dec_rate = np.clip(pred_dec_rpm * dec_clean["distance"].values, 1.0, None)

    # Fill predicted_rate into the original 7-column layout
    dec_result = raw_dec.copy()
    dec_result["predicted_rate"] = np.round(pred_dec_rate, 2)

    # Ensure required column order
    expected_dec_cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    dec_result = dec_result[expected_dec_cols]

    # Overwrite data/december_chart_inputs.csv with predictions
    dec_result.to_csv(DECEMBER_INPUTS_PATH, index=False)
    print(f"Filled 31 December predictions into: {DECEMBER_INPUTS_PATH}")
    print(f"December rates range: ${dec_result['predicted_rate'].min():.2f} - ${dec_result['predicted_rate'].max():.2f} "
          f"(Mean: ${dec_result['predicted_rate'].mean():.2f})")

    # -------------------------------------------------------------
    # 3. Execute score.py verification
    # -------------------------------------------------------------
    print("\n--- Running Official score.py Validation ---")
    score_cmd = [
        sys.executable,
        str(BASE_DIR / "score.py"),
        "--predictions", str(OUTPUT_PREDICTIONS_PATH),
        "--december-predictions", str(DECEMBER_INPUTS_PATH)
    ]
    proc = subprocess.run(score_cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"score.py failed with exit code {proc.returncode}")

    print("All validation checks passed successfully!")


if __name__ == "__main__":
    predict_all()
