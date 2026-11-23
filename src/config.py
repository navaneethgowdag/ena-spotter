"""
Configuration module for Spotter Freight Rate Prediction.
Defines project paths, model hyperparameters, and global constants.
"""
from pathlib import Path
import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SCORER_DIR = BASE_DIR / "scorer_results"

# Data file paths
TRAIN_DATA_PATH = DATA_DIR / "train_test.csv"
VALIDATION_DATA_PATH = DATA_DIR / "validation.csv"
VALIDATION_TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_INPUTS_PATH = DATA_DIR / "december_chart_inputs.csv"

# Output paths
OUTPUT_PREDICTIONS_PATH = BASE_DIR / "validation_predictions.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "rate_model.joblib"
ENCODINGS_ARTIFACT_PATH = MODELS_DIR / "feature_encoders.joblib"

# Fixed December scenario parameters (from assessment & score.py)
FIXED_PICKUP = "Lexington"
FIXED_DELIVERY = "Fort Wayne"
FIXED_DISTANCE = 360.0
FIXED_EQUIPMENT = "Dry Van"
FIXED_WEIGHT = 32_000.0

# 2025 US Federal Holidays
US_HOLIDAYS_2025 = np.array([
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # Martin Luther King Jr. Day
    "2025-02-17",  # Washington's Birthday (Presidents Day)
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-10-13",  # Columbus Day
    "2025-11-11",  # Veterans Day
    "2025-11-27",  # Thanksgiving Day
    "2025-12-25",  # Christmas Day
], dtype="datetime64[D]")

# Model hyperparameters
LGBM_PARAMS = {
    "objective": "regression_l1",  # MAE optimization
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 25,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": -1
}
