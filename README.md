# Spotter: Freight Rate Prediction Challenge

End-to-end Machine Learning pipeline to predict spot freight load rates from shipment specifications (origin/destination cities, haul distance, equipment type, weight, and dispatch dates).

This solution is developed for the Spotter Machine Learning Engineer take-home assessment, adhering strictly to the official scorer requirements and production-grade software engineering standards.

---

## Quickstart & Reproduction

### 1. Environment Setup
Install required dependencies (compatible with Python 3.10+):

```bash
python -m pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution
Run model training and prediction generation with a single command:

```bash
python -m src.train && python -m src.predict
```

This command will:
1. Clean and validate development data from `data/train_test.csv`.
2. Run forward-chaining (expanding-window) cross-validation across months.
3. Fit the production LightGBM model on all 48,000 loads and save artifacts to `models/`.
4. Generate 12,000 predictions for `validation_predictions.csv`.
5. Populate 31 daily predictions in `data/december_chart_inputs.csv`.
6. Automatically invoke `score.py` to validate outputs and generate `scorer_results/candidate_december.png`.

### 3. Run the Official Scorer
To manually verify the generated predictions against official validation rules:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Expected output:
```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

---

## Validation Strategy & Benchmark Results

### Why Time-Based Splitting Matters
Freight spot rates fluctuate under macroeconomic freight cycles, diesel fuel surcharges, and localized lane tightness. A **random K-fold split creates severe temporal leakage** because loads dispatched on the same lane within days of each other share near-identical market conditions, drastically overstating real-world performance.

To accurately simulate production deployment, we implemented a **two-tier time-based validation strategy**:
- **Held-Out Test Period (Sep 1 – Oct 31, 2025):** The final two months of development data (9,523 loads) were held out as a blind test set, mirroring the task of forecasting unseen November and December loads.
- **Forward-Chaining Cross-Validation:** Inside the January–August training partition (38,477 loads), we ran 4 sequential expanding-window folds (Fold 1: train Jan–Apr -> validate May; Fold 2: train Jan–May -> validate Jun; Fold 3: train Jan–Jun -> validate Jul; Fold 4: train Jan–Jul -> validate Aug). Target encodings were recomputed strictly within each training fold.

### Performance Benchmark (Held-Out Sep–Oct 2025 Test Partition)

| Model Architecture | MAE ($) | RMSE ($) | MAPE (%) | SMAPE (%) | R² Score | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1: Global Median RPM** | $256.95 | $684.25 | 11.63% | 11.02% | 0.7990 | Naive global rate-per-mile heuristic |
| **Baseline 2: Per-Lane Median RPM** | $196.95 | $654.16 | 8.58% | 7.82% | 0.8163 | Historical lane median with fallback |
| **Ridge Linear Regression** | $148.83 | $634.25 | 6.58% | 5.47% | 0.8273 | L2 regularized linear model on log(rate) |
| **Random Forest Regressor** | $148.84 | $660.08 | 6.47% | 5.37% | 0.8129 | Non-linear bagged decision trees |
| **LightGBM Gradient Boosting (Winner)** | **$106.16** | **$632.66** | **4.62%** | **3.78%** | **0.8281** | L1 loss on rate-per-mile with cyclic calendar |

*The LightGBM model cuts baseline prediction error by **46%**, achieving an MAE of $106.16 and MAPE below 4.7%.*

---

## Key Data Quality Findings & Architectural Decisions

1. **Negative Weights (Intake Sign Glitch):**
   - *Finding:* 292 rows in train and 145 in validation had negative weights (e.g., `-36,559 lbs`). The absolute values follow the exact statistical distribution of valid positive freight weights.
   - *Remediation:* Fixed across all datasets via `np.abs(weight)`.
2. **Missing Attribute Handling:**
   - *Finding:* ~1% of rows lacked weight values.
   - *Remediation:* Imputed with the development median (31,436.5 lbs) and tagged with a binary `weight_na` indicator flag.
3. **Unseen Cities & Lanes in Validation:**
   - *Finding:* Validation introduced 8 new cities (e.g., Chicago, San Diego, Charlotte, Laredo) and 736 previously unseen lanes (17.5% of validation loads).
   - *Remediation:* Constructed a **hierarchical Bayesian smoothed target encoder** that falls back from lane-level priors to origin/destination regional priors and equipment medians, supplemented by continuous geographic coordinates (latitude and longitude).
4. **Target Formulation (Rate-per-Mile):**
   - *Decision:* Rather than modeling raw dollar rate directly, the model predicts **rate-per-mile ($/mi)** and converts back via `predicted_rate = predicted_rpm * distance`.
   - *Benefit:* Naturally enforces strictly positive predictions, captures haul-length tapering economics, and eliminates negative rate predictions.
5. **Leakage Audit (`quote_signal` & `market_index`):**
   - *Finding:* `quote_signal` is present in `train_test.csv` and `validation.csv`, but **omitted** from `december_chart_inputs.csv`.
   - *Decision:* Rigorous time-series benchmarking revealed that training purely on fundamental shipment attributes and calendar features outperformed the model containing `quote_signal` (MAE **$106.16** vs $108.33), while guaranteeing zero target leakage and 100% architectural parity across all inference targets.
6. **December 2025 Extrapolation:**
   - *Finding:* The fixed scenario (Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb) required 31 daily forecasts beyond the training range (Jan–Oct).
   - *Remediation:* Leveraged cyclical calendar features (sine/cosine day-of-week, proximity to US federal holidays, and day-of-month). The resulting curve matches historical ground truth (train mean: $823.31; December predictions: $814.77 – $848.25, mean: $834.20) with authentic Sunday weekend dips, midweek shipper surges, and pre-Christmas capacity tightening.

---

## Repository Structure

```text
ena-spotter/
├── README.md                                 # Setup, run commands, methodology & results summary
├── requirements.txt                          # Pinned dependency ranges
├── score.py                                  # Official Spotter validation script (unchanged)
├── validation_predictions.csv                # 12,000 predictions (load_id,predicted_rate)
├── data/
│   ├── train_test.csv                        # 48,000 labeled development loads (Jan-Oct 2025)
│   ├── validation.csv                        # 12,000 unlabeled evaluation loads (Nov-Dec 2025)
│   ├── validation_predictions_template.csv   # Template file for validation IDs
│   └── december_chart_inputs.csv             # Completed 31-day forecast inputs
├── docs/
│   └── Freight_Rate_ML_Assessment.pdf        # Official assessment specification
├── models/
│   ├── rate_model.joblib                     # Serialized LightGBM production model
│   └── feature_encoders.joblib               # Fitted TargetEncoder & spatial metadata
├── notebooks/
│   └── 01_eda.ipynb                          # Exploratory data analysis notebook
├── reports/
│   ├── Freight_Rate_Report.pdf               # Comprehensive 4-page submission report
│   ├── loom_script.md                        # 2.5-minute presentation script
│   ├── model_comparison_metrics.csv          # Benchmark evaluation metrics
│   └── figures/                              # Publication figures (EDA, benchmarks, importance)
└── src/
    ├── __init__.py                           # Package marker
    ├── config.py                             # Paths, constants, holidays, and hyperparameters
    ├── data_prep.py                          # Unified cleaning and sanity validation
    ├── features.py                           # TargetEncoder, spatial lookups, calendar signals
    ├── train.py                              # Expanding-window CV and production training
    └── predict.py                            # End-to-end inference and score.py verification
```

---

## Submission Artifacts Checklist

- [x] **GitHub Repository:** Accessible repository with complete commit history, modular code, and documentation.
- [x] **`validation_predictions.csv`:** Exactly 12,000 rows with columns `load_id,predicted_rate`, strictly positive rates, matching template IDs.
- [x] **`data/december_chart_inputs.csv`:** Completed with 31 predicted rates, retaining original 7 columns and row ordering.
- [x] **`scorer_results/candidate_december.png`:** Verified and generated by `score.py` with zero errors.
- [x] **`reports/Freight_Rate_Report.pdf`:** Comprehensive report covering data quality, validation strategy, benchmark metrics table, feature importance, and December forecast interpretation.
- [x] **`reports/loom_script.md`:** 2.5 to 3-minute talking points script for video walkthrough.
