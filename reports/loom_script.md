# Spotter Machine Learning Engineer Assessment — Loom Presentation Script

**Presenter:** Navaneeth Gowda  
**Target Duration:** 2.5 to 3.0 minutes  
**Format:** Screen share with camera on  
**Artifacts to show on screen:**
1. `reports/Freight_Rate_Report.pdf` (Executive Summary, Table 1 Benchmark, Feature Importance)
2. VS Code / Terminal showing `src/` modular structure (`src/features.py`, `src/train.py`, `src/predict.py`)
3. `scorer_results/candidate_december.png` and terminal output of `score.py`

---

## 0:00 – 0:30 | Introduction & Challenge Objective

**[On Screen: `reports/Freight_Rate_Report.pdf` title page or GitHub repository overview]**

> "Hi everyone, I'm Navaneeth, and today I'm walking through my solution for the Spotter Freight Rate Prediction take-home assessment.
> 
> The objective of this challenge is to predict the posted dollar freight rate for spot loads based on load attributes like origin, destination, haul distance, equipment type, weight, and dispatch date. We were tasked with training on 48,000 historical loads from January through October 2025, predicting 12,000 loads in validation for November and December, and producing a 31-day forecast for a fixed scenario in December 2025 on the Lexington to Fort Wayne lane."

---

## 0:30 – 1:05 | Key EDA Findings & Data-Quality Anomaly Remediation

**[On Screen: `reports/Freight_Rate_Report.pdf` Section 1: Data Exploration & Quality Audit]**

> "During exploratory data analysis, I audited the data schemas and uncovered several crucial findings:
> 
> 1. **Negative Weights:** 292 rows in train and 145 in validation had negative weights down to -47,500 lbs. An absolute value check revealed that `abs(weight)` follows the exact statistical distribution of valid positive weights—identifying this as an upstream sign ingestion glitch. I resolved it uniformly with `np.abs(weight)`.
> 2. **Missing Attributes:** Roughly 1% of loads had missing weights; these were imputed with the dataset median alongside an explicit indicator flag.
> 3. **Validation Category Shift:** While equipment types remained identical, the validation set introduced 8 new cities and 736 previously unseen lanes (17.5% of validation loads). A standard categorical one-hot encoding would fail here. Instead, I designed a hierarchical Bayesian smoothed target encoder that shrinks sparse lanes towards regional and global priors, augmented by continuous latitude and longitude coordinates.
> 4. **Signal Audit (`quote_signal` & `market_index`):** While `quote_signal` appeared in train and validation, it was completely absent from the December scenario inputs. Rigorous time-series benchmarking revealed that training purely on fundamental shipment attributes and cyclical calendar features actually outperformed the model with `quote_signal` (MAE $106.16 vs $108.33), while guaranteeing zero target leakage and 100% architectural parity across all datasets."

---

## 1:05 – 1:45 | Validation Strategy: Why Time-Based Splitting Matters

**[On Screen: Table 1: Model Benchmarking & Performance Comparison in Report]**

> "Validation strategy was a core focus of this assessment.
> 
> In freight logistics, random K-fold splits are fundamentally flawed: loads occurring on the same lane within days of each other share the same macroeconomic tightness and fuel surcharges. Random splits cause severe temporal leakage and drastically overestimate performance.
> 
> To ensure realistic evaluation, I implemented a dual-tier time-based framework:
> - **Held-Out Test Set:** Quarantined the final two months of development data (September and October, 9,523 loads) as a blind test set.
> - **Expanding-Window Cross-Validation:** Inside the January-August training portion, I evaluated models across 4 sequential forward-chaining folds.
> 
> Looking at our benchmark table:
> - The naive **Global Median RPM baseline** scored an MAE of $256.95.
> - The **Per-Lane Median baseline** brought MAE down to $196.95.
> - Regularized **Ridge Regression** and **Random Forest** achieved an MAE of ~$148.83.
> - Our production **LightGBM Gradient Boosting model** won decisively, achieving an **MAE of $106.16** and **R² of 0.8281**—cutting the baseline error nearly in half."

---

## 1:45 – 2:25 | Code Walkthrough (The 4 Most Important Modules)

**[On Screen: Switch to VS Code showing `src/` directory]**

> "Let's quickly walk through the four key code modules in `src/`:
> 
> 1. **`src/data_prep.py`**: Enforces strict idempotency. Every cleaning rule—weight sign correction, median imputation, string normalization, and coordinate mapping—is applied identically across train, validation, and December inputs.
> 2. **`src/features.py`**: Houses our `TargetEncoder`. It applies empirical Bayesian shrinkage on historical lane rate-per-mile with fallback hierarchies for unseen lanes. Crucially, we model **rate-per-mile ($/mi)** directly rather than raw dollar rate. Multiplying by distance (`predicted_rate = predicted_rpm * distance`) ensures that all predictions are strictly positive and haul-length economies of scale are preserved.
> 3. **`src/train.py`**: Executes forward-chaining cross-validation, fits the production model on all 48,000 loads, and serializes the model and encoder bundles to `models/`.
> 4. **`src/predict.py`**: Generates `validation_predictions.csv` for all 12,000 loads, populates `data/december_chart_inputs.csv`, and automatically invokes `score.py`."

---

## 2:25 – 3:00 | December Forecast, Scorer Verification & Conclusion

**[On Screen: Terminal showing `score.py` passing & `scorer_results/candidate_december.png`]**

> "Finally, let's examine the December forecast and scorer results.
> 
> For the fixed December scenario (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb):
> - In historical data, Dry Van loads on this lane averaged $823.31.
> - Our December predictions range from **$814.77 to $848.25** (mean: $834.20).
> - As shown in `candidate_december.png`, the curve captures authentic freight market dynamics: Sunday weekend dips (~$823), midweek shipper volume peaks (~$840), and capacity tightening up to Christmas Day ($848.25).
> 
> The pipeline was validated via the official `score.py`:
> ```bash
> python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
> ```
> Both files passed with zero errors, exactly 12,000 IDs matched, positive rates confirmed, and all 31 December dates validated.
> 
> Everything can be reproduced with a single command:
> `python -m src.train && python -m src.predict`.
> 
> Thank you for your time, and I look forward to discussing the architecture further!"
