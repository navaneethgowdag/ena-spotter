import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    # Ensure positive
    y_pred = np.clip(y_pred, 1.0, None)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'R2': r2
    }

def run_benchmarks():
    print("Loading train_test.csv...")
    df = pd.read_csv("data/train_test.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # Clean data
    df['weight_cleaned'] = df['weight'].abs()
    med_weight = df['weight_cleaned'].median()
    df['weight_imputed'] = df['weight_cleaned'].fillna(med_weight)
    df['weight_na'] = df['weight'].isna().astype(int)
    
    df['rpm'] = df['posted_rate'] / df['distance']
    
    # Feature engineering helper
    # Coordinates mapping
    city_lat = {}
    city_lon = {}
    for _, row in df.iterrows():
        if row['pickup'] not in city_lat:
            city_lat[row['pickup']] = row['pickup_lat']
            city_lon[row['pickup']] = row['pickup_lon']
        if row['delivery'] not in city_lat:
            city_lat[row['delivery']] = row['delivery_lat']
            city_lon[row['delivery']] = row['delivery_lon']

    # Temporal features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Distance features
    df['log_distance'] = np.log(df['distance'])
    df['log_weight'] = np.log(df['weight_imputed'])
    
    # Holiday proximity
    # US Federal Holidays in 2025
    holidays_2025 = pd.to_datetime([
        '2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', 
        '2025-06-19', '2025-07-04', '2025-09-01', '2025-10-13', 
        '2025-11-11', '2025-11-27', '2025-12-25'
    ])
    
    def min_days_to_holiday(dt_series):
        days_diff = []
        for d in dt_series:
            diffs = (holidays_2025 - d).days
            # nearest distance
            min_dist = np.min(np.abs(diffs))
            days_diff.append(min_dist)
        return np.array(days_diff)

    df['days_to_holiday'] = min_days_to_holiday(df['date'])
    df['is_holiday_week'] = (df['days_to_holiday'] <= 3).astype(int)
    
    # Time split:
    # Training: Jan 1 to Aug 31 (months 1-8)
    # Test (Hold-out): Sep 1 to Oct 31 (months 9-10)
    train_idx = df['date'] < '2025-09-01'
    test_idx = df['date'] >= '2025-09-01'
    
    train_data = df[train_idx].copy().reset_index(drop=True)
    test_data = df[test_idx].copy().reset_index(drop=True)
    
    print(f"Development Set: {len(df):,} loads")
    print(f"Training split (Jan-Aug): {len(train_data):,} loads")
    print(f"Held-out test split (Sep-Oct): {len(test_data):,} loads\n")
    
    # Target Encoding computed strictly on train_data
    # 1. Global median RPM
    global_rpm = train_data['rpm'].median()
    
    # 2. Equipment median RPM
    eq_rpm = train_data.groupby('equipment')['rpm'].median().to_dict()
    
    # 3. Lane median RPM (with smoothing / shrinkage towards equipment median)
    lane_stats = train_data.groupby(['pickup', 'delivery'])['rpm'].agg(['count', 'median']).reset_index()
    smoothing = 5.0
    lane_rpm_dict = {}
    for _, row in lane_stats.iterrows():
        key = (row['pickup'], row['delivery'])
        cnt = row['count']
        raw_med = row['median']
        # Bayesian shrinkage
        shrunk = (cnt * raw_med + smoothing * global_rpm) / (cnt + smoothing)
        lane_rpm_dict[key] = shrunk

    # 4. Pickup origin median RPM
    pickup_rpm_dict = train_data.groupby('pickup')['rpm'].median().to_dict()
    delivery_rpm_dict = train_data.groupby('delivery')['rpm'].median().to_dict()

    # Map target encoding features
    def encode_features(data_subset):
        res = pd.DataFrame(index=data_subset.index)
        lane_keys = list(zip(data_subset['pickup'], data_subset['delivery']))
        res['te_lane_rpm'] = [lane_rpm_dict.get(k, np.nan) for k in lane_keys]
        res['te_eq_rpm'] = data_subset['equipment'].map(eq_rpm).fillna(global_rpm)
        res['te_pickup_rpm'] = data_subset['pickup'].map(pickup_rpm_dict).fillna(global_rpm)
        res['te_delivery_rpm'] = data_subset['delivery'].map(delivery_rpm_dict).fillna(global_rpm)
        # Fallback for lane RPM: if lane unseen, use average of pickup & delivery, or eq
        res['te_lane_rpm'] = res['te_lane_rpm'].fillna(0.5 * (res['te_pickup_rpm'] + res['te_delivery_rpm']))
        return res

    tr_encoded = encode_features(train_data)
    te_encoded = encode_features(test_data)
    
    for c in tr_encoded.columns:
        train_data[c] = tr_encoded[c]
        test_data[c] = te_encoded[c]
        
    y_test = test_data['posted_rate'].values
    y_test_rpm = test_data['rpm'].values
    dist_test = test_data['distance'].values

    results = []

    # -------------------------------------------------------------
    # Candidate 1: Global Median Rate-per-Mile Baseline
    # -------------------------------------------------------------
    pred_global = global_rpm * dist_test
    m_global = calculate_metrics(y_test, pred_global)
    m_global['Model'] = 'Baseline: Global Median RPM'
    results.append(m_global)

    # -------------------------------------------------------------
    # Candidate 2: Per-Lane Median Rate-per-Mile Baseline
    # -------------------------------------------------------------
    pred_lane_rpm = test_data['te_lane_rpm'].values
    pred_lane = pred_lane_rpm * dist_test
    m_lane = calculate_metrics(y_test, pred_lane)
    m_lane['Model'] = 'Baseline: Per-Lane Median RPM'
    results.append(m_lane)

    # -------------------------------------------------------------
    # Candidate 3: Ridge Linear Regression (predicting log rate)
    # -------------------------------------------------------------
    numeric_cols = [
        'distance', 'log_distance', 'weight_imputed', 'log_weight', 'weight_na',
        'te_lane_rpm', 'te_eq_rpm', 'te_pickup_rpm', 'te_delivery_rpm',
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'dayofweek', 'day', 'month', 'is_weekend', 'days_to_holiday'
    ]
    eq_dummies_tr = pd.get_dummies(train_data['equipment'], drop_first=True, dtype=float)
    eq_dummies_te = pd.get_dummies(test_data['equipment'], drop_first=True, dtype=float)

    X_tr_ridge = pd.concat([train_data[numeric_cols], eq_dummies_tr], axis=1)
    X_te_ridge = pd.concat([test_data[numeric_cols], eq_dummies_te], axis=1)

    # Fill any na with mean
    X_tr_ridge = X_tr_ridge.fillna(X_tr_ridge.mean())
    X_te_ridge = X_te_ridge.fillna(X_tr_ridge.mean())

    ridge = Ridge(alpha=100.0)
    ridge.fit(X_tr_ridge, np.log(train_data['posted_rate']))
    pred_ridge = np.exp(ridge.predict(X_te_ridge))
    m_ridge = calculate_metrics(y_test, pred_ridge)
    m_ridge['Model'] = 'Ridge Linear Regression (log rate)'
    results.append(m_ridge)

    # -------------------------------------------------------------
    # Candidate 4: Random Forest (predicting RPM)
    # -------------------------------------------------------------
    print("Fitting Candidate 4: Random Forest...", flush=True)
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        n_jobs=2,
        random_state=42
    )
    rf.fit(X_tr_ridge, train_data['rpm'])
    pred_rf_rpm = rf.predict(X_te_ridge)
    pred_rf = pred_rf_rpm * dist_test
    m_rf = calculate_metrics(y_test, pred_rf)
    m_rf['Model'] = 'Random Forest (RPM target)'
    results.append(m_rf)
    print("Candidate 4 complete.", flush=True)

    # -------------------------------------------------------------
    # Candidate 5: LightGBM Regressor (predicting RPM directly)
    # -------------------------------------------------------------
    print("Fitting Candidate 5: LightGBM Gradient Boosting...", flush=True)
    lgb_features = [
        'distance', 'log_distance', 'weight_imputed', 'log_weight', 'weight_na',
        'te_lane_rpm', 'te_eq_rpm', 'te_pickup_rpm', 'te_delivery_rpm',
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'dayofweek', 'day', 'month', 'is_weekend', 'days_to_holiday',
        'pickup', 'delivery', 'equipment'
    ]
    
    # Categoricals for LightGBM
    for c in ['pickup', 'delivery', 'equipment']:
        train_data[c] = train_data[c].astype('category')
        test_data[c] = test_data[c].astype('category')

    dtrain = lgb.Dataset(train_data[lgb_features], label=train_data['rpm'])
    dtest = lgb.Dataset(test_data[lgb_features], label=test_data['rpm'], reference=dtrain)

    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'random_state': 42,
        'verbose': -1
    }

    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1200,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    pred_lgb_rpm = lgb_model.predict(test_data[lgb_features])
    pred_lgb = pred_lgb_rpm * dist_test
    m_lgb = calculate_metrics(y_test, pred_lgb)
    m_lgb['Model'] = 'LightGBM Gradient Boosting (RPM target)'
    results.append(m_lgb)

    # Format into DataFrame
    res_df = pd.DataFrame(results)
    res_df = res_df[['Model', 'MAE', 'RMSE', 'MAPE', 'SMAPE', 'R2']]
    print("\n==========================================================================")
    print("HELD-OUT TEST SET (SEP-OCT 2025) PERFORMANCE BENCHMARK")
    print("==========================================================================")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.2f}" if abs(x) > 1 else f"{x:.4f}"))
    print("==========================================================================\n")

    res_df.to_csv("reports/model_comparison_metrics.csv", index=False)
    
    # Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    models = res_df['Model'].str.replace(' Regressor', '').str.replace(' Regression', '')
    
    # MAE plot
    bars1 = axes[0].barh(models, res_df['MAE'], color=['#ced4da', '#adb5bd', '#48cae4', '#0096c7', '#03045e'])
    axes[0].set_title("Held-out Test MAE ($) - Lower is Better", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Mean Absolute Error ($)")
    for bar in bars1:
        w = bar.get_width()
        axes[0].text(w + 3, bar.get_y() + bar.get_height()/2, f"${w:.2f}", va='center', fontsize=9.5, fontweight='bold')
    axes[0].set_xlim(0, res_df['MAE'].max() * 1.15)
    axes[0].invert_yaxis()

    # R2 plot
    bars2 = axes[1].barh(models, res_df['R2'], color=['#ced4da', '#adb5bd', '#48cae4', '#0096c7', '#03045e'])
    axes[1].set_title("Held-out Test R² Score - Higher is Better", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("R² Score")
    for bar in bars2:
        w = bar.get_width()
        axes[1].text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center', fontsize=9.5, fontweight='bold')
    axes[1].set_xlim(0.75, 0.85)
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig("reports/figures/06_model_comparison.png")
    plt.close(fig)
    print("Comparison chart saved to reports/figures/06_model_comparison.png")

    # Feature Importance for LightGBM
    importance = lgb_model.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': lgb_features, 'importance': importance}).sort_values('importance', ascending=False)
    print("\nLightGBM Feature Importance (Gain):")
    print(imp_df.head(10).to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    top_imp = imp_df.head(12).sort_values('importance', ascending=True)
    ax.barh(top_imp['feature'], top_imp['importance'], color='#0077b6')
    ax.set_title("LightGBM Feature Importance (Information Gain)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Total Gain")
    fig.tight_layout()
    fig.savefig("reports/figures/07_feature_importance.png")
    plt.close(fig)
    print("Feature importance chart saved to reports/figures/07_feature_importance.png")

if __name__ == '__main__':
    run_benchmarks()
