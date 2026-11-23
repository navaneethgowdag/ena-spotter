import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run():
    out_lines = []
    def log(msg=""):
        out_lines.append(str(msg))

    log("=======================================================")
    log("SPOTTER FREIGHT RATE PREDICTION - COMPREHENSIVE EDA LOG")
    log("=======================================================\n")

    train_path = "data/train_test.csv"
    val_path = "data/validation.csv"
    dec_path = "data/december_chart_inputs.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    dec_df = pd.read_csv(dec_path)

    log(f"1. DATASET SHAPES & COLUMNS:")
    log(f"   train_test: {train_df.shape[0]:,} rows x {train_df.shape[1]} cols")
    log(f"   Columns: {train_df.columns.tolist()}")
    log(f"   validation: {val_df.shape[0]:,} rows x {val_df.shape[1]} cols")
    log(f"   Columns: {val_df.columns.tolist()}")
    log(f"   december_chart_inputs: {dec_df.shape[0]:,} rows x {dec_df.shape[1]} cols")
    log(f"   Columns: {dec_df.columns.tolist()}\n")

    # Dates
    train_df['date'] = pd.to_datetime(train_df['date'])
    val_df['date'] = pd.to_datetime(val_df['date'])
    dec_df['date'] = pd.to_datetime(dec_df['date'])

    log(f"2. TEMPORAL COVERAGE:")
    log(f"   train_test dates: {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')} ({train_df['date'].nunique()} days)")
    log(f"   validation dates: {val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')} ({val_df['date'].nunique()} days)")
    log(f"   december dates:   {dec_df['date'].min().strftime('%Y-%m-%d')} to {dec_df['date'].max().strftime('%Y-%m-%d')} ({dec_df['date'].nunique()} days)")

    # Monthly distribution
    log("\n   Monthly volume (train_test):")
    train_m = train_df.set_index('date').resample('ME').size()
    for dt, cnt in train_m.items():
        log(f"     {dt.strftime('%Y-%m')}: {cnt:,} loads")

    log("\n   Monthly volume (validation):")
    val_m = val_df.set_index('date').resample('ME').size()
    for dt, cnt in val_m.items():
        log(f"     {dt.strftime('%Y-%m')}: {cnt:,} loads")

    # Missing values
    log(f"\n3. MISSING VALUES:")
    log("   train_test missing:")
    for col in train_df.columns:
        n_miss = train_df[col].isna().sum()
        if n_miss > 0:
            log(f"     {col}: {n_miss:,} ({n_miss / len(train_df) * 100:.2f}%)")
    if train_df.isna().sum().sum() == 0:
        log("     None!")

    log("   validation missing:")
    for col in val_df.columns:
        n_miss = val_df[col].isna().sum()
        if n_miss > 0:
            log(f"     {col}: {n_miss:,} ({n_miss / len(val_df) * 100:.2f}%)")
    if val_df.isna().sum().sum() == 0:
        log("     None!")

    log("   december_chart_inputs missing (excluding predicted_rate):")
    for col in dec_df.columns:
        if col == 'predicted_rate':
            continue
        n_miss = dec_df[col].isna().sum()
        if n_miss > 0:
            log(f"     {col}: {n_miss:,} ({n_miss / len(dec_df) * 100:.2f}%)")

    # Duplicates & IDs
    log(f"\n4. IDENTIFIERS & DUPLICATES:")
    log(f"   train_test load_id unique: {train_df['load_id'].nunique():,} / {len(train_df):,}")
    log(f"   validation load_id unique: {val_df['load_id'].nunique():,} / {len(val_df):,}")
    log(f"   Overlap between train & val load_id: {len(set(train_df['load_id']) & set(val_df['load_id']))}")

    # Target stats
    log(f"\n5. TARGET VARIABLE (posted_rate) DISTRIBUTION:")
    log(train_df['posted_rate'].describe(percentiles=[0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]).to_string())
    log(f"   Non-positive values count: {(train_df['posted_rate'] <= 0).sum()}")

    # Distance and Rate Per Mile
    train_df['rate_per_mile'] = train_df['posted_rate'] / train_df['distance']
    log(f"\n6. RATE PER MILE (posted_rate / distance) DISTRIBUTION:")
    log(train_df['rate_per_mile'].describe(percentiles=[0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]).to_string())

    # Categorical distributions & shift
    log(f"\n7. CATEGORICAL FEATURES & DISTRIBUTION SHIFT:")
    for cat in ['equipment', 'pickup', 'delivery']:
        train_vals = set(train_df[cat].dropna().unique())
        val_vals = set(val_df[cat].dropna().unique())
        dec_vals = set(dec_df[cat].dropna().unique())
        unseen_val = val_vals - train_vals
        unseen_dec = dec_vals - train_vals
        log(f"   {cat.upper()}:")
        log(f"     Train categories: {len(train_vals)}, Val: {len(val_vals)}, Dec: {len(dec_vals)}")
        log(f"     Unseen in Val: {len(unseen_val)} -> {unseen_val}")
        log(f"     Unseen in Dec: {len(unseen_dec)} -> {unseen_dec}")

    train_lanes = set(zip(train_df['pickup'], train_df['delivery']))
    val_lanes = set(zip(val_df['pickup'], val_df['delivery']))
    dec_lanes = set(zip(dec_df['pickup'], dec_df['delivery']))
    log(f"   LANES (pickup -> delivery):")
    log(f"     Train unique lanes: {len(train_lanes):,}")
    log(f"     Val unique lanes: {len(val_lanes):,}")
    unseen_lanes = val_lanes - train_lanes
    log(f"     Unseen lanes in validation: {len(unseen_lanes):,} ({len(unseen_lanes)/len(val_lanes)*100:.2f}%)")
    log(f"     Dec lane in train? {dec_lanes.issubset(train_lanes)}: {dec_lanes}")

    # Numerical feature stats comparison
    log(f"\n8. NUMERICAL COMPARISON (Train vs Val):")
    for num_col in ['distance', 'weight', 'market_index', 'quote_signal']:
        t_desc = train_df[num_col].describe()
        v_desc = val_df[num_col].describe()
        log(f"   {num_col.upper()}:")
        log(f"     Train: mean={t_desc['mean']:.2f}, std={t_desc['std']:.2f}, min={t_desc['min']:.2f}, median={t_desc['50%']:.2f}, max={t_desc['max']:.2f}")
        log(f"     Val:   mean={v_desc['mean']:.2f}, std={v_desc['std']:.2f}, min={v_desc['min']:.2f}, median={v_desc['50%']:.2f}, max={v_desc['max']:.2f}")

    # Correlation analysis
    log(f"\n9. CORRELATION MATRIX (Train):")
    num_cols = ['posted_rate', 'rate_per_mile', 'distance', 'weight', 'market_index', 'quote_signal']
    log(train_df[num_cols].corr().to_string())

    # Deep dive on quote_signal and market_index
    log(f"\n10. NATURE OF market_index & quote_signal:")
    mi_group = train_df.groupby('date')['market_index'].agg(['count', 'nunique', 'min', 'max', 'std'])
    log(f"    market_index daily variation: is market_index strictly constant per date? {(mi_group['nunique'] == 1).all()}")
    log(f"    market_index overall range: [{train_df['market_index'].min():.4f}, {train_df['market_index'].max():.4f}]")
    log(f"    validation market_index overall range: [{val_df['market_index'].min():.4f}, {val_df['market_index'].max():.4f}]")

    # quote_signal check:
    # Notice: quote_signal vs posted_rate / distance
    # How close is quote_signal * distance to posted_rate?
    prod = train_df['quote_signal'] * train_df['distance']
    ratio = train_df['posted_rate'] / prod
    log(f"    posted_rate / (quote_signal * distance) summary:")
    log(f"      mean={ratio.mean():.4f}, std={ratio.std():.4f}, min={ratio.min():.4f}, median={ratio.median():.4f}, max={ratio.max():.4f}")

    # Let's check december_chart_inputs.csv
    log(f"\n11. DECEMBER INPUTS CHARACTERISTICS:")
    log(f"    Pickup: {dec_df['pickup'].unique()}")
    log(f"    Delivery: {dec_df['delivery'].unique()}")
    log(f"    Distance: {dec_df['distance'].unique()}")
    log(f"    Equipment: {dec_df['equipment'].unique()}")
    log(f"    Weight: {dec_df['weight'].unique()}")
    log(f"    Dates: {len(dec_df)} rows from {dec_df['date'].min().strftime('%Y-%m-%d')} to {dec_df['date'].max().strftime('%Y-%m-%d')}")
    log(f"    Are market_index or quote_signal in december_chart_inputs? {'market_index' in dec_df.columns or 'quote_signal' in dec_df.columns}")

    # Check Lexington to Fort Wayne in train_test
    lex_fw_train = train_df[(train_df['pickup'] == 'Lexington') & (train_df['delivery'] == 'Fort Wayne')]
    log(f"    Historical loads in Train for Lexington -> Fort Wayne: {len(lex_fw_train)}")
    if len(lex_fw_train) > 0:
        log(f"    Historical rates for Lexington -> Fort Wayne:")
        log(lex_fw_train[['distance', 'equipment', 'weight', 'date', 'posted_rate', 'rate_per_mile', 'market_index', 'quote_signal']].head(10).to_string())

    # Write output to file
    with open("reports/eda_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("EDA summary written to reports/eda_summary.txt")

    # NOW GENERATE HIGH-QUALITY EDA FIGURES
    os.makedirs("reports/figures", exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Target and Rate per Mile distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=200)
    axes[0].hist(train_df['posted_rate'], bins=60, color='#1b4965', edgecolor='white', alpha=0.85)
    axes[0].set_title("Distribution of Posted Rate ($)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Posted Rate ($)")
    axes[0].set_ylabel("Load Count")

    axes[1].hist(np.log1p(train_df['posted_rate']), bins=60, color='#62b6cb', edgecolor='white', alpha=0.85)
    axes[1].set_title("Distribution of log(1 + Posted Rate)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("log(1 + Posted Rate)")

    axes[2].hist(train_df['rate_per_mile'].clip(upper=10), bins=60, color='#5fa8d3', edgecolor='white', alpha=0.85)
    axes[2].set_title("Distribution of Rate Per Mile ($/mi)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Rate Per Mile ($/mi)")
    fig.tight_layout()
    fig.savefig("reports/figures/01_target_distribution.png")
    plt.close(fig)

    # Figure 2: Rate vs Distance & Weight
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=200)
    sample_sub = train_df.sample(min(5000, len(train_df)), random_state=42)
    for eq, grp in sample_sub.groupby('equipment'):
        axes[0].scatter(grp['distance'], grp['posted_rate'], label=eq, alpha=0.35, s=15)
    axes[0].set_title("Posted Rate vs Distance by Equipment", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Distance (miles)")
    axes[0].set_ylabel("Posted Rate ($)")
    axes[0].legend()

    axes[1].scatter(sample_sub['distance'], sample_sub['rate_per_mile'], alpha=0.35, s=15, color='#ca6702')
    axes[1].set_title("Rate Per Mile vs Distance (Haul Length Economics)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Distance (miles)")
    axes[1].set_ylabel("Rate Per Mile ($/mi)")
    axes[1].set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig("reports/figures/02_rate_vs_distance.png")
    plt.close(fig)

    # Figure 3: Equipment Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    equipments = sorted(train_df['equipment'].dropna().unique())
    rate_by_eq = [train_df[train_df['equipment'] == eq]['posted_rate'] for eq in equipments]
    rpm_by_eq = [train_df[train_df['equipment'] == eq]['rate_per_mile'] for eq in equipments]
    
    axes[0].boxplot(rate_by_eq, labels=equipments, patch_artist=True, boxprops=dict(facecolor='#bee9e8'))
    axes[0].set_title("Posted Rate by Equipment Type", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Posted Rate ($)")

    axes[1].boxplot(rpm_by_eq, labels=equipments, patch_artist=True, boxprops=dict(facecolor='#62b6cb'))
    axes[1].set_title("Rate Per Mile by Equipment Type", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Rate Per Mile ($/mi)")
    axes[1].set_ylim(0, 7)
    fig.tight_layout()
    fig.savefig("reports/figures/03_equipment_comparison.png")
    plt.close(fig)

    # Figure 4: Seasonality and Temporal Trend
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=200)
    train_df['month'] = train_df['date'].dt.month
    train_df['dayofweek'] = train_df['date'].dt.day_name()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    daily_avg = train_df.groupby('date')['rate_per_mile'].mean()
    axes[0, 0].plot(daily_avg.index, daily_avg.values, color='#005f73', lw=1.2)
    daily_roll = daily_avg.rolling(7, center=True).mean()
    axes[0, 0].plot(daily_roll.index, daily_roll.values, color='#ee9b00', lw=2.2, label='7-day MA')
    axes[0, 0].set_title("Daily Average Rate Per Mile (Trend Over Time)", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Avg Rate Per Mile ($/mi)")
    axes[0, 0].legend()

    daily_mi = train_df.groupby('date')['market_index'].first()
    axes[0, 1].plot(daily_mi.index, daily_mi.values, color='#9b2226', lw=1.8)
    axes[0, 1].set_title("Daily Market Index", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("Market Index")

    monthly_rpm = train_df.groupby('month')['rate_per_mile'].mean()
    axes[1, 0].bar(monthly_rpm.index, monthly_rpm.values, color='#0a9396', width=0.6)
    axes[1, 0].set_title("Monthly Average Rate Per Mile", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Avg Rate Per Mile ($/mi)")

    dow_rpm = train_df.groupby('dayofweek')['rate_per_mile'].mean().reindex(dow_order)
    axes[1, 1].bar(dow_rpm.index, dow_rpm.values, color='#94d2bd', width=0.6)
    axes[1, 1].set_title("Day of Week Average Rate Per Mile", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Avg Rate Per Mile ($/mi)")
    axes[1, 1].tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig("reports/figures/04_seasonality_and_trends.png")
    plt.close(fig)

    # Figure 5: Quote Signal vs Rate Per Mile
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.scatter(sample_sub['quote_signal'], sample_sub['rate_per_mile'], alpha=0.3, s=15, color='#001219')
    max_val = min(sample_sub['quote_signal'].max(), sample_sub['rate_per_mile'].max())
    ax.plot([1, 6], [1, 6], color='#d62828', linestyle='--', lw=2, label='1:1 Line')
    ax.set_title("Quote Signal vs Actual Rate Per Mile", fontsize=13, fontweight='bold')
    ax.set_xlabel("Quote Signal ($/mi)")
    ax.set_ylabel("Actual Rate Per Mile ($/mi)")
    ax.set_xlim(1, 6)
    ax.set_ylim(1, 6)
    ax.legend()
    fig.tight_layout()
    fig.savefig("reports/figures/05_quote_signal_vs_rpm.png")
    plt.close(fig)

    print("All figures generated successfully in reports/figures/")

if __name__ == "__main__":
    run()
