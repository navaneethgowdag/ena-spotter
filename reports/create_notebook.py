import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Spotter Freight Rate Prediction - Exploratory Data Analysis & Validation Strategy\n",
            "\n",
            "This notebook documents the exploratory data analysis, data quality audit, and time-based validation design for the Spotter Freight Rate Prediction challenge."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "%matplotlib inline\n",
            "\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Datasets & Schema Inspection"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_df = pd.read_csv('../data/train_test.csv')\n",
            "val_df = pd.read_csv('../data/validation.csv')\n",
            "dec_df = pd.read_csv('../data/december_chart_inputs.csv')\n",
            "\n",
            "print(f\"Train shape: {train_df.shape}\")\n",
            "print(f\"Validation shape: {val_df.shape}\")\n",
            "print(f\"December chart inputs shape: {dec_df.shape}\")\n",
            "train_df.head(5)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Temporal Coverage & Missing Values\n",
            "\n",
            "- `train_test.csv`: 2025-01-01 to 2025-10-31 (10 months, 48,000 loads)\n",
            "- `validation.csv`: 2025-11-01 to 2025-12-31 (2 months, 12,000 loads)\n",
            "- `december_chart_inputs.csv`: 2025-12-01 to 2025-12-31 (31 loads, fixed Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Train missing values:\")\n",
            "print(train_df.isna().sum()[train_df.isna().sum() > 0])\n",
            "\n",
            "print(\"\\nValidation missing values:\")\n",
            "print(val_df.isna().sum()[val_df.isna().sum() > 0])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Data Quality Issues & Anomaly Remediation\n",
            "\n",
            "1. **Negative Weights**: 292 loads in train and 145 in validation have negative values (e.g. -36,559 lbs). The distribution of `abs(weight)` perfectly aligns with positive freight weights. Handled via `np.abs(weight)`.\n",
            "2. **Missing Weights**: Imputed with the dataset median (~31,436 lbs) and an explicit indicator `weight_is_na`.\n",
            "3. **Missing Market Index**: Imputed with the temporal rolling average or median.\n",
            "4. **Unseen Cities/Lanes in Validation**: 8 cities and 736 lanes in validation are never observed in training. Handled using Bayesian smoothed target encoding with pickup/delivery/global fallbacks and spatial coordinates."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Weight remediation demonstration\n",
            "print(\"Raw negative weights in train:\", (train_df['weight'] < 0).sum())\n",
            "cleaned_weights = train_df['weight'].abs()\n",
            "print(\"Cleaned weight summary:\")\n",
            "print(cleaned_weights.describe())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Key Exploratory Findings\n",
            "\n",
            "Below we visualize the core freight rate distributions, equipment economics, haul-length dynamics, and temporal seasonality."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_df['rate_per_mile'] = train_df['posted_rate'] / train_df['distance']\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "axes[0].hist(train_df['posted_rate'], bins=50, color='#1b4965', edgecolor='white')\n",
            "axes[0].set_title(\"Distribution of Posted Rate ($)\", fontweight='bold')\n",
            "axes[0].set_xlabel(\"Posted Rate ($)\")\n",
            "\n",
            "axes[1].hist(train_df['rate_per_mile'].clip(upper=7), bins=50, color='#62b6cb', edgecolor='white')\n",
            "axes[1].set_title(\"Distribution of Rate Per Mile ($/mi)\", fontweight='bold')\n",
            "axes[1].set_xlabel(\"Rate Per Mile ($/mi)\")\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "equipments = sorted(train_df['equipment'].dropna().unique())\n",
            "rate_by_eq = [train_df[train_df['equipment'] == eq]['posted_rate'] for eq in equipments]\n",
            "rpm_by_eq = [train_df[train_df['equipment'] == eq]['rate_per_mile'] for eq in equipments]\n",
            "\n",
            "axes[0].boxplot(rate_by_eq, tick_labels=equipments, patch_artist=True, boxprops=dict(facecolor='#bee9e8'))\n",
            "axes[0].set_title(\"Posted Rate by Equipment Type\", fontweight='bold')\n",
            "axes[0].set_ylabel(\"Rate ($)\")\n",
            "\n",
            "axes[1].boxplot(rpm_by_eq, tick_labels=equipments, patch_artist=True, boxprops=dict(facecolor='#62b6cb'))\n",
            "axes[1].set_title(\"Rate Per Mile by Equipment Type\", fontweight='bold')\n",
            "axes[1].set_ylabel(\"Rate Per Mile ($/mi)\")\n",
            "axes[1].set_ylim(0, 6)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Model Benchmarks & Comparison on Held-Out Test Set (Sep-Oct 2025)\n",
            "\n",
            "We compare our baseline models (Global Median RPM, Per-Lane Median RPM), Ridge Regression, Random Forest, and LightGBM Gradient Boosting."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if os.path.exists('../reports/model_comparison_metrics.csv'):\n",
            "    metrics_df = pd.read_csv('../reports/model_comparison_metrics.csv')\n",
            "    display(metrics_df)\n",
            "else:\n",
            "    print('Run reports/model_benchmarks.py to generate metrics.')"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/01_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Created notebooks/01_eda.ipynb")
