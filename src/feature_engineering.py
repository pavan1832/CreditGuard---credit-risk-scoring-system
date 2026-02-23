"""
Feature Engineering Pipeline
------------------------------
Handles: imputation, encoding, scaling, feature creation, time-based split.

Design decisions:
- Median imputation for numeric → robust to outliers in financial data
- Frequency encoding for high-cardinality categoricals (loan_purpose)
- One-hot encoding for low-cardinality categoricals (employment_status)
- StandardScaler on numeric features → required for Logistic Regression baseline
- Time-based train/val split → prevents data leakage (critical in production)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = "data/raw/loan_data.csv"
PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "src/artifacts"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET = "loan_default"
DATE_COL = "application_date"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "loan_tenure_months",
    "interest_rate",
    "credit_score",
    "credit_history_length",
    "past_defaults",
    "num_open_accounts",
    "num_credit_inquiries",
    "debt_to_income",
    "loan_to_income",
]

CATEGORICAL_FEATURES = ["employment_status", "loan_purpose"]

# Business-derived features (created after imputation)
DERIVED_FEATURES = [
    "income_per_year_employed",   # earning power per year worked
    "emi_burden_score",           # proxy for payment stress
    "credit_risk_index",          # composite risk signal
]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    print(f"[load] Raw data: {df.shape} | Default rate: {df[TARGET].mean():.2%}")
    return df


def impute_missing(df: pd.DataFrame, fit_medians: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Median imputation fitted on train set, applied to val/test.
    Returns (imputed_df, medians_dict)
    """
    impute_cols = ["credit_history_length", "employment_years", "num_credit_inquiries"]

    if fit_medians is None:
        fit_medians = {}
        for col in impute_cols:
            fit_medians[col] = df[col].median()

    for col in impute_cols:
        df[col] = df[col].fillna(fit_medians[col])

    return df, fit_medians


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-driven derived features.
    These are engineered AFTER imputation.
    """
    # Income per year of employment (0 → small constant to avoid div-by-zero)
    df["income_per_year_employed"] = df["annual_income"] / (df["employment_years"] + 1)

    # EMI burden score: monthly EMI relative to income, scaled by interest rate
    monthly_income = df["annual_income"] / 12
    emi = (
        df["loan_amount"]
        * (df["interest_rate"] / 1200)
        * (1 + df["interest_rate"] / 1200) ** df["loan_tenure_months"]
        / ((1 + df["interest_rate"] / 1200) ** df["loan_tenure_months"] - 1)
    )
    df["emi_burden_score"] = (emi / monthly_income).clip(0, 5)

    # Credit risk index: combines past defaults + inquiries − credit score signal
    df["credit_risk_index"] = (
        df["past_defaults"] * 2
        + df["num_credit_inquiries"] * 0.5
        - (df["credit_score"] - 600) / 100
    ).clip(-5, 10)

    return df


def encode_categoricals(df: pd.DataFrame, fit_encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    OHE for employment_status (4 levels), label-frequency encode loan_purpose.
    Returns (encoded_df, encoders_dict)
    """
    if fit_encoders is None:
        fit_encoders = {}

        # One-Hot Encode: employment_status
        dummies = pd.get_dummies(df["employment_status"], prefix="emp", drop_first=False, dtype=int)
        fit_encoders["emp_cols"] = dummies.columns.tolist()
        df = pd.concat([df, dummies], axis=1)

        # Frequency encode: loan_purpose
        freq_map = df["loan_purpose"].value_counts(normalize=True).to_dict()
        fit_encoders["loan_purpose_freq"] = freq_map
        df["loan_purpose_freq"] = df["loan_purpose"].map(freq_map)

    else:
        # Apply pre-fitted encoders (for val/test)
        dummies = pd.get_dummies(df["employment_status"], prefix="emp", drop_first=False, dtype=int)
        for col in fit_encoders["emp_cols"]:
            if col not in dummies.columns:
                dummies[col] = 0
        df = pd.concat([df, dummies[fit_encoders["emp_cols"]]], axis=1)

        df["loan_purpose_freq"] = df["loan_purpose"].map(fit_encoders["loan_purpose_freq"]).fillna(0.01)

    df = df.drop(columns=["employment_status", "loan_purpose"], errors="ignore")
    return df, fit_encoders


def scale_features(df: pd.DataFrame, feature_cols: list, scaler=None):
    """
    StandardScaler fit on train, transform on val/test.
    Returns (df_with_scaled_cols, fitted_scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


def time_based_split(df: pd.DataFrame, date_col: str = DATE_COL, val_ratio: float = 0.2):
    """
    Time-based split: earliest records → train, latest → validation.
    This is the ONLY correct split strategy for financial time-series.
    Random splits cause data leakage in deployment scenarios.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_ratio))
    train = df.iloc[:split_idx].copy()
    val = df.iloc[split_idx:].copy()
    print(f"[split] Train: {train.shape}, Val: {val.shape}")
    print(f"[split] Train default rate: {train[TARGET].mean():.2%} | Val: {val[TARGET].mean():.2%}")
    return train, val


def get_feature_cols(df: pd.DataFrame) -> list:
    """Get all model input features (exclude target and date)."""
    exclude = [TARGET, DATE_COL, "application_date"]
    return [c for c in df.columns if c not in exclude]


def run_pipeline(save: bool = True):
    """
    Full feature engineering pipeline.
    Saves train/val CSVs and preprocessing artifacts.
    """
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # 1. Load
    df = load_raw_data()

    # 2. Time-based split (before any fitting to prevent leakage)
    train, val = time_based_split(df)

    # 3. Impute on train, apply to val
    train, medians = impute_missing(train)
    val, _ = impute_missing(val, fit_medians=medians)

    # 4. Feature engineering
    train = engineer_features(train)
    val = engineer_features(val)

    # 5. Encode categoricals
    train, encoders = encode_categoricals(train)
    val, _ = encode_categoricals(val, fit_encoders=encoders)

    # 6. Identify numeric cols for scaling
    feature_cols = get_feature_cols(train)
    numeric_cols_to_scale = [
        c for c in feature_cols
        if train[c].dtype in [np.float64, np.int64, float, int]
    ]

    # 7. Scale
    train, scaler = scale_features(train, numeric_cols_to_scale)
    val, _ = scale_features(val, numeric_cols_to_scale, scaler=scaler)

    feature_cols = get_feature_cols(train)
    print(f"\n[features] Total model features: {len(feature_cols)}")
    print(f"[features] {feature_cols}")

    # 8. Save artifacts
    if save:
        train.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
        val.to_csv(f"{PROCESSED_DIR}/val.csv", index=False)

        artifacts = {
            "medians": medians,
            "encoders": encoders,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "numeric_cols_scaled": numeric_cols_to_scale,
        }
        with open(f"{ARTIFACTS_DIR}/preprocessing.pkl", "wb") as f:
            pickle.dump(artifacts, f)

        print(f"\n[save] Saved processed data to {PROCESSED_DIR}/")
        print(f"[save] Saved preprocessing artifacts to {ARTIFACTS_DIR}/preprocessing.pkl")

    return train, val, feature_cols


if __name__ == "__main__":
    train, val, feature_cols = run_pipeline()
    print("\nSample train features:")
    print(train[feature_cols].head(3).to_string())
