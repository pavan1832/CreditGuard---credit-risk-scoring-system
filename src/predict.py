"""
Prediction & Explainability Module
------------------------------------
Handles single/batch predictions and feature-contribution explainability.
Uses SHAP when available, falls back to correct log-odds decomposition for LR.
"""
import os, pickle, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

warnings.filterwarnings("ignore")
ARTIFACTS_DIR = "src/artifacts"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET = "loan_default"
DATE_COL = "application_date"
RISK_THRESHOLDS = {"LOW": 0.20, "MEDIUM": 0.45}

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def load_model_and_artifacts():
    prep = pickle.load(open(f"{ARTIFACTS_DIR}/preprocessing.pkl", "rb"))
    meta = pickle.load(open(f"{ARTIFACTS_DIR}/model_metadata.pkl", "rb"))
    best = meta["best_model"].lower().replace(" ", "_")
    model = pickle.load(open(f"{ARTIFACTS_DIR}/model_{best}.pkl", "rb"))
    return model, prep, meta["best_model"]


def preprocess_single_customer(customer_data: dict, prep: dict) -> pd.DataFrame:
    df = pd.DataFrame([customer_data])

    # Impute
    for col, med in prep["medians"].items():
        if col not in df.columns or pd.isna(df[col].iloc[0]):
            df[col] = med

    # Derive features
    df["income_per_year_employed"] = df["annual_income"] / (df["employment_years"] + 1)
    r = df["interest_rate"].iloc[0]
    L = df["loan_amount"].iloc[0]
    T = df["loan_tenure_months"].iloc[0]
    mi = df["annual_income"].iloc[0] / 12
    emi = (L * (r/1200) * (1+r/1200)**T) / ((1+r/1200)**T - 1)
    df["emi_burden_score"] = min(emi / mi, 5)
    df["credit_risk_index"] = (
        df["past_defaults"].iloc[0] * 2
        + df["num_credit_inquiries"].iloc[0] * 0.5
        - (df["credit_score"].iloc[0] - 600) / 100
    )

    # Encode categoricals
    enc = prep["encoders"]
    dummies = pd.get_dummies(df["employment_status"], prefix="emp", drop_first=False, dtype=int)
    for col in enc["emp_cols"]:
        if col not in dummies.columns:
            dummies[col] = 0
    df = pd.concat([df, dummies[enc["emp_cols"]]], axis=1)
    df["loan_purpose_freq"] = df["loan_purpose"].map(enc["loan_purpose_freq"]).fillna(0.01)
    df = df.drop(columns=["employment_status", "loan_purpose"], errors="ignore")

    # Scale
    numeric_cols = prep["numeric_cols_scaled"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
    df[numeric_cols] = prep["scaler"].transform(df[numeric_cols])

    # Align to model features
    fc = prep["feature_cols"]
    for col in fc:
        if col not in df.columns:
            df[col] = 0
    return df[fc]


def get_risk_label(prob: float) -> str:
    if prob < RISK_THRESHOLDS["LOW"]:
        return "🟢 LOW RISK"
    elif prob < RISK_THRESHOLDS["MEDIUM"]:
        return "🟡 MEDIUM RISK"
    return "🔴 HIGH RISK"


def compute_local_explanation(model, X: pd.DataFrame, model_name: str):
    """
    Compute per-feature contributions to the prediction.

    For Logistic Regression: exact log-odds decomposition.
        log-odds = intercept + sum(coef_i * x_i)
        contribution_i = coef_i * x_i  (in log-odds space)
        This is the correct, interpretable breakdown — not an approximation.

    For tree models (XGBoost/GBM): use SHAP TreeExplainer if available,
        else use built-in feature_importances_ weighted by feature value sign.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    # ── Logistic Regression: exact log-odds decomposition ──────────────────
    if isinstance(model, LogisticRegression):
        coef = model.coef_[0]          # shape (n_features,)
        x = X.iloc[0].values           # scaled feature vector
        contributions = coef * x       # log-odds contribution per feature
        base_value = float(expit(model.intercept_[0]))
        return contributions, base_value, "log-odds contribution (coef × scaled feature)"

    # ── XGBoost: SHAP TreeExplainer ─────────────────────────────────────────
    if SHAP_AVAILABLE:
        try:
            from xgboost import XGBClassifier
            if isinstance(model, XGBClassifier):
                expl = shap.TreeExplainer(model)
                sv = expl.shap_values(X)
                return sv[0], float(expl.expected_value), "SHAP TreeExplainer"
        except Exception:
            pass

    # ── GradientBoosting / fallback: SHAP KernelExplainer sample ───────────
    if SHAP_AVAILABLE:
        try:
            val = pd.read_csv("data/processed/val.csv").sample(100, random_state=42)
            fc = X.columns.tolist()
            for c in fc:
                if c not in val.columns:
                    val[c] = 0
            bg = val[fc]
            expl = shap.KernelExplainer(model.predict_proba, bg)
            sv = expl.shap_values(X, nsamples=50)
            # sv is list [class0, class1] for binary
            sv1 = sv[1] if isinstance(sv, list) else sv
            return sv1[0], float(expl.expected_value[1]), "SHAP KernelExplainer"
        except Exception:
            pass

    # ── Final fallback: feature_importances weighted by value direction ─────
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        x = X.iloc[0].values
        # Weight importance by direction (sign of feature value relative to mean=0 after scaling)
        contributions = fi * np.sign(x) * np.abs(x).clip(0, 3)
        # Normalize to probability scale
        prob = float(model.predict_proba(X)[:, 1][0])
        contributions = contributions / (contributions.sum() + 1e-9) * (prob - 0.5) * 2
    else:
        contributions = np.zeros(X.shape[1])

    base_value = float(model.predict_proba(X)[:, 1][0])
    return contributions, base_value, "feature importance × value direction"


def predict_single(customer_data: dict) -> dict:
    model, prep, model_name = load_model_and_artifacts()
    X = preprocess_single_customer(customer_data, prep)
    prob = float(model.predict_proba(X)[:, 1][0])
    contributions, base_val, method = compute_local_explanation(model, X, model_name)

    return {
        "default_probability": round(prob, 4),
        "risk_label": get_risk_label(prob),
        "model_name": model_name,
        "shap_values": contributions,
        "shap_base_value": base_val,
        "feature_names": X.columns.tolist(),
        "feature_values": X.iloc[0].values,
        "explanation_method": method,
    }


def compute_global_importance(model, X_val: pd.DataFrame, model_name: str):
    """Global feature importance."""
    sample = X_val.sample(min(500, len(X_val)), random_state=42)

    if SHAP_AVAILABLE:
        try:
            from xgboost import XGBClassifier
            if isinstance(model, XGBClassifier):
                expl = shap.TreeExplainer(model)
                sv = expl.shap_values(sample)
                return np.abs(sv).mean(axis=0), sample.columns.tolist()
        except Exception:
            pass

    from sklearn.linear_model import LogisticRegression
    if isinstance(model, LogisticRegression):
        # Mean absolute log-odds contribution across validation set
        contribs = np.abs(sample.values * model.coef_[0])
        return contribs.mean(axis=0), sample.columns.tolist()

    if hasattr(model, "feature_importances_"):
        return model.feature_importances_, sample.columns.tolist()

    return np.ones(sample.shape[1]), sample.columns.tolist()


def plot_global_importance(model, X_val: pd.DataFrame, model_name: str):
    importances, feature_names = compute_global_importance(model, X_val, model_name)
    s = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(s)))
    s.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Mean |Contribution| to Default Probability (Higher = Stronger Risk Driver)")
    ax.set_title(f"Global Feature Importance — {model_name}")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    path = f"{REPORTS_DIR}/shap_global_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[explain] Saved: {path}")
    return s


def plot_local_explanation(result: dict, customer_id: str = "CUST_001"):
    sv = result["shap_values"]
    fn = result["feature_names"]
    prob = result["default_probability"]
    method = result.get("explanation_method", "feature contribution")

    sorted_idx = np.argsort(np.abs(sv))[::-1][:12]
    top_feat = [fn[i] for i in sorted_idx]
    top_sv   = [sv[i] for i in sorted_idx]
    colors   = ["#e74c3c" if v > 0 else "#2980b9" for v in top_sv]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(top_feat))
    bars = ax.barh(y, top_sv, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(top_feat, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Feature Contribution (Red = ↑ Default Risk | Blue = ↓ Default Risk)")
    ax.set_title(f"Decision Explanation — {customer_id}\nDefault Probability: {prob:.1%} | {result['risk_label']}")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, top_sv):
        if abs(val) > 1e-6:
            ax.text(val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height()/2,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8)
    plt.tight_layout()
    path = f"{REPORTS_DIR}/shap_local_{customer_id}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[explain] Saved: {path}")


def run_explainability():
    print("=" * 60)
    print("EXPLAINABILITY PIPELINE")
    print("=" * 60)
    model, prep, model_name = load_model_and_artifacts()
    val = pd.read_csv("data/processed/val.csv", parse_dates=[DATE_COL])
    X_val = val[prep["feature_cols"]]

    imp = plot_global_importance(model, X_val, model_name)
    print("\nTop 5 Global Risk Drivers:")
    print(imp.tail(5).sort_values(ascending=False).to_string())

    example = {
        "age": 32, "annual_income": 450000, "employment_status": "Self-Employed",
        "employment_years": 2, "loan_amount": 300000, "loan_tenure_months": 36,
        "interest_rate": 16.5, "loan_purpose": "Personal", "credit_score": 610,
        "credit_history_length": 3.0, "past_defaults": 1, "num_open_accounts": 5,
        "num_credit_inquiries": 4, "debt_to_income": 0.35, "loan_to_income": 0.67,
    }
    result = predict_single(example)
    print(f"\nExample: P(default)={result['default_probability']:.1%} | {result['risk_label']}")
    print(f"Explanation method: {result['explanation_method']}")
    plot_local_explanation(result, "EXAMPLE_CUST_001")
    print("\n✅ Explainability complete.")


if __name__ == "__main__":
    run_explainability()
