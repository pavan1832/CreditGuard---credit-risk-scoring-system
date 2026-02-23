"""
Model Evaluation Module
------------------------
Business-oriented evaluation framework for credit risk models.

Why accuracy is NOT the right metric:
- At 15% default rate, a model predicting "no default" for everyone achieves 85% accuracy
- But it catches ZERO defaults → massive credit losses for the business
- We care about ROC-AUC, KS Statistic, and business-cost-optimized threshold

Metrics used:
  ROC-AUC    → Overall discrimination power (threshold-agnostic)
  PR-AUC     → Performance under class imbalance (more informative than ROC for rare events)
  KS Stat    → Industry standard for credit scorecards (regulatory compliance)
  Gini       → Common in Basel II/III credit risk frameworks (= 2*AUC - 1)
  Threshold  → Optimized for business cost (asymmetric: FN >> FP)
"""

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "src/artifacts"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET = "loan_default"
DATE_COL = "application_date"

# Business cost assumptions (INR)
# A missed default (FN) costs the business ~5x more than a wrongly rejected applicant (FP)
COST_FN = 50000   # Expected loss from one default (loan write-off)
COST_FP = 10000   # Opportunity cost: lost interest income from rejected good borrower


def load_artifacts():
    with open(f"{ARTIFACTS_DIR}/preprocessing.pkl", "rb") as f:
        prep = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/model_metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    models = {}
    for name in meta["model_names"]:
        safe_name = name.lower().replace(" ", "_")
        with open(f"{ARTIFACTS_DIR}/model_{safe_name}.pkl", "rb") as f:
            models[name] = pickle.load(f)

    val = pd.read_csv(f"{PROCESSED_DIR}/val.csv", parse_dates=[DATE_COL])
    feature_cols = prep["feature_cols"]

    return models, val, feature_cols, meta


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    KS Statistic = max separation between cumulative TPR and FPR curves.
    Standard credit scorecard metric. Desired: KS > 0.3 is good, > 0.5 is excellent.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    return ks


def compute_gini(auc: float) -> float:
    """Gini = 2 * AUC - 1. Used in Basel II credit risk modeling."""
    return 2 * auc - 1


def find_optimal_threshold_cost(y_true, y_prob, cost_fn=COST_FN, cost_fp=COST_FP) -> tuple:
    """
    Find threshold that minimizes total business cost.
    
    Total Cost = FN * COST_FN + FP * COST_FP
    
    This is why pure accuracy is wrong: the costs are asymmetric.
    The business must explicitly decide the trade-off between:
      - Approving risky loans (FN) → credit loss
      - Rejecting safe loans (FP) → opportunity cost
    """
    thresholds = np.linspace(0.1, 0.9, 100)
    best_thresh = 0.5
    best_cost = np.inf

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = fn * cost_fn + fp * cost_fp
        if total_cost < best_cost:
            best_cost = total_cost
            best_thresh = thresh

    return best_thresh, best_cost


def evaluate_model(model, X_val, y_val, model_name: str) -> dict:
    """Full business-oriented evaluation of a single model."""
    y_prob = model.predict_proba(X_val)[:, 1]

    # Core metrics
    auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    ks = compute_ks_statistic(y_val.values, y_prob)
    gini = compute_gini(auc)

    # Threshold optimization
    opt_thresh, opt_cost = find_optimal_threshold_cost(y_val.values, y_prob)
    y_pred_opt = (y_prob >= opt_thresh).astype(int)
    y_pred_default = (y_prob >= 0.5).astype(int)

    # Confusion matrix at optimal threshold
    cm = confusion_matrix(y_val, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "model_name": model_name,
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "ks_statistic": round(ks, 4),
        "gini": round(gini, 4),
        "optimal_threshold": round(opt_thresh, 3),
        "optimal_business_cost": int(opt_cost),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "precision": round(tp / (tp + fp + 1e-9), 4),
        "recall": round(tp / (tp + fn + 1e-9), 4),
        "y_prob": y_prob,
        "y_true": y_val.values,
        "classification_report": classification_report(y_val, y_pred_opt),
    }

    return results


def print_results(results: dict):
    print(f"\n{'=' * 55}")
    print(f"EVALUATION: {results['model_name']}")
    print(f"{'=' * 55}")
    print(f"  ROC-AUC          : {results['roc_auc']:.4f}")
    print(f"  PR-AUC           : {results['pr_auc']:.4f}")
    print(f"  KS Statistic     : {results['ks_statistic']:.4f}  (>0.3 = good, >0.5 = excellent)")
    print(f"  Gini Coefficient : {results['gini']:.4f}")
    print(f"  Optimal Threshold: {results['optimal_threshold']:.3f}")
    print(f"  Est. Business Cost: ₹{results['optimal_business_cost']:,}")
    print(f"\n  Confusion Matrix @ Optimal Threshold:")
    print(f"    TP={results['tp']} FP={results['fp']}")
    print(f"    FN={results['fn']} TN={results['tn']}")
    print(f"\n  Precision: {results['precision']:.4f} | Recall: {results['recall']:.4f}")
    print(f"\n  Classification Report:")
    print(results["classification_report"])


def plot_roc_curves(all_results: list):
    """Plot ROC curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC curves
    ax = axes[0]
    for res in all_results:
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, lw=2, label=f"{res['model_name']} (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Credit Default Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Precision-Recall curves
    ax = axes[1]
    for res in all_results:
        precision, recall, _ = precision_recall_curve(res["y_true"], res["y_prob"])
        ax.plot(recall, precision, lw=2, label=f"{res['model_name']} (PR-AUC={res['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Better for Imbalanced Classes)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{REPORTS_DIR}/roc_pr_curves.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {path}")


def plot_ks_curve(results: dict):
    """KS Statistic visualization — cumulative default/non-default distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sorted_idx = np.argsort(results["y_prob"])[::-1]
    y_sorted = results["y_true"][sorted_idx]
    n = len(y_sorted)

    cum_pos = np.cumsum(y_sorted) / y_sorted.sum()
    cum_neg = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    pct_pop = np.arange(1, n + 1) / n

    ax.plot(pct_pop, cum_pos, label="Cumulative Defaults", color="red", lw=2)
    ax.plot(pct_pop, cum_neg, label="Cumulative Non-Defaults", color="blue", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    ks_val = results["ks_statistic"]
    ax.set_title(f"KS Curve — {results['model_name']} | KS = {ks_val:.4f}")
    ax.set_xlabel("Population % (ranked by risk score)")
    ax.set_ylabel("Cumulative %")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = f"{REPORTS_DIR}/ks_curve_{results['model_name'].lower().replace(' ','_')}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {path}")


def plot_score_distribution(results: dict):
    """Distribution of predicted default probabilities by actual class."""
    fig, ax = plt.subplots(figsize=(9, 5))

    defaults = results["y_prob"][results["y_true"] == 1]
    non_defaults = results["y_prob"][results["y_true"] == 0]

    ax.hist(non_defaults, bins=50, alpha=0.6, color="steelblue", label="Non-Default (0)", density=True)
    ax.hist(defaults, bins=50, alpha=0.6, color="crimson", label="Default (1)", density=True)
    ax.axvline(results["optimal_threshold"], color="orange", lw=2, linestyle="--",
               label=f"Optimal Threshold = {results['optimal_threshold']:.3f}")
    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {results['model_name']}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = f"{REPORTS_DIR}/score_distribution_{results['model_name'].lower().replace(' ','_')}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {path}")


def why_not_accuracy():
    """Print the business case for AUC over accuracy."""
    print("\n" + "=" * 60)
    print("WHY ACCURACY IS NOT THE RIGHT METRIC FOR CREDIT RISK")
    print("=" * 60)
    print("""
    Problem: Dataset has ~15% defaults (class imbalance).
    
    A naive model that predicts 'No Default' for EVERY applicant:
      ✓ Achieves 85% accuracy
      ✗ Catches ZERO actual defaults
      ✗ Results in 100% of credit losses passing through
    
    What we actually care about:
      ROC-AUC  → Rank-orders customers by risk (threshold-agnostic)
      KS Stat  → Measures separation between good/bad borrowers
      PR-AUC   → Precision vs Recall trade-off under imbalance
      Threshold → Set based on BUSINESS COST, not 0.5 default
    
    In production:
      - Risk team sets threshold based on portfolio loss appetite
      - Regulators (RBI/SEBI) require explainable, fair decisions
      - We optimize F1 or cost-weighted metric, not accuracy
    """)


def run_evaluation():
    print("=" * 60)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 60)

    why_not_accuracy()

    models, val, feature_cols, meta = load_artifacts()
    X_val = val[feature_cols]
    y_val = val[TARGET]

    all_results = []
    for name, model in models.items():
        res = evaluate_model(model, X_val, y_val, name)
        print_results(res)
        all_results.append(res)
        plot_ks_curve(res)
        plot_score_distribution(res)

    plot_roc_curves(all_results)

    # Save summary
    summary = pd.DataFrame([{
        k: v for k, v in r.items()
        if k not in ["y_prob", "y_true", "classification_report"]
    } for r in all_results])
    summary.to_csv(f"{REPORTS_DIR}/evaluation_summary.csv", index=False)
    print(f"\n[save] Saved: {REPORTS_DIR}/evaluation_summary.csv")

    print("\n✅ Evaluation complete.")
    return all_results


if __name__ == "__main__":
    run_evaluation()
