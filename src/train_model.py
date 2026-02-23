"""
Model Training Pipeline
------------------------
Trains and compares:
  1. Logistic Regression (interpretable baseline)
  2. XGBoost (preferred) / GradientBoostingClassifier (fallback)
"""
import os, pickle, warnings
from datetime import datetime
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[warn] XGBoost not found → using GradientBoostingClassifier")

warnings.filterwarnings("ignore")
PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "src/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
TARGET = "loan_default"
DATE_COL = "application_date"


def load_data():
    train = pd.read_csv(f"{PROCESSED_DIR}/train.csv", parse_dates=[DATE_COL])
    val   = pd.read_csv(f"{PROCESSED_DIR}/val.csv",   parse_dates=[DATE_COL])
    with open(f"{ARTIFACTS_DIR}/preprocessing.pkl", "rb") as f:
        arts = pickle.load(f)
    fc = arts["feature_cols"]
    X_tr, y_tr = train[fc], train[TARGET]
    X_va, y_va = val[fc],   val[TARGET]
    print(f"[load] Train {X_tr.shape} | Val {X_va.shape}")
    print(f"[load] Default rate — Train: {y_tr.mean():.2%} | Val: {y_va.mean():.2%}")
    return X_tr, y_tr, X_va, y_va, fc


def train_logistic_regression(X_train, y_train):
    print("\n" + "=" * 50)
    print("TRAINING: Logistic Regression (Baseline)")
    print("=" * 50)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_c, best_score = 0.1, 0
    for c in [0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=42)
        sc = cross_val_score(lr, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  C={c:.2f} → CV AUC: {sc.mean():.4f} (±{sc.std():.4f})")
        if sc.mean() > best_score:
            best_score, best_c = sc.mean(), c
    print(f"\n  Best C: {best_c} | CV AUC: {best_score:.4f}")
    model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    label = "XGBoost" if XGBOOST_AVAILABLE else "GradientBoosting"
    print(f"\n{'=' * 50}")
    print(f"TRAINING: {label} (Primary Model)")
    print("=" * 50)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos
    print(f"  scale_pos_weight: {spw:.2f}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if XGBOOST_AVAILABLE:
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1, 0.15],
            "subsample": [0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_alpha": [0, 0.1, 0.5],
        }
        base = XGBClassifier(scale_pos_weight=spw, objective="binary:logistic",
                             eval_metric="auc", random_state=42, n_jobs=-1, verbosity=0)
        fit_params = {}
    else:
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1, 0.15],
            "subsample": [0.7, 0.8, 0.9],
            "min_samples_leaf": [10, 20, 30],
            "max_features": ["sqrt", 0.8, 1.0],
        }
        base = GradientBoostingClassifier(random_state=42)
        fit_params = {"sample_weight": np.where(y_train == 1, spw, 1.0)}

    search = RandomizedSearchCV(base, param_dist, n_iter=20, scoring="roc_auc",
                                cv=cv, random_state=42, n_jobs=-1, verbose=1, refit=True)
    if fit_params:
        search.fit(X_train, y_train, **fit_params)
    else:
        search.fit(X_train, y_train)

    print(f"\n  Best params: {search.best_params_}")
    print(f"  Best CV AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def compare_models(models, X_val, y_val):
    print("\n" + "=" * 50)
    print("MODEL COMPARISON — VALIDATION SET")
    print("=" * 50)
    results = {}
    for name, model in models.items():
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        results[name] = auc
        print(f"  {name:35s} → AUC: {auc:.4f}")
    best = max(results, key=results.get)
    print(f"\n  ✅ Best: {best} (AUC={results[best]:.4f})")
    return best, results


def save_models(models, best_model_name):
    for name, model in models.items():
        path = f"{ARTIFACTS_DIR}/model_{name.lower().replace(' ','_')}.pkl"
        pickle.dump(model, open(path, "wb"))
        print(f"  Saved: {path}")
    meta = {"best_model": best_model_name, "model_names": list(models.keys()),
            "trained_at": datetime.now().isoformat(), "xgboost_available": XGBOOST_AVAILABLE}
    pickle.dump(meta, open(f"{ARTIFACTS_DIR}/model_metadata.pkl", "wb"))
    print(f"  Saved: {ARTIFACTS_DIR}/model_metadata.pkl")


def run_training():
    print("=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)
    X_tr, y_tr, X_va, y_va, fc = load_data()
    lr  = train_logistic_regression(X_tr, y_tr)
    gb  = train_gradient_boosting(X_tr, y_tr)
    label = "XGBoost" if XGBOOST_AVAILABLE else "GradientBoosting"
    models = {"Logistic Regression": lr, label: gb}
    best, results = compare_models(models, X_va, y_va)
    print("\n[save] Saving models...")
    save_models(models, best)
    print("\n✅ Training complete.")
    return models, best, fc


if __name__ == "__main__":
    run_training()
