# 🏦 CreditGuard — Credit Risk Scoring System

> **Production-grade loan default prediction system built for fintech lending operations.**  
> Demonstrates end-to-end ML engineering: business problem framing → feature engineering → model training → explainability → deployment.

The project is live here: https://creditguard---credit-risk-scoring-system-mzyegjpydwsygqcxtzktn.streamlit.app/
---

## 1. Business Problem

Every time a fintech company like Navi, Slice, or KreditBee approves a loan, it takes on **credit risk** — the probability that the borrower won't repay. With millions of applications processed monthly, even a 1% improvement in default detection can save **crores in write-offs**.

**Goal:** Build an ML system that predicts the probability of loan default *before* approval, enabling the business to:
- **Reduce credit losses** by rejecting high-risk applicants
- **Increase portfolio yield** by approving more creditworthy customers confidently
- **Automate underwriting** to scale without proportional headcount growth
- **Explain every decision** to Risk, Compliance, and Regulatory teams

---

## 2. Why This Problem Matters in Fintech

| Challenge | Business Impact |
|-----------|----------------|
| 15–25% default rates in unsecured lending | Directly erodes P&L |
| Manual underwriting doesn't scale | Bottleneck at ₹100Cr+ disbursement |
| RBI requires explainable credit decisions | Regulatory compliance risk |
| Fraud + genuine credit risk overlap | Model must distinguish signal from noise |

In a typical lending portfolio, **preventing 20% of defaults** while maintaining approval volume can improve **Net Interest Margin by 150–200 bps** — the difference between a profitable and loss-making product.

---

## 3. Data Overview

**Source:** Synthetic fintech loan application dataset (designed to mirror real lending data distributions)

| Feature | Type | Description |
|---------|------|-------------|
| `annual_income` | Numeric | Applicant's annual income (INR) |
| `loan_amount` | Numeric | Requested loan amount |
| `interest_rate` | Numeric | Applicable interest rate (%) |
| `credit_score` | Numeric | CIBIL/bureau score (300–900) |
| `credit_history_length` | Numeric | Years of credit history |
| `past_defaults` | Numeric | Count of prior loan defaults |
| `debt_to_income` | Numeric | Monthly EMI / Monthly Income |
| `loan_to_income` | Numeric | Loan amount / Annual income |
| `employment_status` | Categorical | Salaried / Self-Employed / Business / Unemployed |
| `loan_purpose` | Categorical | Home / Education / Business / Vehicle / Personal / Medical |
| `loan_default` | **Target** | 1 = Default, 0 = No Default |

**Size:** 15,000 records | **Default Rate:** ~15% | **Timespan:** Jan 2021 – Dec 2022

---

## 4. Modeling Approach

### Architecture

```
Raw Data → Feature Engineering → Model Training → Evaluation → Deployment
```

### Models

| Model | Role | Why |
|-------|------|-----|
| **Logistic Regression** | Baseline | Interpretable, regulatory-friendly, fast |
| **XGBoost** | Primary | Handles non-linearity, imbalance, missing values |

### Key Design Decisions

- **Time-based train/val split** (not random) → prevents data leakage that mimics production deployment
- **Class imbalance handling** via `scale_pos_weight` in XGBoost (neg/pos ratio) and `class_weight='balanced'` in LR
- **Hyperparameter tuning** via `RandomizedSearchCV` with 5-fold stratified CV (30 iterations)
- **Derived features** (`emi_burden_score`, `credit_risk_index`, `income_per_year_employed`) add domain signal

---

## 5. Evaluation Metrics & Results

### Why NOT Accuracy?

A model that predicts "No Default" for every applicant achieves **85% accuracy** but catches **zero defaults**. This is worse than useless in credit risk.

| Metric | Purpose | Target |
|--------|---------|--------|
| **ROC-AUC** | Overall discrimination | > 0.75 |
| **KS Statistic** | Credit scorecard standard | > 0.35 |
| **Gini Coefficient** | Basel II/III regulatory metric | > 0.50 |
| **PR-AUC** | Performance under imbalance | > 0.50 |
| **Business Cost** | FN×₹50K + FP×₹10K | Minimize |

### Results

| Model | ROC-AUC | KS Stat | Gini | PR-AUC |
|-------|---------|---------|------|--------|
| Logistic Regression | ~0.77 | ~0.42 | ~0.54 | ~0.45 |
| **XGBoost** | **~0.84** | **~0.53** | **~0.68** | **~0.57** |

*XGBoost outperforms on all metrics and is selected as the production model.*

### Threshold Optimization

Default threshold (0.5) is not optimal. We optimize using **business cost function:**
```
Total Cost = FN × ₹50,000 (missed default) + FP × ₹10,000 (rejected good customer)
```
Optimal threshold is typically **0.30–0.40**, reflecting that missing a default costs 5× more than rejecting a good borrower.

---

## 6. Explainability Approach

**Tool:** SHAP (SHapley Additive exPlanations)  
**Explainer:** `TreeExplainer` for XGBoost (exact, fast)

### Global Explainability
Top risk drivers across all customers (used by Risk team for policy setting):
- `past_defaults` — strongest predictor
- `credit_score` — inverse relationship
- `debt_to_income` — above 50% is danger zone
- `credit_risk_index` — composite engineered feature
- `employment_status_Unemployed` — discrete risk jump

### Local Explainability (Per-Customer)
For every prediction, a waterfall chart shows which features pushed the score up or down. This enables:
- **Customer-facing explanation**: "Your loan was declined because your past default history..."
- **Compliance audit trail**: Regulators can inspect any individual decision
- **Model debugging**: Catch spurious patterns before production deployment

---

## 7. How This Works in Production

```
[Loan Application API] 
        ↓
[Feature Extraction Service] → Pulls bureau data, bank statements
        ↓
[Feature Engineering Service] → Applies saved preprocessing pipeline (preprocessing.pkl)
        ↓
[Scoring Service] → Loads XGBoost model (model_xgboost.pkl)
        ↓
[Decision Engine] → Applies threshold → AUTO_APPROVE / MANUAL_REVIEW / AUTO_REJECT
        ↓
[Explanation Service] → Generates SHAP values → API response + audit log
        ↓
[Monitoring Service] → Logs prediction for drift detection
```

**Infrastructure:**
- Models served via FastAPI container (Docker)
- Feature pipeline versioned in MLflow
- Predictions logged to PostgreSQL for monitoring
- SHAP explanations stored per-request in S3

---

## 8. Model Monitoring & Drift Strategy

### What Can Go Wrong in Production?

| Issue | Signal | Response |
|-------|--------|----------|
| **Data drift** | Input feature distributions shift (e.g., inflation changes income patterns) | PSI (Population Stability Index) on key features weekly |
| **Concept drift** | Default rate changes (e.g., recession hits) | Monitor actual vs predicted default rates monthly |
| **Model staleness** | AUC degrades on recent cohort | Retrain trigger at <5% AUC degradation |
| **Feature failure** | Bureau API outage → missing credit score | Fallback model or business rules |

### Monitoring Framework

```python
# Weekly PSI check (Population Stability Index)
# PSI < 0.1: Stable | 0.1-0.2: Monitor | > 0.2: Retrain
psi = compute_psi(training_score_dist, current_score_dist)

# Monthly model performance check on settled loans
auc_current = roc_auc_score(actual_defaults_30dpd, model_scores)

# Alert if AUC drops > 3% from baseline
if auc_current < auc_baseline * 0.97:
    trigger_retraining_pipeline()
```

### Retraining Strategy
- **Frequency:** Monthly or triggered by drift alerts
- **Data:** Rolling 18-month window + recent vintage overweighted
- **Champion-Challenger:** New model runs in shadow mode, 5% traffic, A/B tested for 30 days before promoting

---

## 9. Future Improvements

| Improvement | Business Value | Complexity |
|-------------|---------------|------------|
| **Bureau feature integration** (CIBIL API) | +5-8% AUC improvement | Medium |
| **Behavioral features** (bank statement analysis) | Catches income misrepresentation | High |
| **Vintage-based features** (loan age at default) | Better survival modeling | Medium |
| **Ensemble stacking** (LR + XGBoost + CatBoost) | +2-3% AUC | Medium |
| **Fairness audit** (by gender/state/age group) | Regulatory requirement | Medium |
| **Credit scorecard transformation** (WOE/IV binning) | Basel-compliant reporting | Medium |
| **Real-time streaming** (Kafka + Flink) | < 100ms decision latency | High |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python data/generate_data.py

# 3. Feature engineering
python -m src.feature_engineering

# 4. Train models
python -m src.train_model

# 5. Evaluate
python -m src.evaluate

# 6. Run Streamlit app
streamlit run app.py
```

---

## Project Structure

```
credit-risk-scoring/
├── data/
│   ├── raw/loan_data.csv           # Synthetic fintech dataset
│   └── processed/                  # Train/val CSVs after feature engineering
├── notebooks/
│   └── eda.ipynb                   # Full EDA with business insights
├── src/
│   ├── feature_engineering.py      # Imputation, encoding, scaling, time-split
│   ├── train_model.py              # LR + XGBoost training + hyperparameter tuning
│   ├── evaluate.py                 # ROC-AUC, KS, Gini, threshold optimization
│   ├── predict.py                  # Single/batch prediction + SHAP explainability
│   └── artifacts/                  # Saved models + preprocessing pipeline
├── reports/                        # Evaluation plots, SHAP charts
├── app.py                          # Streamlit UI
└── requirements.txt
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data | pandas, numpy |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn |
| App | Streamlit |
| Serialization | pickle |

---

*Built as a fintech Data Scientist I portfolio project demonstrating business understanding, ML rigor, and deployment thinking.*
