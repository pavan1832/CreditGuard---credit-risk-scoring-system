"""
Synthetic Fintech Loan Dataset Generator
Mimics real-world loan application data with realistic distributions
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
N = 15000  # 15k records — realistic, under 50MB


def generate_dataset(n=N):
    # --- Demographic & Financial Features ---
    age = np.random.normal(35, 10, n).clip(21, 65).astype(int)
    annual_income = np.random.lognormal(mean=11.0, sigma=0.5, size=n).clip(100000, 5000000).astype(int)
    employment_status = np.random.choice(
        ["Salaried", "Self-Employed", "Business", "Unemployed"],
        p=[0.55, 0.25, 0.15, 0.05],
        size=n,
    )
    employment_years = np.where(
        employment_status == "Unemployed",
        0,
        np.random.exponential(scale=5, size=n).clip(0, 35).astype(int),
    )

    # --- Loan Features ---
    loan_amount = np.random.lognormal(mean=12.5, sigma=0.6, size=n).clip(10000, 2000000).astype(int)
    loan_tenure_months = np.random.choice([12, 24, 36, 48, 60, 84], p=[0.05, 0.15, 0.30, 0.25, 0.20, 0.05], size=n)
    interest_rate = np.random.normal(12.5, 3.5, n).clip(7.0, 28.0).round(2)
    loan_purpose = np.random.choice(
        ["Home", "Education", "Business", "Vehicle", "Personal", "Medical"],
        p=[0.25, 0.15, 0.20, 0.15, 0.20, 0.05],
        size=n,
    )

    # --- Credit History ---
    credit_history_length = np.random.exponential(scale=5, size=n).clip(0, 25).round(1)
    credit_score = np.random.normal(680, 80, n).clip(300, 900).astype(int)
    past_defaults = np.random.choice([0, 1, 2, 3], p=[0.70, 0.18, 0.08, 0.04], size=n)
    num_open_accounts = np.random.poisson(lam=4, size=n).clip(0, 15)
    num_credit_inquiries = np.random.poisson(lam=2, size=n).clip(0, 10)

    # --- Derived Financial Ratios ---
    monthly_income = annual_income / 12
    emi = (loan_amount * (interest_rate / 1200) * (1 + interest_rate / 1200) ** loan_tenure_months) / \
          ((1 + interest_rate / 1200) ** loan_tenure_months - 1)
    debt_to_income = (emi / monthly_income).round(4)

    # --- Loan-to-Income Ratio ---
    loan_to_income = (loan_amount / annual_income).round(4)

    # --- Application date (for time-based split) ---
    start_date = datetime(2021, 1, 1)
    application_date = [start_date + timedelta(days=int(x)) for x in np.random.uniform(0, 730, n)]
    application_date = sorted(application_date)  # chronological order

    # --- Target Variable: Loan Default ---
    # Build a realistic default probability using logistic function
    log_odds = (
        -4.5
        + 0.015 * (past_defaults * 2.5)
        + 0.008 * debt_to_income * 10
        - 0.004 * (credit_score - 600) / 10
        - 0.003 * credit_history_length
        + 0.002 * num_credit_inquiries
        + 0.05 * (employment_status == "Unemployed").astype(int) * 2
        + 0.003 * loan_to_income * 5
        + 0.001 * (interest_rate - 10)
        - 0.002 * (employment_years)
    )
    default_prob = 1 / (1 + np.exp(-log_odds))
    # Add noise
    default_prob = default_prob.clip(0.02, 0.85)
    loan_default = np.random.binomial(1, default_prob, n)

    df = pd.DataFrame({
        "application_date": application_date,
        "age": age,
        "annual_income": annual_income,
        "employment_status": employment_status,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_tenure_months": loan_tenure_months,
        "interest_rate": interest_rate,
        "loan_purpose": loan_purpose,
        "credit_score": credit_score,
        "credit_history_length": credit_history_length,
        "past_defaults": past_defaults,
        "num_open_accounts": num_open_accounts,
        "num_credit_inquiries": num_credit_inquiries,
        "debt_to_income": debt_to_income,
        "loan_to_income": loan_to_income,
        "loan_default": loan_default,
    })

    # Introduce realistic missing values
    for col, rate in [("credit_history_length", 0.04), ("employment_years", 0.03), ("num_credit_inquiries", 0.02)]:
        mask = np.random.rand(n) < rate
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/raw/loan_data.csv", index=False)
    print(f"Dataset generated: {df.shape}")
    print(f"Default rate: {df['loan_default'].mean():.2%}")
    print(df.dtypes)
