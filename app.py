"""
CreditGuard — Credit Risk Scoring Streamlit App
"""
import os, sys, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_single, get_risk_label

st.set_page_config(
    page_title="CreditGuard — Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.risk-low    {background:#d4edda;color:#155724;padding:14px;border-radius:8px;font-size:20px;font-weight:bold;text-align:center}
.risk-medium {background:#fff3cd;color:#856404;padding:14px;border-radius:8px;font-size:20px;font-weight:bold;text-align:center}
.risk-high   {background:#f8d7da;color:#721c24;padding:14px;border-radius:8px;font-size:20px;font-weight:bold;text-align:center}
</style>
""", unsafe_allow_html=True)

st.title("🏦 CreditGuard — Loan Default Risk Scoring")
st.title("Developed By Lokpavan P")
st.markdown("**Production-grade credit risk scoring** | Fintech lending operations")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("📋 Customer Application")
st.sidebar.subheader("Personal Information")
age = st.sidebar.slider("Age", 21, 65, 34)
annual_income = st.sidebar.number_input("Annual Income (₹)", 100000, 5000000, 480000, 10000)
employment_status = st.sidebar.selectbox("Employment Status", ["Salaried","Self-Employed","Business","Unemployed"])
employment_years = st.sidebar.slider("Years of Employment", 0, 35, 5)

st.sidebar.subheader("Loan Details")
loan_amount = st.sidebar.number_input("Loan Amount (₹)", 10000, 2000000, 250000, 10000)
loan_tenure_months = st.sidebar.selectbox("Loan Tenure (Months)", [12,24,36,48,60,84], index=2)
interest_rate = st.sidebar.slider("Interest Rate (%)", 7.0, 28.0, 12.5, 0.5)
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Home","Education","Business","Vehicle","Personal","Medical"])

st.sidebar.subheader("Credit Profile")
credit_score = st.sidebar.slider("Credit Score (CIBIL)", 300, 900, 680)
credit_history_length = st.sidebar.slider("Credit History (Years)", 0.0, 25.0, 5.0, 0.5)
past_defaults = st.sidebar.selectbox("Past Defaults", [0,1,2,3], index=0)
num_open_accounts = st.sidebar.slider("Open Accounts", 0, 15, 3)
num_credit_inquiries = st.sidebar.slider("Credit Inquiries (6M)", 0, 10, 2)

predict_btn = st.sidebar.button("🔍 Assess Credit Risk", use_container_width=True, type="primary")

# Computed preview metrics
monthly_income = annual_income / 12
emi = (loan_amount * (interest_rate/1200) * (1+interest_rate/1200)**loan_tenure_months) / ((1+interest_rate/1200)**loan_tenure_months - 1)
dti = min(emi / monthly_income, 3.0)
lti = min(loan_amount / annual_income, 5.0)

col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.subheader("📊 Loan Summary")
    summary = pd.DataFrame({
        "Metric": ["Monthly Income", "Estimated EMI", "Debt-to-Income", "Loan-to-Income"],
        "Value": [f"₹{monthly_income:,.0f}", f"₹{emi:,.0f}", f"{dti:.2%}", f"{lti:.2f}x"],
        "Flag": ["✅" if monthly_income > 30000 else "⚠️",
                 "✅" if dti < 0.4 else "⚠️",
                 "✅" if dti < 0.4 else "🔴",
                 "✅" if lti < 3 else "🔴"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("ℹ️ Risk Tiers")
    st.info("""
**🟢 LOW** (< 20%) — Auto-approve | Standard terms  
**🟡 MEDIUM** (20–45%) — Manual underwriter review  
**🔴 HIGH** (> 45%) — Auto-reject | Escalate to senior credit officer

*Thresholds configurable per portfolio risk appetite.*
    """)

with col_r:
    if predict_btn:
        customer = {
            "age": age, "annual_income": annual_income,
            "employment_status": employment_status, "employment_years": employment_years,
            "loan_amount": loan_amount, "loan_tenure_months": loan_tenure_months,
            "interest_rate": interest_rate, "loan_purpose": loan_purpose,
            "credit_score": credit_score, "credit_history_length": credit_history_length,
            "past_defaults": past_defaults, "num_open_accounts": num_open_accounts,
            "num_credit_inquiries": num_credit_inquiries,
            "debt_to_income": round(dti, 4), "loan_to_income": round(lti, 4),
        }
        with st.spinner("Running credit risk model..."):
            try:
                result = predict_single(customer)
                prob = result["default_probability"]
                risk = result["risk_label"]

                st.subheader("🎯 Prediction Result")
                st.metric("Default Probability", f"{prob:.1%}")
                st.progress(min(prob, 1.0))

                cls = "risk-low" if "LOW" in risk else ("risk-medium" if "MEDIUM" in risk else "risk-high")
                st.markdown(f'<div class="{cls}">{risk}</div>', unsafe_allow_html=True)
                st.markdown("")

                if "LOW" in risk:
                    st.success("✅ **Recommendation:** Approve at standard terms")
                elif "MEDIUM" in risk:
                    st.warning("⚠️ **Recommendation:** Refer to underwriter for manual review")
                else:
                    st.error("❌ **Recommendation:** Reject or escalate to senior officer")

                st.caption(f"Model: {result['model_name']}")

                # Feature contribution chart
                method = result.get("explanation_method", "feature contribution")
                st.subheader("🔍 Decision Explanation")
                st.caption(f"Method: *{method}*")

                sv = np.array(result["shap_values"], dtype=float)
                fn = result["feature_names"]

                # Filter to features with meaningful contribution (avoid clutter)
                threshold = np.abs(sv).max() * 0.02  # at least 2% of max
                mask = np.abs(sv) >= threshold
                sv_filtered = sv[mask]
                fn_filtered = [fn[i] for i, m in enumerate(mask) if m]

                # Sort by absolute value, take top 12
                idx = np.argsort(np.abs(sv_filtered))[::-1][:12]
                top_f = [fn_filtered[i] for i in idx]
                top_v = [float(sv_filtered[i]) for i in idx]

                # Reverse for top-to-bottom display
                top_f = top_f[::-1]
                top_v = top_v[::-1]
                colors = ["#e74c3c" if v > 0 else "#2980b9" for v in top_v]

                fig, ax = plt.subplots(figsize=(8, max(4, len(top_f) * 0.45)))
                y = np.arange(len(top_f))
                bars = ax.barh(y, top_v, color=colors, alpha=0.85, edgecolor="white", height=0.6)
                ax.set_yticks(y)
                ax.set_yticklabels(top_f, fontsize=9)
                ax.axvline(0, color="black", lw=1.0)
                ax.set_xlabel("Feature Contribution  (Red = ↑ Default Risk | Blue = ↓ Default Risk)", fontsize=9)
                ax.set_title("Risk Score Breakdown — Top Drivers", fontweight="bold", fontsize=11)
                ax.grid(True, alpha=0.3, axis="x")

                x_range = max(abs(v) for v in top_v) if top_v else 1
                offset = x_range * 0.03
                for bar, val in zip(bars, top_v):
                    ax.text(
                        val + (offset if val >= 0 else -offset),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}",
                        va="center",
                        ha="left" if val >= 0 else "right",
                        fontsize=8,
                        fontweight="bold",
                        color="#333333",
                    )

                ax.set_xlim(-(x_range * 1.35), x_range * 1.35)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.caption(
                    "Each bar shows how much a feature pushed the risk score **up** (🔴 red) "
                    "or **down** (🔵 blue). Positive = increases default probability. "
                    "Satisfies RBI explainability requirements for credit decisions."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Ensure training pipeline has been run: `python -m src.feature_engineering && python -m src.train_model`")
    else:
        st.subheader("👈 Fill in customer details and click **Assess Credit Risk**")
        st.markdown("""
        **How this works:**
        1. Enter customer application details in the sidebar
        2. Click **Assess Credit Risk**
        3. Get real-time default probability + risk tier
        4. Review AI-driven feature explanation for the decision

        ---
        **Built with:** GradientBoosting • SHAP • Streamlit • scikit-learn  
        **Compliant with:** RBI explainability guidelines | Fair lending principles
        """)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#888;font-size:12px;'>CreditGuard v1.0 | Internal use only | All decisions subject to Fair Lending compliance review</p>", unsafe_allow_html=True)
