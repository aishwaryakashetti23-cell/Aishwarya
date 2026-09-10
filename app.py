import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)

# =========================================================
# LOGIN
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.markdown(
        "<h1 style='text-align:center;'>💰 Loan Default Prediction</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;'>Secure access to the Loan Risk Assessment System</p>",
        unsafe_allow_html=True
    )

    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.subheader("🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        st.info("Demo login: username `admin` and password `loan123`")

        if st.button("Login", type="primary"):
            if username == "admin" and password == "loan123":
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.caption("🎓 AI & Data Science Internship Project | Educational use only")
    st.stop()


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("loan_default_model.pkl")

# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 25px;
}

.card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #dddddd;
    margin-bottom: 15px;
}

.result-card {
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #dddddd;
}

.big-number {
    font-size: 32px;
    font-weight: 700;
}

.small-note {
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<div class='main-title'>💰 Loan Default Prediction System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-powered assessment of loan default risk</div>",
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")

st.sidebar.write("👤 Logged in as: **Admin**")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.write("---")

st.sidebar.subheader("🤖 Model Information")
st.sidebar.write(
    """
**Algorithm:** Logistic Regression

**Target:**
- 0 → Fully Paid
- 1 → Not Fully Paid / Default

**Evaluation:**
- Accuracy: 64.09%
- Precision: 24.01%
- Recall: 57.33%
- F1 Score: 33.85%
- ROC-AUC: 68.44%
"""
)

st.sidebar.write("---")
st.sidebar.info(
    "This application is intended for educational/project purposes "
    "and should not be used as the sole basis for real financial decisions."
)

# =========================================================
# PROJECT OVERVIEW
# =========================================================

with st.expander("ℹ️ About the Application", expanded=True):

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🤖 Model", "Logistic Regression")

    with c2:
        st.metric("🎯 Prediction", "Loan Default")

    with c3:
        st.metric("📊 Features", "13")

    with c4:
        st.metric("⚡ Type", "Classification")

    st.write(
        "This system uses historical loan information and a Logistic "
        "Regression model to estimate the probability that a loan may "
        "not be fully paid."
    )

# =========================================================
# APPLICANT PROFILE
# =========================================================

st.header("👤 Applicant Profile")

p1, p2, p3 = st.columns(3)

with p1:
    applicant_name = st.text_input("Applicant Name", value="Applicant")

with p2:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25,
        step=1
    )

with p3:
    employment = st.selectbox(
        "Employment Type",
        [
            "Salaried",
            "Self-Employed",
            "Business Owner",
            "Student",
            "Other"
        ]
    )

# =========================================================
# LOAN DETAILS
# =========================================================

st.header("🏦 Loan Details")

l1, l2 = st.columns(2)

with l1:
    annual_income = st.number_input(
        "Annual Income",
        min_value=1000.0,
        max_value=10000000.0,
        value=50000.0,
        step=1000.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=500.0,
        max_value=1000000.0,
        value=10000.0,
        step=500.0
    )

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "credit_card",
            "debt_consolidation",
            "educational",
            "home_improvement",
            "major_purchase",
            "small_business",
            "all_other"
        ]
    )

with l2:
    interest_rate = st.number_input(
        "Interest Rate",
        min_value=0.01,
        max_value=0.50,
        value=0.12,
        step=0.01,
        format="%.2f"
    )

    installment = st.number_input(
        "Monthly Installment",
        min_value=10.0,
        max_value=10000.0,
        value=300.0,
        step=10.0
    )

    dti = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.5
    )

# =========================================================
# CREDIT PROFILE
# =========================================================

st.header("💳 Credit Profile")

c1, c2 = st.columns(2)

with c1:
    fico = st.number_input(
        "FICO Credit Score",
        min_value=300,
        max_value=850,
        value=700,
        step=1
    )

    days_with_cr_line = st.number_input(
        "Days with Credit Line",
        min_value=0.0,
        max_value=50000.0,
        value=5000.0,
        step=100.0
    )

    revol_bal = st.number_input(
        "Revolving Balance",
        min_value=0.0,
        max_value=1000000.0,
        value=5000.0,
        step=500.0
    )

with c2:
    revol_util = st.number_input(
        "Credit Utilization (%)",
        min_value=0.0,
        max_value=200.0,
        value=30.0,
        step=1.0
    )

    inquiries = st.number_input(
        "Credit Inquiries in Last 6 Months",
        min_value=0,
        max_value=50,
        value=1,
        step=1
    )

    delinq_2yrs = st.number_input(
        "Delinquencies in Last 2 Years",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    pub_rec = st.number_input(
        "Public Records",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

credit_policy = st.selectbox(
    "Meets Credit Policy?",
    ["Yes", "No"]
)

# =========================================================
# IMPORTANT MODEL NOTE
# =========================================================

st.caption(
    "Note: Loan Amount, Age and Employment Type are collected for the "
    "applicant summary. The saved Logistic Regression model does not use "
    "these fields because they were not part of its original 13 training features."
)

# =========================================================
# PREDICT
# =========================================================

st.write("")

if st.button("🔮 Predict Loan Risk", type="primary"):

    try:

        # Same transformation used by the original model
        log_annual_income = np.log(annual_income)

        credit_policy_value = 1 if credit_policy == "Yes" else 0

        # EXACTLY the 13 features used by the saved model
        input_data = pd.DataFrame({
            "credit.policy": [credit_policy_value],
            "purpose": [purpose],
            "int.rate": [interest_rate],
            "installment": [installment],
            "log.annual.inc": [log_annual_income],
            "dti": [dti],
            "fico": [fico],
            "days.with.cr.line": [days_with_cr_line],
            "revol.bal": [revol_bal],
            "revol.util": [revol_util],
            "inq.last.6mths": [inquiries],
            "delinq.2yrs": [delinq_2yrs],
            "pub.rec": [pub_rec]
        })

        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        probability_percentage = probability * 100

        # Risk level
        if probability < 0.30:
            risk_level = "LOW RISK"
            risk_icon = "🟢"
            risk_message = (
                "The applicant has a relatively low estimated probability "
                "of loan default based on the information provided."
            )
        elif probability < 0.60:
            risk_level = "MEDIUM RISK"
            risk_icon = "🟡"
            risk_message = (
                "The applicant has a moderate estimated probability of "
                "default. Additional financial assessment is recommended."
            )
        else:
            risk_level = "HIGH RISK"
            risk_icon = "🔴"
            risk_message = (
                "The applicant has a relatively high estimated probability "
                "of loan default. Careful assessment is recommended."
            )

        # =================================================
        # RESULT
        # =================================================

        st.write("---")
        st.header("📊 Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Default Probability",
                f"{probability_percentage:.2f}%"
            )

        with r2:
            st.metric(
                "Risk Level",
                f"{risk_icon} {risk_level}"
            )

        with r3:
            if prediction == 1:
                prediction_text = "Possible Default"
            else:
                prediction_text = "Likely Fully Paid"

            st.metric(
                "Model Prediction",
                prediction_text
            )

        # Probability bar
        st.subheader("📈 Default Probability")
        st.progress(float(probability))
        st.caption(
            f"Estimated chance of the loan not being fully paid: "
            f"{probability_percentage:.2f}%"
        )

        # Result explanation
        st.subheader("💡 Result Explanation")
        st.info(risk_message)

        if probability < 0.30:
            st.write(
                """
**What this means:**
- The estimated default probability is relatively low.
- The provided financial and credit information appears comparatively favourable.
- Normal verification procedures should still be followed.
"""
            )
        elif probability < 0.60:
            st.write(
                """
**What this means:**
- The estimated default probability is moderate.
- Additional review of income, debt level and credit history is advisable.
- The result should be considered together with other financial information.
"""
            )
        else:
            st.write(
                """
**What this means:**
- The estimated default probability is relatively high.
- Additional financial evaluation is recommended.
- Credit history, debt obligations and repayment capacity should be reviewed carefully.
"""
            )

        # =================================================
        # APPLICANT SUMMARY
        # =================================================

        st.subheader("📋 Applicant Summary")

        summary = pd.DataFrame({
            "Information": [
                "Applicant Name",
                "Age",
                "Employment Type",
                "Annual Income",
                "Loan Amount",
                "Loan Purpose",
                "Interest Rate",
                "Monthly Installment",
                "Debt-to-Income Ratio",
                "FICO Score",
                "Days with Credit Line",
                "Revolving Balance",
                "Credit Utilization",
                "Credit Inquiries",
                "Delinquencies",
                "Public Records",
                "Credit Policy"
            ],
            "Value": [
                applicant_name,
                age,
                employment,
                f"{annual_income:,.2f}",
                f"{loan_amount:,.2f}",
                purpose.replace("_", " ").title(),
                f"{interest_rate:.2%}",
                f"{installment:,.2f}",
                f"{dti:.2f}",
                fico,
                f"{days_with_cr_line:,.0f}",
                f"{revol_bal:,.2f}",
                f"{revol_util:.1f}%",
                inquiries,
                delinq_2yrs,
                pub_rec,
                credit_policy
            ]
        })

        st.dataframe(summary)

        # =================================================
        # QUICK ADVICE
        # =================================================

        st.subheader("📝 Quick Assessment")

        q1, q2, q3 = st.columns(3)

        with q1:
            if fico >= 700:
                st.success("✅ Strong FICO Score")
            elif fico >= 600:
                st.warning("⚠️ Moderate FICO Score")
            else:
                st.error("❗ Low FICO Score")

        with q2:
            if dti <= 20:
                st.success("✅ Relatively Low DTI")
            elif dti <= 40:
                st.warning("⚠️ Moderate DTI")
            else:
                st.error("❗ High DTI")

        with q3:
            if revol_util <= 30:
                st.success("✅ Low Credit Utilization")
            elif revol_util <= 60:
                st.warning("⚠️ Moderate Utilization")
            else:
                st.error("❗ High Utilization")

        # =================================================
        # HOW IT WORKS
        # =================================================

        with st.expander("🔎 How does the system work?"):

            st.write(
                """
**1. Applicant Input**

Financial and credit information is entered into the application.

**2. Data Transformation**

Annual income is converted to the logarithmic form used by the
original training dataset.

**3. Machine Learning**

The 13 model features are passed to the trained Logistic Regression model.

**4. Probability Estimation**

The model calculates the estimated probability of loan default.

**5. Risk Classification**

The probability is interpreted as Low, Medium or High Risk.

**6. Result Explanation**

The application presents the prediction, probability and applicant
summary in an easy-to-understand format.
"""
            )

    except Exception as e:

        st.error("❌ Unable to generate prediction.")
        st.exception(e)

# =========================================================
# FOOTER
# =========================================================

st.write("---")

st.caption(
    "🎓 AI & Data Science Internship Project | Loan Default Prediction"
)

st.caption(
    "⚠️ This application is for educational/project demonstration purposes only."
)
