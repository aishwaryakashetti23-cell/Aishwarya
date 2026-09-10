import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# 1. LOAD TRAINED MODEL
# =========================================================

model = joblib.load("loan_default_model.pkl")


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)


# =========================================================
# 3. TITLE
# =========================================================

st.title("💰 Loan Default Prediction System")

st.write(
    "This application predicts the probability of a loan default "
    "using a machine learning model."
)

st.info(
    "The model was trained using Logistic Regression on historical "
    "loan data."
)


# =========================================================
# 4. SIDEBAR
# =========================================================

st.sidebar.title("📌 About the Model")

st.sidebar.write(
    """
    **Model:** Logistic Regression
    
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

st.sidebar.write(
    "The model is intended for educational/project purposes "
    "and should not be used as the sole basis for real financial decisions."
)


# =========================================================
# 5. APPLICANT INFORMATION
# =========================================================

st.header("👤 Applicant Information")

col1, col2 = st.columns(2)

with col1:

    annual_income = st.number_input(
        "Annual Income ($)",
        min_value=1000.0,
        max_value=10000000.0,
        value=50000.0,
        step=1000.0
    )

    loan_amount = st.number_input(
        "Loan Amount ($)",
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

    interest_rate = st.number_input(
        "Interest Rate",
        min_value=0.01,
        max_value=0.50,
        value=0.12,
        step=0.01,
        format="%.2f"
    )


with col2:

    installment = st.number_input(
        "Monthly Installment ($)",
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


# =========================================================
# 6. CREDIT INFORMATION
# =========================================================

st.header("💳 Credit Information")

col3, col4 = st.columns(2)

with col3:

    revol_bal = st.number_input(
        "Revolving Balance ($)",
        min_value=0.0,
        max_value=1000000.0,
        value=5000.0,
        step=500.0
    )

    revol_util = st.number_input(
        "Revolving Credit Utilization (%)",
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

with col4:

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
        [
            "Yes",
            "No"
        ]
    )


# =========================================================
# 7. PREDICTION BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "🔮 Predict Loan Risk",
    use_container_width=True
)


# =========================================================
# 8. MAKE PREDICTION
# =========================================================

if predict_button:

    # Convert annual income into the same format
    # used by the original dataset.
    log_annual_income = np.log(annual_income)

    # Convert credit policy from Yes/No to 1/0
    credit_policy_value = 1 if credit_policy == "Yes" else 0


    # IMPORTANT:
    # The model was trained on exactly these 13 features.
    # loan_amount is NOT included because the original
    # dataset does not contain a loan_amount column.

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


    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    probability_percentage = probability * 100


    # =====================================================
    # RISK LEVEL
    # =====================================================

    if probability < 0.30:

        risk_level = "LOW RISK"
        risk_icon = "🟢"

    elif probability < 0.60:

        risk_level = "MEDIUM RISK"
        risk_icon = "🟡"

    else:

        risk_level = "HIGH RISK"
        risk_icon = "🔴"


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.write("---")

    st.header("📊 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Default Probability",
            f"{probability_percentage:.2f}%"
        )

    with result_col2:

        st.metric(
            "Risk Level",
            f"{risk_icon} {risk_level}"
        )

    with result_col3:

        if prediction == 1:

            prediction_text = "Possible Default"

        else:

            prediction_text = "Likely Fully Paid"

        st.metric(
            "Model Prediction",
            prediction_text
        )


    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    st.subheader("Default Probability")

    st.progress(
        float(probability)
    )


    # =====================================================
    # EXPLANATION
    # =====================================================

    if prediction == 1:

        st.error(
            "⚠️ The model predicts that this applicant has a "
            "higher likelihood of not fully paying the loan."
        )

    else:

        st.success(
            "✅ The model predicts that this applicant is "
            "more likely to fully pay the loan."
        )


    # =====================================================
    # APPLICANT SUMMARY
    # =====================================================

    st.subheader("📋 Applicant Summary")

    summary = pd.DataFrame({
        "Information": [
            "Annual Income",
            "Loan Amount",
            "Loan Purpose",
            "Interest Rate",
            "Monthly Installment",
            "Debt-to-Income Ratio",
            "FICO Score",
            "Revolving Balance",
            "Credit Utilization"
        ],

        "Value": [
            f"${annual_income:,.2f}",
            f"${loan_amount:,.2f}",
            purpose.replace("_", " ").title(),
            f"{interest_rate:.2%}",
            f"${installment:,.2f}",
            f"{dti:.2f}",
            f"{fico}",
            f"${revol_bal:,.2f}",
            f"{revol_util:.1f}%"
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 9. FOOTER
# =========================================================

st.write("")
st.write("---")

st.caption(
    "Loan Default Prediction System | Machine Learning Project"
)

st.caption(
    "⚠️ This application is for educational purposes and "
    "should not be used as the sole basis for real lending decisions."
)