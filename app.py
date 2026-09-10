import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 25px;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    margin-bottom: 15px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #dddddd;
    margin-top: 20px;
    margin-bottom: 20px;
}

.result-title {
    font-size: 28px;
    font-weight: 700;
}

.result-score {
    font-size: 42px;
    font-weight: 700;
}

.section-title {
    font-size: 23px;
    font-weight: 650;
    margin-top: 15px;
    margin-bottom: 10px;
}

.small-text {
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

@st.cache_resource
def load_and_train():

    df = pd.read_csv("loan_data.csv")

    # Find target column
    target_col = None

    for c in df.columns:
        if (
            'default' in c.lower()
            or 'status' in c.lower()
            or 'target' in c.lower()
            or 'not.fully.paid' in c.lower()
        ):
            target_col = c
            break

    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    cat_cols = X.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pre = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    model = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(
            random_state=42,
            n_estimators=100
        ))
    ])

    model.fit(X, y)

    return model, X, target_col


model, X_cols, target = load_and_train()


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<div class="main-title">💰 Loan Default Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An AI-powered machine learning application for assessing loan default risk'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------
# PROJECT INFORMATION
# ---------------------------------------------------

with st.expander("ℹ️ About this Project", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🤖 Algorithm", "Random Forest")

    with col2:
        st.metric("📊 Training Records", X_cols.shape[0])

    with col3:
        st.metric("🎯 Target", str(target))

    st.write(
        """
        This application uses a **Random Forest Machine Learning model**
        to analyse applicant information and estimate the likelihood of
        loan default.
        """
    )

    st.info(
        "The prediction is generated from the information entered below. "
        "It should be used as a supporting analysis tool and not as the "
        "only basis for financial decisions."
    )


# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">📝 Applicant Information</div>',
    unsafe_allow_html=True
)

st.write("Enter the applicant's financial and credit information below.")

user_input = {}

cols = st.columns(2)

for i, col_name in enumerate(X_cols.columns):

    c = cols[i % 2]

    if X_cols[col_name].dtype in [np.float64, np.int64]:

        median_value = float(X_cols[col_name].median())

        user_input[col_name] = c.number_input(
            col_name,
            value=median_value
        )

    else:

        default_value = str(
            X_cols[col_name].dropna().iloc[0]
        )

        user_input[col_name] = c.text_input(
            col_name,
            value=default_value
        )


st.write("")


# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

predict_clicked = st.button(
    "🔍 Predict Loan Risk",
    type="primary"
)


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict_clicked:

    df = pd.DataFrame([user_input])

    try:

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0]

        # Probability of predicted class
        confidence = max(prob)

        # Try to identify probability of default
        classes = model.classes_

        if 1 in classes:

            default_index = list(classes).index(1)
            default_probability = prob[default_index]

        else:

            default_probability = (
                confidence
                if str(pred).lower()
                in ['1', 'yes', 'true', 'default']
                else 1 - confidence
            )

        default_percentage = default_probability * 100


        # ---------------------------------------------------
        # RISK LEVEL
        # ---------------------------------------------------

        if default_percentage < 30:

            risk_level = "LOW RISK"
            risk_message = (
                "The applicant shows a relatively low probability "
                "of loan default based on the information provided."
            )

            st.success("✅ LOW RISK")

        elif default_percentage < 60:

            risk_level = "MEDIUM RISK"
            risk_message = (
                "The applicant has a moderate estimated probability "
                "of default. Additional financial assessment is recommended."
            )

            st.warning("⚠️ MEDIUM RISK")

        else:

            risk_level = "HIGH RISK"
            risk_message = (
                "The applicant shows a relatively high estimated "
                "probability of loan default. Careful assessment is recommended."
            )

            st.error("🚨 HIGH RISK")


        # ---------------------------------------------------
        # RESULT
        # ---------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Risk Level",
                risk_level
            )

        with r2:
            st.metric(
                "Default Probability",
                f"{default_percentage:.1f}%"
            )

        with r3:
            st.metric(
                "Model Confidence",
                f"{confidence * 100:.1f}%"
            )


        # ---------------------------------------------------
        # PROBABILITY BAR
        # ---------------------------------------------------

        st.write("### 📈 Default Probability")

        st.progress(
            min(int(default_percentage), 100)
        )

        st.caption(
            f"Estimated probability of loan default: "
            f"{default_percentage:.1f}%"
        )


        # ---------------------------------------------------
        # RESULT EXPLANATION
        # ---------------------------------------------------

        st.write("### 💡 What does this result mean?")

        st.info(risk_message)

        if default_percentage < 30:

            st.write(
                """
                **Interpretation:**
                - The predicted default probability is relatively low.
                - The applicant's provided financial information appears
                  comparatively favourable.
                - Normal verification procedures should still be followed.
                """
            )

        elif default_percentage < 60:

            st.write(
                """
                **Interpretation:**
                - The predicted default probability is moderate.
                - Some financial factors may require additional review.
                - Income, debt level and credit history should be examined carefully.
                """
            )

        else:

            st.write(
                """
                **Interpretation:**
                - The predicted default probability is relatively high.
                - The applicant may require additional financial evaluation.
                - Credit history, debt obligations and repayment capacity
                  should be reviewed carefully.
                """
            )


        # ---------------------------------------------------
        # INPUT SUMMARY
        # ---------------------------------------------------

        st.write("### 📋 Applicant Summary")

        summary = pd.DataFrame({
            "Information": list(user_input.keys()),
            "Value": list(user_input.values())
        })

        st.dataframe(summary)


        # ---------------------------------------------------
        # HOW MODEL WORKS
        # ---------------------------------------------------

        with st.expander("🔎 How does the prediction work?"):

            st.write(
                """
                **Step 1 — Input**
                
                Applicant financial and credit information is entered
                into the application.

                **Step 2 — Preprocessing**
                
                Missing numerical values are handled using median
                imputation and categorical values are handled using
                most-frequent imputation and one-hot encoding.

                **Step 3 — Machine Learning**
                
                The processed information is passed to a Random Forest
                classification model.

                **Step 4 — Prediction**
                
                The model estimates the probability of loan default.

                **Step 5 — Risk Classification**
                
                The estimated probability is used to display the applicant's
                risk level.
                """
            )


    except Exception as e:

        st.error("❌ Unable to generate prediction.")

        st.exception(e)


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "🎓 AI & Data Science Internship Project | Loan Default Prediction"
)

st.caption(
    "⚠️ This application is for educational/project demonstration purposes only."
)
