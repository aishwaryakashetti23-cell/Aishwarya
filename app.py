import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# PAGE CONFIG
st.set_page_config(page_title="Loan Default Prediction", page_icon="💰", layout="wide")

# LOGIN
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>💰 Loan Default Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Secure access to the Loan Risk Assessment System</p>", unsafe_allow_html=True)
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        st.info("Demo login: username `admin` and password `loan123`")
        if st.button("Login", type="primary"):
            if username == "admin" and password == "loan123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
    st.stop()

# TRAIN MODEL FROM CSV (No pkl needed!)
@st.cache_resource
def load_model():
    df = pd.read_csv("loan_data.csv")
    # Target column detection
    target_col = None
    for c in ["not.fully.paid", "not_fully_paid", "default", "target", "loan_status"]:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[-1]
    
    # Features used in UI
    feature_cols = ["credit.policy", "purpose", "int.rate", "installment", "log.annual.inc", "dti", "fico", "days.with.cr.line", "revol.bal", "revol.util", "inq.last.6mths", "delinq.2yrs", "pub.rec"]
    # If log.annual.inc not in csv, create it
    if "log.annual.inc" not in df.columns and "annual.inc" in df.columns:
        df["log.annual.inc"] = np.log(df["annual.inc"].replace(0, 1))
    if "annual.inc" in df.columns and "log.annual.inc" not in df.columns:
        df["log.annual.inc"] = np.log(df["annual.inc"] + 1)

    X = df[feature_cols]
    y = df[target_col]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["purpose"])
    ], remainder="passthrough")

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model

model = load_model()

