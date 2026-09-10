import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Loan Default Prediction", page_icon="💰")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("loan_default.csv")
    # Auto-detect target column - common names
    target_col = None
    for col in ['default', 'Default', 'loan_default', 'Loan_Default', 'Status', 'loan_status', 'TARGET']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1] # last column as target

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Separate numeric and categorical
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    return model, X.columns.tolist(), numeric_features, categorical_features

model, all_features, num_feats, cat_feats = load_and_train()

st.title("💰 Loan Default Prediction System")
st.write("This application predicts the probability of loan default using machine learning model trained on your dataset.")

st.header("Applicant Information")

# Create dynamic inputs based on your CSV columns
user_input = {}
cols = st.columns(2)

for i, feature in enumerate(all_features):
    col = cols[i % 2]
    if feature in num_feats:
        user_input[feature] = col.number_input(f"{feature}", value=0.0)
    else:
        # For categorical, get unique values from sample
        user_input[feature] = col.text_input(f"{feature}", value="")

if st.button("Predict Loan Default", type="primary"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    if prediction == 1 or prediction == "Yes" or str(prediction).lower() == "default":
        st.error(f"⚠️ High Risk of Default! Probability: {max(prob)*100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Default! Probability: {max(prob)*100:.2f}%")

st.caption("Model trained live from loan_default.csv - no version mismatch")
