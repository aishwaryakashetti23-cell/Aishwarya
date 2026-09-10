import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Loan Default Prediction", page_icon="💰")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("loan_data.csv")
    # Find target column
    target_col = None
    for c in df.columns:
        if 'default' in c.lower() or 'status' in c.lower() or 'target' in c.lower():
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    pre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    model = Pipeline([('pre', pre), ('clf', RandomForestClassifier(random_state=42))])
    model.fit(X, y)
    return model, X, target_col

model, X_cols, target = load_and_train()

st.title("💰 Loan Default Prediction System")
st.success(f"Model trained on loan_data.csv - {X_cols.shape[0]} rows")
st.write(f"Predicting: {target}")

user_input = {}
cols = st.columns(2)
for i, col_name in enumerate(X_cols.columns):
    c = cols[i%2]
    if X_cols[col_name].dtype in [np.float64, np.int64]:
        user_input[col_name] = c.number_input(col_name, value=float(X_cols[col_name].median()))
    else:
        user_input[col_name] = c.text_input(col_name, value=str(X_cols[col_name].iloc[0]))

if st.button("Predict", type="primary"):
    df = pd.DataFrame([user_input])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    if str(pred).lower() in ['1','yes','true','default']:
        st.error(f"⚠️ High Risk - {max(prob)*100:.1f}%")
    else:
        st.success(f"✅ Low Risk - {max(prob)*100:.1f}%")
