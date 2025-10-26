# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# App settings
# -----------------------------
st.set_page_config(
    page_title="🩺 Stroke Prediction App",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for background & footer
# -----------------------------
st.markdown(
    """
    <style>
    /* Full-page background color */
    .stApp {
        background-color: #FFF8E7;
    }
    /* Footer */
    footer {
        visibility: visible;
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 16px;
        color: #5D6D7E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <div style="background-color:#FFFAF0; padding:10px; border-radius:10px; margin-bottom:20px;">
        <h1 style="color:#2E4053; text-align:center;">🩺 Stroke Prediction App</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load model, scaler, and columns
# -----------------------------
with open("random_forest_stroke_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# -----------------------------
# User input
# -----------------------------
st.sidebar.header("Enter Patient Details")

def user_input():
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    hypertension = st.sidebar.selectbox("Hypertension", [0,1])
    heart_disease = st.sidebar.selectbox("Heart Disease", [0,1])
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.sidebar.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=80.0)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"])

    data = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": 1 if ever_married=="Yes" else 0,
        "gender": 1 if gender=="Male" else 0,
        "Residence_type": 1 if residence_type=="Urban" else 0,
        "avg_glucose_level": np.log1p(avg_glucose_level),
        "bmi": np.log1p(bmi),
        "work_type_Self-employed": 1 if work_type=="Self-employed" else 0,
        "work_type_Private": 1 if work_type=="Private" else 0,
        "work_type_children": 1 if work_type=="children" else 0,
        "smoking_status_never smoked": 1 if smoking_status=="never smoked" else 0,
        "smoking_status_formerly smoked": 1 if smoking_status=="formerly smoked" else 0,
        "smoking_status_smokes": 1 if smoking_status=="smokes" else 0
    }

    df = pd.DataFrame(data, index=[0])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

input_df = user_input()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Stroke Risk"):
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0][1]

    st.markdown(
        f"""
        <div style="background-color:#E8F8F5; padding:15px; border-radius:10px">
        <h2 style="color:#1B4F72;">Prediction Result:</h2>
        <p style="font-size:20px;">🩺 Stroke Risk: <b>{'Yes' if pred==1 else 'No'}</b></p>
        <p style="font-size:16px;">Risk Probability: {pred_proba:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "<footer>Created by Sunmathi</footer>",
    unsafe_allow_html=True
)
