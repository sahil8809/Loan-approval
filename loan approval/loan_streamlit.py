import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("loan_model.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
education_encoder = joblib.load("education_encoder.pkl")

# ---------------- TITLE ----------------
st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant details below.")

# ---------------- USER INPUT ----------------
gender = st.selectbox("Gender", ["Male", "Female"])

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=150
)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    # Convert text to numbers using saved encoders
    gender_encoded = gender_encoder.transform([gender])[0]
    education_encoded = education_encoder.transform([education])[0]

    # Create DataFrame
    input_data = pd.DataFrame({
        "Gender": [gender_encoded],
        "Education": [education_encoded],
        "Income": [income],
        "LoanAmount": [loan_amount]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")