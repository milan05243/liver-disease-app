import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Liver Disease Predictor", layout="wide")

model = joblib.load("liver_model.pkl")

st.title("Liver Disease Prediction System")
st.write("Provide clinical details to estimate liver disease risk.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Details")
    age = st.number_input("Age", min_value=1, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    st.subheader("Bilirubin & Enzymes")
    total_bilirubin = st.number_input("Total Bilirubin", value=0.8)
    direct_bilirubin = st.number_input("Direct Bilirubin", value=0.2)
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase", value=180)

with col2:
    st.subheader("Liver Enzymes")
    alt = st.number_input("ALT", value=25)
    ast = st.number_input("AST", value=30)

    st.subheader("Protein Levels")
    total_proteins = st.number_input("Total Proteins", value=7.0)
    albumin = st.number_input("Albumin", value=4.0)
    ag_ratio = st.number_input("A/G Ratio", value=1.2)

st.markdown("")

center_col = st.columns([2,1,2])

with center_col[1]:
    predict = st.button("Predict")

st.markdown("---")

if predict:
    bilirubin_ratio = direct_bilirubin / total_bilirubin if total_bilirubin != 0 else 0
    enzyme_ratio = ast / alt if alt != 0 else 0

    input_data = np.array([[
        age,
        gender,
        total_bilirubin,
        direct_bilirubin,
        alkaline_phosphotase,
        alt,
        ast,
        total_proteins,
        albumin,
        ag_ratio,
        bilirubin_ratio,
        enzyme_ratio
    ]])

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    st.progress(int(probability * 100))

    if probability < 0.5:
        st.success("Low Risk")
    elif probability < 0.75:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

    st.write(f"Risk Probability: {probability * 100:.2f}%")

st.markdown("---")
st.caption("Note: This tool provides an estimate and should not be used as a substitute for medical advice.")
