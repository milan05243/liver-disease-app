import streamlit as st
import numpy as np
import joblib

model = joblib.load("liver_model.pkl")
imputer = joblib.load("imputer.pkl")

st.title("Liver Disease Prediction System")

age = st.number_input("Age")
gender = st.selectbox("Gender (1=Male, 0=Female)", [1, 0])
tb = st.number_input("Total Bilirubin")
db = st.number_input("Direct Bilirubin")
alkphos = st.number_input("Alkaline Phosphotase")
alt = st.number_input("ALT")
ast = st.number_input("AST")
tp = st.number_input("Total Proteins")
alb = st.number_input("Albumin")
ag_ratio = st.number_input("A/G Ratio")

if st.button("Predict"):
    input_data = np.array([[age, gender, tb, db, alkphos, alt, ast, tp, alb, ag_ratio]])
    input_data = imputer.transform(input_data)

    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.3:
        st.success(f"🟢 Healthy (Probability: {prob:.2f})")
    elif prob < 0.7:
        st.warning(f"🟡 Risk Zone (Probability: {prob:.2f})")
    else:
        st.error(f"🔴 High Risk (Probability: {prob:.2f})")