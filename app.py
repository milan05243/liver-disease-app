import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Liver Predictor", page_icon="🧬", layout="wide")

model = joblib.load("liver_model.pkl")

st.title("🧠 Liver Disease Prediction System")
st.markdown("### AI-powered health risk assessment tool")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", value=35)
    tb = st.number_input("Total Bilirubin", value=0.8)
    alkphos = st.number_input("Alkaline Phosphotase", value=180)
    tp = st.number_input("Total Proteins", value=7.0)
    alb = st.number_input("Albumin", value=4.0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0
    db = st.number_input("Direct Bilirubin", value=0.2)
    alt = st.number_input("ALT", value=25)
    ast = st.number_input("AST", value=30)
    ag_ratio = st.number_input("A/G Ratio", value=1.2)

if st.button("Predict"):
    bilirubin_ratio = db / tb if tb != 0 else 0
    enzyme_ratio = ast / alt if alt != 0 else 0

    input_data = np.array([[
        age,
        gender,
        tb,
        db,
        alkphos,
        alt,
        ast,
        tp,
        alb,
        ag_ratio,
        bilirubin_ratio,
        enzyme_ratio
    ]])

    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.progress(int(prob * 100))

if prob < 0.5:
    st.success("🟢 Low Risk")
elif prob < 0.75:
    st.warning("🟡 Medium Risk")
else:
    st.error("🔴 High Risk")

    st.metric(label="Risk Probability", value=f"{prob*100:.1f}%")


# Warning
st.warning("⚠️ This is an AI-based prediction tool and not a medical diagnosis. Please consult a doctor.")

# Footer
st.markdown("---")
st.markdown("Developed by Milan Choudhary | AI Project")
