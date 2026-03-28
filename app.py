import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Liver Predictor", page_icon="🧬", layout="wide")

model = joblib.load("liver_model.pkl")

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>🧠 Liver Disease Prediction System</h1>
    <p style='text-align: center;'>AI-powered health risk assessment tool</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧍 Patient Info")
    age = st.number_input("Age", value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    st.markdown("### 🧪 Blood Parameters")
    tb = st.number_input("Total Bilirubin", value=0.8)
    db = st.number_input("Direct Bilirubin", value=0.2)
    alkphos = st.number_input("Alkaline Phosphotase", value=180)

with col2:
    st.markdown("### 🧬 Enzyme Levels")
    alt = st.number_input("ALT", value=25)
    ast = st.number_input("AST", value=30)

    st.markdown("### 🧾 Protein Levels")
    tp = st.number_input("Total Proteins", value=7.0)
    alb = st.number_input("Albumin", value=4.0)
    ag_ratio = st.number_input("A/G Ratio", value=1.2)

st.markdown("<br>", unsafe_allow_html=True)

# Center button
center_col = st.columns([1,2,1])
with center_col[1]:
    predict = st.button("🚀 Predict")

if predict:
    bilirubin_ratio = db / tb if tb != 0 else 0
    enzyme_ratio = ast / alt if alt != 0 else 0

    input_data = np.array([[
        age, gender, tb, db, alkphos,
        alt, ast, tp, alb, ag_ratio,
        bilirubin_ratio, enzyme_ratio
    ]])

    prob = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.progress(int(prob * 100))

    if prob < 0.5:
        st.success(f"🟢 Low Risk\n\nProbability: {prob:.2f}")
    elif prob < 0.75:
        st.warning(f"🟡 Medium Risk\n\nProbability: {prob:.2f}")
    else:
        st.error(f"🔴 High Risk\n\nProbability: {prob:.2f}")

    st.metric("Risk Probability", f"{prob*100:.1f}%")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Developed by Milan Choudhary | AI Project</p>",
    unsafe_allow_html=True
)

st.warning("⚠️ This is an AI-based prediction tool and not a medical diagnosis. Please consult a doctor.")
