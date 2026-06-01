import streamlit as st
import numpy as np
import joblib
import shap

# ================= CONFIG =================
st.set_page_config(page_title="Dropout AI ", layout="wide")

model = joblib.load("dropout_model.pkl")

feature_names = [
    "G1", "G2", "studytime", "failures", "absences",
    "age", "sex", "internet", "schoolsup", "famsup"
]

# ================= SIDEBAR NAV =================
st.sidebar.title("🎓 Dropout AI ")
page = st.sidebar.radio("Navigate", ["🏠 Dashboard", "🔍 Predict", "📊 Explain AI"])

# ================= DASHBOARD =================
if page == "🏠 Dashboard":

    st.title("📊 Student Dropout Prediction")
    st.markdown("### AI-powered risk detection system (Random Forest Classifier)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Type", "Random Forest")

    with col2:
        st.metric("Explainability", "SHAP AI")

    with col3:
        st.metric("Status", "Active 🟢")

    st.info("Use sidebar → Predict student risk with AI explanation")

# ================= PREDICT PAGE =================
elif page == "🔍 Predict":

    st.title("🔍 Predict Student Dropout Risk")

    col1, col2 = st.columns(2)

    with col1:
        g1 = st.number_input("G1 Marks", 0, 20, 10)
        g2 = st.number_input("G2 Marks", 0, 20, 10)
        studytime = st.number_input("Study Time", 1, 4, 2)
        failures = st.number_input("Failures", 0, 4, 0)
        absences = st.number_input("Absences", 0, 100, 5)

    with col2:
        age = st.number_input("Age", 15, 30, 18)
        sex = st.selectbox("Gender", ["Female", "Male"])
        internet = st.selectbox("Internet", ["No", "Yes"])
        schoolsup = st.selectbox("School Support", ["No", "Yes"])
        famsup = st.selectbox("Family Support", ["No", "Yes"])

    sex = 1 if sex == "Male" else 0
    internet = 1 if internet == "Yes" else 0
    schoolsup = 1 if schoolsup == "Yes" else 0
    famsup = 1 if famsup == "Yes" else 0

    if st.button("🚀 Predict Risk"):

        features = np.array([[g1, g2, studytime, failures, absences,
                              age, sex, internet, schoolsup, famsup]])

        prediction = model.predict(features)[0]
        risk = model.predict_proba(features)[0][1] * 100
        risk = max(0, min(round(float(risk), 2), 100))

        st.session_state.features = features
        st.session_state.risk = risk
        st.session_state.prediction = prediction

        st.success("Prediction completed! Go to Explain AI tab.")

        st.metric("Risk Score", f"{risk}%")

        st.progress(int(risk))

# ================= AI EXPLANATION PAGE =================
elif page == "📊 Explain AI":

    st.title("🧠 AI Explanation Dashboard")

    if "risk" not in st.session_state:
        st.warning("Please run prediction first.")
        st.stop()

    risk = st.session_state.risk
    features = st.session_state.features

    # ================= RISK DISPLAY =================
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Dropout Risk", f"{risk}%")

    with col2:
        st.progress(int(risk))

    # ================= SHAP =================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    st.subheader("📌 Feature Impact Analysis")

    for name, value in zip(feature_names, shap_vals):

        value = float(np.array(value).flatten()[0])

        if value > 0.05:
            st.error(f"🔴 {name} increases risk ({round(value,3)})")
        elif value > 0:
            st.warning(f"🟡 {name} slightly increases risk ({round(value,3)})")
        else:
            st.success(f"🟢 {name} reduces risk ({round(value,3)})")

    # ================= AI REASONING =================
    st.subheader("🤖 AI Reasoning Summary")

    positive = [n for n, v in zip(feature_names, shap_vals) if float(np.array(v).flatten()[0]) > 0]

    if risk > 70:
        st.error(f"High risk due to {', '.join(positive[:3])}")
    elif risk > 40:
        st.warning(f"Moderate risk due to {', '.join(positive[:2])}")
    else:
        st.success("Low risk student with stable academic performance.")