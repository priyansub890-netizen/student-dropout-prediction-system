import streamlit as st
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="DROPOUT AI",
    page_icon="🎓",
    layout="wide"
)

# ================= SIDEBAR =================
st.sidebar.markdown("## 🎓 DROPOUT AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Dashboard", "🔍 Predict Student Risk", "🧠 AI Explanation"]
)

st.sidebar.markdown("---")
st.sidebar.info("AI-powered Dropout Prediction System")

st.sidebar.markdown("### ⚙️ System Status")
st.sidebar.success("Model: Active 🟢")
st.sidebar.success("Explainability: LIME 🟢")

# ================= LOAD MODEL =================
model = joblib.load("dropout_model.pkl")

feature_names = [
    "G1", "G2", "studytime", "failures", "absences",
    "age", "sex", "internet", "schoolsup", "famsup"
]

# ⚠️ Replace with real training data for best accuracy
X_train = np.array([
    [10, 12, 0.5, 0.1, 0.2, 0.6, 1, 1, 0, 1],
    [5, 6, 0.3, 0.4, 0.5, 0.7, 0, 1, 0, 0],
    [18, 17, 0.8, 0.0, 0.1, 0.5, 1, 1, 1, 1],
])

explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=["Safe", "Risk"],
    mode="classification"
)

# ================= DASHBOARD =================
if page == "🏠 Dashboard":

    st.title("📊 EduRisk AI Dashboard")

    st.markdown("### Welcome to your Student Dropout Prediction Platform")

    col1, col2, col3 = st.columns(3)

    col1.metric("🤖 Model", "Random Forest")
    col2.metric("📊 AI Accuracy", "90%")
    col3.metric("⚡ Status", "Live")

    st.divider()

    st.info("Use sidebar to predict student risk and view AI explanations.")

# ================= PREDICTION PAGE =================
elif page == "🔍 Predict Student Risk":

    st.title("🔍 Student Dropout Risk Prediction")

    st.markdown("Enter student details below:")

    col1, col2 = st.columns(2)

    with col1:
        g1 = st.number_input("G1 Marks", 0, 20, 10)
        g2 = st.number_input("G2 Marks", 0, 20, 10)

        studytime = st.number_input("Study Time (0–24 hrs)", 0, 24, 2)
        failures = st.number_input("Failures (0–25)", 0, 25, 0)

        absences = st.number_input("Absences", 0, 100, 5)

    with col2:
        age = st.number_input("Age", 15, 30, 18)
        sex = st.selectbox("Gender", ["Female", "Male"])
        internet = st.selectbox("Internet", ["No", "Yes"])
        schoolsup = st.selectbox("School Support", ["No", "Yes"])
        famsup = st.selectbox("Family Support", ["No", "Yes"])

    # ================= ENCODING =================
    sex = 1 if sex == "Male" else 0
    internet = 1 if internet == "Yes" else 0
    schoolsup = 1 if schoolsup == "Yes" else 0
    famsup = 1 if famsup == "Yes" else 0

    # ================= NORMALIZATION =================
    studytime = studytime / 24
    failures = failures / 25
    absences = absences / 100
    age = age / 30

    if st.button("🚀 Predict Risk"):

        features = np.array([[
            g1, g2, studytime, failures, absences,
            age, sex, internet, schoolsup, famsup
        ]])

        pred = model.predict(features)[0]
        risk = model.predict_proba(features)[0][1] * 100
        risk = max(0, min(risk, 100))

        st.session_state.features = features
        st.session_state.risk = risk

        st.progress(int(risk))

        if pred == 1:
            st.error(f"⚠️ HIGH DROP OUT RISK: {risk:.2f}%")
        else:
            st.success(f"✅ SAFE STUDENT: {risk:.2f}%")

# ================= AI EXPLANATION PAGE =================
elif page == "🧠 AI Explanation":

    st.title("🧠 Explainable AI (LIME Engine)")

    if "features" not in st.session_state:
        st.warning("⚠️ Please run prediction first")
        st.stop()

    features = st.session_state.features
    risk = st.session_state.risk

    exp = explainer.explain_instance(
        features[0],
        model.predict_proba,
        num_features=6
    )

    lime_list = exp.as_list()

    st.subheader("📌 Feature Impact Analysis")

    colA, colB = st.columns(2)

    with colA:
        for feature, impact in lime_list:
            if impact > 0:
                st.error(f"🔴 {feature} → increases risk")
            else:
                st.success(f"🟢 {feature} → reduces risk")

    with colB:
        st.subheader("🤖 AI Summary")

        risk_factors = [f for f, v in lime_list if v > 0]

        if risk > 70:
            st.warning("High risk due to:")
            st.write(", ".join(risk_factors[:3]))
        elif risk > 40:
            st.info("Moderate risk due to:")
            st.write(", ".join(risk_factors[:2]))
        else:
            st.success("Low risk student with stable performance")