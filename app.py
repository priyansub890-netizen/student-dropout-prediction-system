from flask import Flask, render_template, request
import joblib
import numpy as np
import shap

app = Flask(__name__)

model = joblib.load("dropout_model.pkl")

feature_names = [
    "G1", "G2", "studytime", "failures", "absences",
    "age", "sex", "internet", "schoolsup", "famsup"
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predictpage')
def predictpage():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():

    # ================= INPUT =================
    features = np.array([[
        float(request.form['G1']),
        float(request.form['G2']),
        float(request.form['studytime']),
        float(request.form['failures']),
        float(request.form['absences']),
        float(request.form['age']),
        int(request.form['sex']),
        int(request.form['internet']),
        int(request.form['schoolsup']),
        int(request.form['famsup'])
    ]])

    # ================= MODEL =================
    prediction = model.predict(features)[0]

    risk = model.predict_proba(features)[0][1] * 100
    risk = max(0, min(round(float(risk), 2), 100))

    result_text = "⚠️ Student is AT RISK" if prediction == 1 else "✅ Student is SAFE"

    # ================= SHAP (FULL FIX) =================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # SAFE HANDLING FOR ALL SHAP VERSIONS
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    # ================= EXPLANATION FIX =================
    explanations = []

    for name, value in zip(feature_names, shap_vals):

        # 🔥 FIX: force scalar safely
        value = float(np.array(value).flatten()[0])

        if value > 0.05:
            label = "🔴 Strongly increases risk"
            color = "red"
        elif value > 0:
            label = "🟡 Slightly increases risk"
            color = "orange"
        else:
            label = "🟢 Reduces risk"
            color = "green"

        explanations.append({
            "feature": name,
            "value": round(value, 3),
            "label": label,
            "color": color
        })

    explanations = sorted(explanations, key=lambda x: abs(x["value"]), reverse=True)

    # ================= HUMAN SUMMARY =================
    positive = [e["feature"] for e in explanations if e["value"] > 0]

    if risk > 70:
        summary = f"High risk mainly due to {', '.join(positive[:3])}."
    elif risk > 40:
        summary = f"Moderate risk influenced by {', '.join(positive[:2])}."
    else:
        summary = "Low risk. Student shows good academic performance."

    # ================= RETURN =================
    return render_template(
        "result.html",
        prediction_text=result_text,
        risk_score=risk,
        explanations=explanations,
        summary=summary
    )


if __name__ == "__main__":
    app.run(debug=True)