from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

model = tf.keras.models.load_model("stroke_lstm_model.keras")
scaler = joblib.load("scaler.pkl")

gender_map = {"Female": 0, "Male": 1, "Other": 2}
married_map = {"No": 0, "Yes": 1}
work_map = {
    "Govt_job": 0,
    "Never_worked": 1,
    "Private": 2,
    "Self-employed": 3,
    "children": 4
}
residence_map = {"Rural": 0, "Urban": 1}
smoking_map = {
    "Unknown": 0,
    "formerly smoked": 1,
    "never smoked": 2,
    "smokes": 3
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    gender = gender_map[request.form["gender"]]
    age = float(request.form["age"])
    hypertension = int(request.form["hypertension"])
    heart_disease = int(request.form["heart_disease"])
    ever_married = married_map[request.form["ever_married"]]
    work_type = work_map[request.form["work_type"]]
    residence_type = residence_map[request.form["Residence_type"]]
    glucose = float(request.form["avg_glucose_level"])
    bmi = float(request.form["bmi"])
    smoking = smoking_map[request.form["smoking_status"]]

    data = np.array([[
        gender,
        age,
        hypertension,
        heart_disease,
        ever_married,
        work_type,
        residence_type,
        glucose,
        bmi,
        smoking
    ]])

    data = scaler.transform(data)
    data = data.reshape(1, 10, 1)

    probability = model.predict(data, verbose=0)[0][0]

    if probability >= 0.5:
        prediction = "⚠️ Stroke Risk Detected"
    else:
        prediction = "✅ Low Stroke Risk"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)