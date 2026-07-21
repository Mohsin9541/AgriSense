from flask import Flask, render_template, request
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --------------------------------
# LOAD TRAINED MODEL AND ENCODERS
# --------------------------------

model = joblib.load("model/model.pkl")
soil_encoder = joblib.load("model/soil_encoder.pkl")
crop_encoder = joblib.load("model/crop_encoder.pkl")
fert_encoder = joblib.load("model/fertilizer_encoder.pkl")


# --------------------------------
# NUTRIENT STATUS FUNCTION
# --------------------------------

def nutrient_status(value, low_limit, high_limit):

    if value < low_limit:
        return "LOW"

    elif value > high_limit:
        return "HIGH"

    else:
        return "NORMAL"


# --------------------------------
# HOME PAGE
# --------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    alert = None
    score = None
    alert_message = None

    nitrogen_status = None
    phosphorus_status = None
    potassium_status = None

    if request.method == "POST":

        # -------------------------
        # GET SENSOR INPUT
        # -------------------------

        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        moisture = float(request.form["moisture"])

        soil_type = request.form["soil_type"]
        crop_type = request.form["crop_type"]

        nitrogen = float(request.form["nitrogen"])
        potassium = float(request.form["potassium"])
        phosphorous = float(request.form["phosphorous"])

        # -------------------------
        # ENCODE CATEGORICAL DATA
        # -------------------------

        soil_encoded = soil_encoder.transform(
            [soil_type]
        )[0]

        crop_encoded = crop_encoder.transform(
            [crop_type]
        )[0]

        # -------------------------
        # CREATE MODEL INPUT
        # -------------------------

        input_data = pd.DataFrame(
            [[
                temperature,
                humidity,
                moisture,
                soil_encoded,
                crop_encoded,
                nitrogen,
                potassium,
                phosphorous
            ]],
            columns=[
                "Temparature",
                "Humidity",
                "Moisture",
                "Soil Type",
                "Crop Type",
                "Nitrogen",
                "Potassium",
                "Phosphorous"
            ]
        )

        # -------------------------
        # ML PREDICTION
        # -------------------------

        predicted_number = model.predict(
            input_data
        )

        prediction = fert_encoder.inverse_transform(
            predicted_number
        )[0]

        # -------------------------
        # PREDICTION CONFIDENCE
        # -------------------------

        probabilities = model.predict_proba(
            input_data
        )

        confidence = round(
            np.max(probabilities) * 100,
            2
        )
         # -------------------------
        # SOIL HEALTH SCORE
        # -------------------------

        score = 100

        if nitrogen < 30:
            score -= 20

        if phosphorous < 20:
            score -= 20

        if potassium < 20:
            score -= 20

        if moisture < 30:
            score -= 20

        if humidity < 40:
            score -= 10

        if temperature > 38:
            score -= 10

        score = max(0, score)

        # -------------------------
        # NPK ANALYSIS
        # -------------------------
        # General demonstration thresholds.
        # These can later be made crop-specific.

        nitrogen_status = nutrient_status(
            nitrogen,
            30,
            70
        )

        phosphorus_status = nutrient_status(
            phosphorous,
            20,
            60
        )

        potassium_status = nutrient_status(
            potassium,
            20,
            60
        )

        # -------------------------
        # FERTILIZER ALERT
        # -------------------------

        deficient = []

        if nitrogen_status == "LOW":
            deficient.append("Nitrogen")

        if phosphorus_status == "LOW":
            deficient.append("Phosphorous")

        if potassium_status == "LOW":
            deficient.append("Potassium")

        if deficient:

            alert = "FERTILIZER REQUIRED"

            alert_message = (
                "Low nutrient levels detected: "
                + ", ".join(deficient)
                + ". Fertilizer application may be required."
            )

        else:

            alert = "NO IMMEDIATE DEFICIENCY"

            alert_message = (
                "No low NPK nutrient levels were detected "
                "using the current alert thresholds."
            )

    # --------------------------------
    # SEND DATA TO WEBSITE
    # --------------------------------

    return render_template(

        "index.html",

        prediction=prediction,

        confidence=confidence,

        alert=alert,
        
        soil_score=score,

        alert_message=alert_message,

        nitrogen_status=nitrogen_status,

        phosphorus_status=phosphorus_status,

        potassium_status=potassium_status,

        soils=list(
            soil_encoder.classes_
        ),

        crops=list(
            crop_encoder.classes_
        ),
        date=datetime.now().strftime("%d-%m-%Y")
    )


# --------------------------------
# RUN FLASK
# --------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )