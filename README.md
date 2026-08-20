# 🌱 AgriSense — AI-Based Fertilizer Alert System

AgriSense is a machine learning–powered web application that recommends the right fertilizer for a crop based on live soil and environmental sensor readings. It combines a trained classification model with rule-based soil health analysis to give farmers a clear, actionable dashboard: a soil health score, NPK nutrient status, a fertilizer deficiency alert, and an AI-predicted fertilizer recommendation with a confidence score.

## ✨ Features

- **AI Fertilizer Prediction** — Predicts the most suitable fertilizer using a trained scikit-learn classifier (Random Forest / Extra Trees, whichever performs best).
- **Prediction Confidence Score** — Shows how confident the model is in its recommendation.
- **Soil Health Score** — A 0–100 rule-based score derived from temperature, humidity, moisture, and NPK levels.
- **NPK Nutrient Status** — Flags Nitrogen, Phosphorous, and Potassium levels as LOW / NORMAL / HIGH.
- **Fertilizer Deficiency Alert** — Warns when any nutrient falls below the healthy threshold.
- **Simple Web Dashboard** — Clean, single-page Flask + HTML interface for entering sensor data and viewing results.

## 🧠 How It Works

1. Sensor readings (temperature, humidity, moisture, soil type, crop type, N/P/K values) are submitted through the web form.
2. Categorical fields (soil type, crop type) are label-encoded using the same encoders used during training.
3. The trained model predicts a fertilizer class and its confidence.
4. In parallel, a rule-based engine calculates a soil health score and NPK status using threshold checks.
5. All results are rendered together on the dashboard.

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn (RandomForestClassifier, ExtraTreesClassifier), pandas, NumPy
- **Model Persistence:** joblib
- **Frontend:** HTML, CSS (inline, no framework)

## 📁 Project Structure

```
AgriSense/
├── app.py                  # Flask application (routes, prediction logic, rule-based scoring)
├── train_model.py          # Trains and saves the fertilizer prediction model
├── requirements.txt        # Python dependencies
├── dataset/
│   ├── train.csv            # Training data
│   └── test.csv             # Test data
├── model/
│   ├── model.pkl             # Trained classifier
│   ├── soil_encoder.pkl      # Label encoder for soil type
│   ├── crop_encoder.pkl      # Label encoder for crop type
│   └── fertilizer_encoder.pkl# Label encoder for fertilizer name
├── static/                 # Static assets (CSS/JS/images)
└── templates/
    └── index.html            # Web dashboard UI
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/AgriSense.git
cd AgriSense

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

### (Optional) Retraining the Model

If you want to retrain the model on the provided dataset:

```bash
python train_model.py
```

This trains both a Random Forest and an Extra Trees classifier, picks the better-performing one, and saves the updated model and encoders to the `model/` directory.

## 📊 Dataset

The model is trained on soil and crop sensor data with the following fields: `Temparature`, `Humidity`, `Moisture`, `Soil Type`, `Crop Type`, `Nitrogen`, `Potassium`, `Phosphorous`, and the target label `Fertilizer Name`.

## ⚠️ Notes

- NPK and soil health thresholds are currently general-purpose demonstration values and can be made crop-specific for improved accuracy.
- Run with `debug=True` for local development only — disable debug mode before deploying to production.

## 👥 Author

- **Mohsin Nawaz**

