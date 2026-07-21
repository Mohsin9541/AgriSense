import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")

df = pd.read_csv("dataset/train.csv")

# Remove ID because it is only an identifier
df = df.drop("id", axis=1)

# Create encoders
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fert_encoder = LabelEncoder()

# Encode text columns
df["Soil Type"] = soil_encoder.fit_transform(df["Soil Type"])
df["Crop Type"] = crop_encoder.fit_transform(df["Crop Type"])
df["Fertilizer Name"] = fert_encoder.fit_transform(
    df["Fertilizer Name"]
)

# Separate features and target
X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# MODEL 1 - RANDOM FOREST
# -----------------------------

print("\nTraining Random Forest...")

start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print(
    "Random Forest Accuracy:",
    round(rf_accuracy * 100, 2),
    "%"
)

print(
    "Time:",
    round(time.time() - start, 2),
    "seconds"
)

# -----------------------------
# MODEL 2 - EXTRA TREES
# -----------------------------

print("\nTraining Extra Trees...")

start = time.time()

et_model = ExtraTreesClassifier(
    n_estimators=150,
    max_depth=25,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

et_model.fit(X_train, y_train)

et_predictions = et_model.predict(X_test)

et_accuracy = accuracy_score(
    y_test,
    et_predictions
)

print(
    "Extra Trees Accuracy:",
    round(et_accuracy * 100, 2),
    "%"
)

print(
    "Time:",
    round(time.time() - start, 2),
    "seconds"
)

# -----------------------------
# SELECT BEST MODEL
# -----------------------------

if et_accuracy > rf_accuracy:

    best_model = et_model
    best_accuracy = et_accuracy
    best_name = "Extra Trees"

else:

    best_model = rf_model
    best_accuracy = rf_accuracy
    best_name = "Random Forest"

print("\n----------------------------")
print("BEST MODEL:", best_name)

print(
    "BEST ACCURACY:",
    round(best_accuracy * 100, 2),
    "%"
)

print("----------------------------")

# Save best model
joblib.dump(
    best_model,
    "model/model.pkl"
)

# Save encoders
joblib.dump(
    soil_encoder,
    "model/soil_encoder.pkl"
)

joblib.dump(
    crop_encoder,
    "model/crop_encoder.pkl"
)

joblib.dump(
    fert_encoder,
    "model/fertilizer_encoder.pkl"
)

print("\nBest model saved successfully!")