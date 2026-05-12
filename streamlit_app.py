import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="CTG Classifier", page_icon=":heartpulse:", layout="centered")
st.title("CTG Fetal State Classifier")
st.write(
    "This app loads a trained CTG model and predicts the fetal state "
    "(`Normal`, `Suspect`, or `Pathologic`)."
)


# -----------------------------
# File paths
# -----------------------------
MODEL_PATH = Path("model.pkl")
PREPROCESS_PATH = Path("preprocessing_pipeline.pkl")


# -----------------------------
# Load model and preprocessing
# -----------------------------
if not MODEL_PATH.exists():
    st.error("`model.pkl` not found. Please generate it from the notebook first.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

preprocess_pipeline = None
if PREPROCESS_PATH.exists():
    with open(PREPROCESS_PATH, "rb") as f:
        preprocess_pipeline = pickle.load(f)
    st.success("Loaded both model and preprocessing pipeline.")
else:
    st.warning(
        "`preprocessing_pipeline.pkl` not found. "
        "The app will use the model directly."
    )


# -----------------------------
# Expected features
# -----------------------------
# These are the CTG features used in the notebook.
FEATURE_COLUMNS = [
    "LB",
    "AC",
    "FM",
    "UC",
    "DL",
    "DS",
    "DP",
    "ASTV",
    "MSTV",
    "ALTV",
    "MLTV",
    "Width",
    "Min",
    "Max",
    "Nmax",
    "Nzeros",
    "Mode",
    "Mean",
    "Median",
    "Variance",
    "Tendency",
]

# Default values are simple, safe starters.
DEFAULT_VALUES = {
    "LB": 130.0,
    "AC": 0.003,
    "FM": 0.0,
    "UC": 0.004,
    "DL": 0.001,
    "DS": 0.0,
    "DP": 0.0,
    "ASTV": 50.0,
    "MSTV": 1.5,
    "ALTV": 10.0,
    "MLTV": 8.0,
    "Width": 80.0,
    "Min": 60.0,
    "Max": 170.0,
    "Nmax": 4.0,
    "Nzeros": 0.0,
    "Mode": 130.0,
    "Mean": 135.0,
    "Median": 134.0,
    "Variance": 20.0,
    "Tendency": 0.0,
}


# -----------------------------
# User input form
# -----------------------------
st.subheader("Enter CTG feature values")
st.caption("Tip: Keep defaults if you are just testing the app.")

input_values = {}
for feature in FEATURE_COLUMNS:
    input_values[feature] = st.number_input(
        label=feature,
        value=float(DEFAULT_VALUES[feature]),
        step=0.1,
        format="%.4f",
    )


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict", type="primary"):
    # Convert user input to DataFrame with the exact feature order.
    input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)

    # Optional standalone preprocessing pipeline.
    if preprocess_pipeline is not None:
        processed_array = preprocess_pipeline.transform(input_df)
        model_input = pd.DataFrame(processed_array, columns=FEATURE_COLUMNS)
    else:
        model_input = input_df

    # Predict class number.
    prediction = model.predict(model_input)[0]

    # Map class number to class label.
    class_map = {
        1: "Normal",
        2: "Suspect",
        3: "Pathologic",
    }
    predicted_label = class_map.get(int(prediction), f"Unknown ({prediction})")

    st.success(f"Predicted NSP class: {prediction} - {predicted_label}")
    st.write("Input used for prediction:")
    st.dataframe(input_df)
