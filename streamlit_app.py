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

# Examples chosen so your saved RandomForest pipeline (`model.pkl`) predicts NSP 1 / 2 / 3 accordingly.
SAMPLE_NORMAL = {
    "LB": 132.0,
    "AC": 0.006379585326953748,
    "FM": 0.0,
    "UC": 0.006379585326953748,
    "DL": 0.003189792663476874,
    "DS": 0.0,
    "DP": 0.0,
    "ASTV": 17.0,
    "MSTV": 2.1,
    "ALTV": 0.0,
    "MLTV": 10.4,
    "Width": 130.0,
    "Min": 68.0,
    "Max": 198.0,
    "Nmax": 6.0,
    "Nzeros": 1.0,
    "Mode": 141.0,
    "Mean": 136.0,
    "Median": 140.0,
    "Variance": 12.0,
    "Tendency": 0.0,
}
SAMPLE_SUSPECT = {
    "LB": 151.0,
    "AC": 0.0,
    "FM": 0.0,
    "UC": 0.0008340283569641367,
    "DL": 0.0008340283569641367,
    "DS": 0.0,
    "DP": 0.0,
    "ASTV": 64.0,
    "MSTV": 1.9,
    "ALTV": 9.0,
    "MLTV": 27.6,
    "Width": 130.0,
    "Min": 56.0,
    "Max": 186.0,
    "Nmax": 2.0,
    "Nzeros": 0.0,
    "Mode": 150.0,
    "Mean": 148.0,
    "Median": 151.0,
    "Variance": 9.0,
    "Tendency": 1.0,
}
SAMPLE_PATHOLOGIC = {
    "LB": 134.0,
    "AC": 0.001049317943336831,
    "FM": 0.0,
    "UC": 0.01049317943336831,
    "DL": 0.00944386149003148,
    "DS": 0.0,
    "DP": 0.002098635886673662,
    "ASTV": 26.0,
    "MSTV": 5.9,
    "ALTV": 0.0,
    "MLTV": 0.0,
    "Width": 150.0,
    "Min": 50.0,
    "Max": 200.0,
    "Nmax": 5.0,
    "Nzeros": 3.0,
    "Mode": 76.0,
    "Mean": 107.0,
    "Median": 107.0,
    "Variance": 170.0,
    "Tendency": 0.0,
}

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
if "inputs" not in st.session_state:
    st.session_state.inputs = DEFAULT_VALUES.copy()
if "widget_version" not in st.session_state:
    st.session_state.widget_version = 0
if "run_predict_next" not in st.session_state:
    st.session_state.run_predict_next = False


def load_sample(sample_dict):
    st.session_state.inputs = {k: float(sample_dict[k]) for k in FEATURE_COLUMNS}
    st.session_state.widget_version += 1
    st.session_state.run_predict_next = True
    st.rerun()

st.subheader("Enter CTG feature values")
st.caption("Use the sample buttons to load labeled examples from the CTG dataset, or edit manually.")

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Sample: Normal (NSP 1)"):
        load_sample(SAMPLE_NORMAL)
with c2:
    if st.button("Sample: Suspect (NSP 2)"):
        load_sample(SAMPLE_SUSPECT)
with c3:
    if st.button("Sample: Pathologic (NSP 3)"):
        load_sample(SAMPLE_PATHOLOGIC)

v = st.session_state.widget_version
input_values = {}
for feature in FEATURE_COLUMNS:
    input_values[feature] = st.number_input(
        label=feature,
        value=float(st.session_state.inputs[feature]),
        step=0.1,
        format="%.4f",
        key=f"{feature}_v{v}",
    )


# -----------------------------
# Prediction
# -----------------------------
want_predict = st.button("Predict", type="primary")
if st.session_state.run_predict_next:
    want_predict = True
    st.session_state.run_predict_next = False

if want_predict:
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
