import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
from pathlib import Path

# --- Configuration --- #
BASE_DIR = Path(__file__).resolve().parent


def resolve_artifact_path(*candidates):
    for candidate in candidates:
        exact_path = BASE_DIR / candidate
        if exact_path.exists():
            return str(exact_path)

    for candidate in candidates:
        matches = list(BASE_DIR.glob(candidate))
        if matches:
            return str(matches[0])

    return str(BASE_DIR / candidates[0])


MODEL_PATH = resolve_artifact_path('best_model.pkl', 'best_model*.pkl')
SCALER_PATH = resolve_artifact_path('scaler.pkl', 'scaler*.pkl')
LABEL_ENCODER_PATH = resolve_artifact_path('label_encoder.pkl', 'label_encoder*.pkl')
CONFIG_PATH = resolve_artifact_path('feature_config.json', 'feature_config*.json')

# --- Load Artifacts --- #
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        with open(CONFIG_PATH, 'r') as f:
            feature_config = json.load(f)
        return model, scaler, label_encoder, feature_config
    except FileNotFoundError as e:
        st.error(f"Error: One or more model files not found. Please ensure all files ({MODEL_PATH}, {SCALER_PATH}, {LABEL_ENCODER_PATH}, {CONFIG_PATH}) are in the same directory as this app.py file. Missing: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model artifacts: {e}")
        st.stop()

model, scaler, label_encoder, feature_config = load_artifacts()

RAW_FEATURES = feature_config['raw_features']
ENGINEERED_FEATURES = feature_config['engineered_features']
ALL_FEATURES = feature_config['all_features']
BEST_MODEL_NAME = feature_config['best_model_name']
MODEL_NEEDS_SCALING = feature_config['model_needs_scaling']

# --- Feature Engineering Function --- #
def apply_feature_engineering(df_input):
    df_engineered = df_input.copy()
    if 'drift_24h' in ENGINEERED_FEATURES:
        df_engineered['drift_24h'] = df_engineered['burn_in_24h'] - df_engineered['initial_measurement']
    if 'drift_96h' in ENGINEERED_FEATURES:
        df_engineered['drift_96h'] = df_engineered['burn_in_96h'] - df_engineered['initial_measurement']
    if 'total_drift' in ENGINEERED_FEATURES:
        df_engineered['total_drift'] = df_engineered['final_burn_in_reading'] - df_engineered['initial_measurement']
    if 'drift_rate' in ENGINEERED_FEATURES:
        # Assuming total_drift is already calculated or its components are available
        if 'total_drift' in df_engineered.columns:
            df_engineered['drift_rate'] = df_engineered['total_drift'] / 3
        else:
            df_engineered['drift_rate'] = (df_engineered['final_burn_in_reading'] - df_engineered['initial_measurement']) / 3
    if 'acceleration' in ENGINEERED_FEATURES:
        # Assuming drift_96h and drift_24h are already calculated
        if 'drift_96h' in df_engineered.columns and 'drift_24h' in df_engineered.columns:
            df_engineered['acceleration'] = df_engineered['drift_96h'] - df_engineered['drift_24h']
        else:
            # Recalculate if not present (should ideally be consistent)
            drift_96h_val = df_engineered['burn_in_96h'] - df_engineered['initial_measurement']
            drift_24h_val = df_engineered['burn_in_24h'] - df_engineered['initial_measurement']
            df_engineered['acceleration'] = drift_96h_val - drift_24h_val
    return df_engineered

# --- Streamlit UI --- #
st.set_page_config(page_title="Burn-in Anomaly Detection", layout="centered")

st.title("📈 Burn-in Anomaly Detector")
st.write("Enter the component's measurement data to predict if it's an ANOMALY or NORMAL.")

st.header("Component Measurements")

# Input fields for raw features
input_data = {}
for feature in RAW_FEATURES:
    input_data[feature] = st.number_input(f"Enter {feature.replace('_', ' ').title()}:", value=9.5, format="%.3f", key=feature)

if st.button("Predict Anomaly"):
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])

    # Apply feature engineering
    processed_df = apply_feature_engineering(input_df)

    # Ensure the order of columns matches what the model was trained on
    # This is crucial for models that rely on feature order (like many scikit-learn models)
    final_features_df = processed_df[ALL_FEATURES]

    # Scale features if the model requires it
    if MODEL_NEEDS_SCALING:
        features_scaled = scaler.transform(final_features_df)
        X_final = features_scaled
    else:
        X_final = final_features_df.values # Convert to numpy array for tree models

    # Make prediction
    prediction_encoded = model.predict(X_final)
    prediction_proba = model.predict_proba(X_final)[:, 1]

    # Decode the prediction
    # Assuming label_encoder.classes_ contains ['NORMAL', 'ANOMALY'] or similar
    # The previous notebook set ANOMALY=1, NORMAL=0. The label_encoder.inverse_transform expects 0 or 1.
    # So if 1 was ANOMALY and 0 was NORMAL in y_encoded, the label_encoder should map them back.

    # Check if the label_encoder was used to map (NORMAL -> 0, ANOMALY -> 1)
    # If y_encoded = 1 - y_encoded was applied, then the original LabelEncoder might map ANOMALY to 0 and NORMAL to 1.
    # It's safer to directly map based on what we know: 1 = ANOMALY, 0 = NORMAL
    if prediction_encoded[0] == 1:
        predicted_label = 'ANOMALY'
        st.error(f"\n\n### Predicted Status: {predicted_label} 🔴")
    else:
        predicted_label = 'NORMAL'
        st.success(f"\n\n### Predicted Status: {predicted_label} 🟢")

    st.write(f"Confidence (Probability of Anomaly): {prediction_proba[0]:.4f}")
    st.write(f"_Prediction made using the '{BEST_MODEL_NAME}' model._")

    with st.expander("View Raw Inputs and Processed Features"):
        st.write("**Raw Inputs:**")
        st.dataframe(input_df)
        st.write("**Processed Features (after engineering):**")
        st.dataframe(processed_df)
        if MODEL_NEEDS_SCALING:
            st.write("**Scaled Features (before prediction):**")
            st.dataframe(pd.DataFrame(X_final, columns=ALL_FEATURES))