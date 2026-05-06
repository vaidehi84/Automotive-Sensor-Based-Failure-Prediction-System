import pickle
from pathlib import Path
import pandas as pd
import streamlit as st

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "model.pkl"
SCALER_PATH = ROOT_DIR / "scaler.pkl"

@st.cache_data
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_model_and_scaler():
    model = load_pickle(MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    return model, scaler

@st.cache_data
def create_input_dataframe(temperature: float, rpm: float, load: float, vibration: float):
    return pd.DataFrame(
        [
            {
                "temperature": temperature,
                "rpm": rpm,
                "load": load,
                "vibration": vibration,
            }
        ]
    )

def main():
    st.set_page_config(
        page_title="Automotive Sensor-Based Failure Predictor",
        page_icon="🚗",
        layout="centered",
    )

    st.title("Automotive Sensor-Based Failure Prediction System")
    st.markdown(
        "Use sensor readings from temperature, RPM, load, and vibration to predict whether a vehicle is likely to fail. "
        "This model is trained with synthetic data and delivers a realistic failure probability score for predictive maintenance scenarios."
    )

    st.divider()

    st.subheader("Sensor input controls")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature (°C)", min_value=20.0, max_value=250.0, value=100.0, step=1.0)
        rpm = st.slider("RPM", min_value=500.0, max_value=8000.0, value=3200.0, step=100.0)
    with col2:
        load = st.slider("Load (%)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
        vibration = st.slider("Vibration (g)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

    input_df = create_input_dataframe(temperature, rpm, load, vibration)

    try:
        model, scaler = load_model_and_scaler()
        scaled_input = scaler.transform(input_df)
        probability = model.predict_proba(scaled_input)[0][1]
        prediction = model.predict(scaled_input)[0]
    except FileNotFoundError:
        st.error("Model files are missing. Run the training script first to generate model.pkl and scaler.pkl.")
        return

    result_text = "FAIL" if prediction == 1 else "HEALTHY"
    result_color = "red" if prediction == 1 else "green"

    st.markdown("### Prediction result")
    st.metric(label="Vehicle Condition", value=result_text, delta=f"{probability * 100:.1f}% failure confidence")

    st.markdown("### Failure probability")
    st.progress(min(max(probability, 0.0), 1.0))
    st.write(
        "This score is the model's estimated probability that the vehicle will experience a failure state given the current sensor readings."
    )

    st.divider()
    st.subheader("Current sensor profile")
    st.bar_chart(input_df.T, height=280)

    st.divider()
    st.subheader("How to use this app")
    st.write(
        "Adjust the sliders to reflect the current sensor readings. The prediction updates automatically, showing whether the system considers the vehicle HEALTHY or at risk of FAILURE. "
        "Use this for prototype predictive maintenance dashboards and to understand how sensor signals influence failure risk." 
    )

    st.markdown("---")
    st.caption("Model: RandomForestClassifier trained on synthetic automotive sensor failure data. Scaler: StandardScaler.")

if __name__ == "__main__":
    main()
