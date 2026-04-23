from pathlib import Path
import warnings

import joblib
import streamlit as st


MODEL_PATH = Path(__file__).with_name("accident_model.pkl")
WARNING_TEXT = "X does not have valid feature names"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)


def format_label(feature_name: str) -> str:
    return feature_name.replace(" - ", "\n")


st.set_page_config(
    page_title="Accident Prediction App",
    layout="centered",
)

st.title("Accident Prediction App")
st.caption("Lightweight Streamlit interface powered by your trained ML model.")

try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()

feature_names = list(getattr(model, "feature_names_in_", []))

if not feature_names:
    st.error("The model does not expose input feature names, so the UI cannot be generated safely.")
    st.stop()

st.write("Enter values for the six features below and click predict.")

with st.form("prediction_form"):
    values = []
    columns = st.columns(2)

    for index, feature_name in enumerate(feature_names):
        with columns[index % 2]:
            value = st.number_input(
                label=format_label(feature_name),
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.2f",
            )
            values.append(value)

    submitted = st.form_submit_button("Predict")

if submitted:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=WARNING_TEXT)
        prediction = float(model.predict([values])[0])

    st.success("Prediction generated successfully.")
    st.metric("Predicted accident estimate", f"{prediction:,.2f}")

with st.expander("Model details"):
    st.write(f"Model type: `{type(model).__name__}`")
    st.write(f"Expected features: `{len(feature_names)}`")
    st.write("Feature order used for prediction:")
    for feature_name in feature_names:
        st.write(f"- {feature_name}")
