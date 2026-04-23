from pathlib import Path
import pickle

import streamlit as st


LITE_MODEL_PATH = Path(__file__).with_name("accident_model_lite.pkl")


@st.cache_resource(show_spinner=False)
def load_lite_model():
    if not LITE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {LITE_MODEL_PATH.name}")
    with LITE_MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def format_label(feature_name: str) -> str:
    return feature_name.replace(" - ", "\n")


def predict_tree(tree: dict, values: list[float]) -> float:
    node = 0
    feature = tree["feature"]
    threshold = tree["threshold"]
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    node_value = tree["value"]

    while feature[node] != -2:
        split_feature = feature[node]
        if values[split_feature] <= threshold[node]:
            node = children_left[node]
        else:
            node = children_right[node]

    return float(node_value[node])


def predict_forest(model_data: dict, values: list[float]) -> float:
    trees = model_data["trees"]
    total = 0.0
    for tree in trees:
        total += predict_tree(tree, values)
    return total / len(trees)


st.set_page_config(page_title="Accident Prediction App", layout="centered")

st.title("Accident Prediction App")
st.caption("Lightweight interface with pure-Python inference for faster deploys.")

try:
    model_data = load_lite_model()
except Exception as exc:
    st.error(f"Unable to load model: {exc}")
    st.stop()

feature_names = model_data.get("feature_names", [])
if not feature_names:
    st.error("No feature names found in model artifact.")
    st.stop()

st.write("Enter values for the six features and click Predict.")

with st.form("prediction_form"):
    values = []
    columns = st.columns(2)
    for index, feature_name in enumerate(feature_names):
        with columns[index % 2]:
            values.append(
                st.number_input(
                    label=format_label(feature_name),
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    format="%.2f",
                )
            )
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        prediction = predict_forest(model_data, values)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        st.success("Prediction generated successfully.")
        st.metric("Predicted accident estimate", f"{prediction:,.2f}")

with st.expander("Model details"):
    st.write("Model file:", LITE_MODEL_PATH.name)
    st.write("Model type:", model_data.get("model_type", "Unknown"))
    st.write("Expected features:", len(feature_names))
