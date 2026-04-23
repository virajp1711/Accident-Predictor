# Accident Prediction Streamlit App

This project includes a lightweight Streamlit app that serves predictions from `accident_model.pkl`.

## Files

- `app.py` - Streamlit UI and inference logic
- `accident_model.pkl` - trained scikit-learn model
- `requirements.txt` - minimal deployment dependencies

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Deploy the folder as-is on Streamlit Community Cloud or any platform that can run:

```powershell
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

The app loads the model directly from the project directory and generates the input form from the model's saved feature names.
