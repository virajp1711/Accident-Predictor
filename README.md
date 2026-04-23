# Accident Prediction Streamlit App

This project includes a lightweight Streamlit app that serves predictions from a compact model artifact.

## Files

- `app.py` - Streamlit UI and inference logic
- `accident_model.pkl` - original trained scikit-learn model (source artifact)
- `accident_model_lite.pkl` - lightweight runtime artifact used by the app
- `requirements.txt` - minimal deployment dependency (`streamlit` only)
- `runtime.txt` - Python runtime for deployment platforms
- `.python-version` - local/dev Python version pin

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

## Performance notes

- Deploy is lighter because runtime does not install `scikit-learn`.
- Inference is pure Python using the pre-exported tree structure in `accident_model_lite.pkl`.
