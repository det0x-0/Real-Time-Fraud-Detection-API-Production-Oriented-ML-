import joblib
from flask import current_app

def load_model():
    model_path = current_app.config["MODEL_PATH"]
    return joblib.load(model_path)
