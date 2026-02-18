import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_key")
    DATABASE_URI = os.getenv(
        "DATABASE_URI",
        "postgresql://user:password@localhost:5432/fraud_db"
    )
    MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
