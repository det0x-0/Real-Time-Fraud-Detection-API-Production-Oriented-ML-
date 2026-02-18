import os
import joblib
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "data/transactions.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")


# -------------------------------
# Load Dataset
# -------------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    logging.info("Loading dataset...")
    df = pd.read_csv(path)
    logging.info(f"Dataset shape: {df.shape}")
    return df


# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_data(df: pd.DataFrame):
    logging.info("Preprocessing dataset...")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    numerical_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features)
        ]
    )

    return X, y, preprocessor


# -------------------------------
# Handle Class Imbalance
# -------------------------------
def compute_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight[]
        class_weight="balanced",
        classes=classes,
        y=y
    

    weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
    logging.info(f"Computed class weights: {weight_dict}")
    return weight_dict


# -------------------------------
# Model Training
# -------------------------------
def train_model(X, y, preprocessor):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    class_weights = compute_weights(y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_weights[1],
        random_state=42,
        eval_metric="logloss"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    logging.info("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    logging.info("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC Score: {auc:.4f}")

    return pipeline


# -------------------------------
# Save Model
# -------------------------------
def save_model(model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(model, MODEL_PATH)
    logging.info(f"Model saved at {MODEL_PATH}")


# -------------------------------
# Main Execution
# -------------------------------
def main():
    df = load_data(DATA_PATH)
    X, y, preprocessor = preprocess_data(df)
    model = train_model(X, y, preprocessor)
    save_model(model)


if __name__ == "__main__":
    main()
