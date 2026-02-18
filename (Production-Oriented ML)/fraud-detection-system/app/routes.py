from flask import Blueprint, request, jsonify, current_app
from .model_loader import load_model
from .database import db, Transaction
from .schemas import transaction_schema
import logging
import numpy as np

fraud_bp = Blueprint("fraud", __name__)
model = None

@fraud_bp.before_app_first_request
def load_model_once():
    global model
    model = load_model()

@fraud_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"}), 200

@fraud_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = transaction_schema.load(request.json)
        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)[0]

        # Save to DB
        transaction = Transaction(
            amount=features[0][0],
            prediction=int(prediction)
        )
        db.session.add(transaction)
        db.session.commit()

        logging.info(f"Prediction made: {prediction}")

        return jsonify({
            "fraud": int(prediction)
        }), 200

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 400

