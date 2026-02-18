from flask import Flask
from .routes import fraud_bp
from .config import Config
from .database import init_db
import logging
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Logging setup
    if not os.path.exists("logs"):
        os.mkdir("logs")

    logging.basicConfig(
        filename="logs/app.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    init_db(app)
    app.register_blueprint(fraud_bp)

    return app
