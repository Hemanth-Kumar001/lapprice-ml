from flask import Flask
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = __import__("os").environ.get("SECRET_KEY", "lapprice-local-dev-key")
    app.config["ROOT_DIR"] = str(ROOT_DIR)
    app.config["MODEL_PATH"] = str(ROOT_DIR / "laptop_price_prediction_model.pkl")

    from .routes import main
    app.register_blueprint(main)
    return app

# Supports Render's current start command: gunicorn app:app
app = create_app()
