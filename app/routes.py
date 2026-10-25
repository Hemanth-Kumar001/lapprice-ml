import csv
import json
from pathlib import Path
from flask import Blueprint, current_app, jsonify, render_template, request
from .predictor import predict_price

main = Blueprint("main", __name__)

def _file(name):
    return Path(current_app.config["ROOT_DIR"]) / name

def _load_json(name):
    with _file(name).open(encoding="utf-8") as f:
        return json.load(f)

def _load_csv(name):
    with _file(name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def _dashboard_data():
    try:
        metrics = _load_json("model_metrics.json")
    except (OSError, ValueError):
        metrics = {}
    try:
        comparison = _load_csv("model_comparison.csv")
    except OSError:
        comparison = []
    try:
        feature_imp = _load_csv("feature_importance.csv")
    except OSError:
        feature_imp = []
    try:
        predictions = _load_csv("prediction_test_results.csv")
    except OSError:
        predictions = []
    return metrics, comparison, feature_imp, predictions

@main.route("/")
def home():
    metrics, comparison, feature_imp, _ = _dashboard_data()
    return render_template("home.html", metrics=metrics, comparison=comparison, feature_imp=feature_imp)

@main.route("/predict", methods=["GET", "POST"])
def predict():
    result, error, form_data = None, None, {}
    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            result = predict_price(form_data, current_app.config["MODEL_PATH"])
        except Exception as exc:
            error = str(exc)
    return render_template("predict.html", result=result, error=error, form_data=form_data)

@main.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    try:
        return jsonify({"success": True, "data": predict_price(data, current_app.config["MODEL_PATH"])})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        current_app.logger.exception("Prediction endpoint failed")
        return jsonify({"success": False, "error": "Prediction could not be completed. Check the model file and input values."}), 500

@main.route("/dashboard")
def dashboard():
    metrics, comparison, feature_imp, predictions = _dashboard_data()
    return render_template("dashboard.html", metrics=metrics, comparison=comparison, feature_imp=feature_imp, predictions=predictions)

@main.route("/api/dashboard")
def api_dashboard():
    metrics, comparison, feature_imp, predictions = _dashboard_data()
    return jsonify({"metrics": metrics, "model_comparison": comparison, "feature_importance": feature_imp[:15], "predictions": predictions})

@main.route("/about")
@main.route("/how-it-works")
def about():
    metrics, _, _, _ = _dashboard_data()
    return render_template("about.html", metrics=metrics)
