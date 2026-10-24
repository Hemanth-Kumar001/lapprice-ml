import os
import json
import csv
import math
from flask import Blueprint, render_template, request, jsonify, current_app

main = Blueprint('main', __name__)

def _load_json(filename):
    path = filename
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), filename)
    with open(path, 'r') as f:
        return json.load(f)

def _load_csv(filename):
    path = filename
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), filename)
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

@main.route('/')
def home():
    try:
        metrics = _load_json('model_metrics.json')
        comparison = _load_csv('model_comparison.csv')
        feature_imp = _load_csv('feature_importance.csv')
    except Exception:
        metrics, comparison, feature_imp = {}, [], []
    return render_template('home.html', metrics=metrics, comparison=comparison, feature_imp=feature_imp)

@main.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    error = None
    form_data = {}
    if request.method == 'POST':
        from .predictor import predict_price
        model_path = os.path.join(os.getcwd(), current_app.config['MODEL_PATH'])
        form_data = request.form.to_dict()
        try:
            result = predict_price(form_data, model_path)
        except Exception as e:
            error = str(e)
    return render_template('predict.html', result=result, error=error, form_data=form_data)

@main.route('/api/predict', methods=['POST'])
def api_predict():
    from .predictor import predict_price
    model_path = os.path.join(os.getcwd(), current_app.config['MODEL_PATH'])
    try:
        data = request.get_json() or request.form.to_dict()
        result = predict_price(data, model_path)
        return jsonify({'success': True, 'data': result})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': 'Prediction failed: ' + str(e)}), 500

@main.route('/dashboard')
def dashboard():
    try:
        metrics = _load_json('model_metrics.json')
        comparison = _load_csv('model_comparison.csv')
        feature_imp = _load_csv('feature_importance.csv')
        predictions = _load_csv('prediction_test_results.csv')
    except Exception as e:
        metrics, comparison, feature_imp, predictions = {}, [], [], []
    return render_template('dashboard.html',
                           metrics=metrics,
                           comparison=comparison,
                           feature_imp=feature_imp,
                           predictions=predictions)

@main.route('/api/dashboard')
def api_dashboard():
    try:
        metrics = _load_json('model_metrics.json')
        comparison = _load_csv('model_comparison.csv')
        feature_imp = _load_csv('feature_importance.csv')
        predictions = _load_csv('prediction_test_results.csv')
        return jsonify({
            'metrics': metrics,
            'model_comparison': comparison,
            'feature_importance': feature_imp[:15],
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/how-it-works')
@main.route('/about')
def about():
    try:
        metrics = _load_json('model_metrics.json')
    except Exception:
        metrics = {}
    return render_template('about.html', metrics=metrics)
