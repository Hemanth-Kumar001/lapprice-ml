import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import os
import math
import joblib
import pandas as pd
import numpy as np

_model = None
_model_path = None

def load_model(model_path='laptop_price_prediction_model.pkl'):
    global _model, _model_path
    if _model is None or model_path != _model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _model = joblib.load(model_path)
        _model_path = model_path
    return _model

FEATURE_COLUMNS = [
    "brand", "device_category", "cpu_brand", "cpu_family", "cpu_series",
    "cpu_model", "cpu_suffix", "cpu_core_count", "cpu_thread_count",
    "cpu_p_cores", "cpu_e_cores", "cpu_lp_e_cores", "ram_gb", "storage_gb",
    "display_width_px", "display_height_px", "display_size_inch",
    "gpu_brand", "gpu_series", "gpu_model", "gpu_vram_gb", "gpu_type",
    "os_name", "warranty_years", "pixels_mp", "ppi", "has_dgpu"
]

def engineer_features(data: dict) -> dict:
    """Compute derived features: pixels_mp, ppi, has_dgpu."""
    w = float(data['display_width_px'])
    h = float(data['display_height_px'])
    d = float(data['display_size_inch'])
    data['pixels_mp'] = (w * h) / 1_000_000
    data['ppi'] = math.sqrt(w**2 + h**2) / d
    data['has_dgpu'] = 1 if str(data.get('gpu_type', '')).strip().lower() == 'dedicated' else 0
    return data

def predict_price(form_data: dict, model_path: str = 'laptop_price_prediction_model.pkl') -> dict:
    """
    Build input DataFrame, run model prediction, return result dict.
    Raises ValueError for invalid inputs.
    """
    # Parse and validate numeric inputs
    try:
        cpu_core_count = int(form_data.get('cpu_core_count', 0))
        cpu_thread_count = int(form_data.get('cpu_thread_count', 0))
        cpu_p_cores = int(form_data.get('cpu_p_cores', 0))
        cpu_e_cores = int(form_data.get('cpu_e_cores', 0))
        cpu_lp_e_cores = int(form_data.get('cpu_lp_e_cores', 0))
        ram_gb = int(form_data.get('ram_gb', 0))
        storage_gb = int(form_data.get('storage_gb', 0))
        display_width_px = int(form_data.get('display_width_px', 0))
        display_height_px = int(form_data.get('display_height_px', 0))
        display_size_inch = float(form_data.get('display_size_inch', 0))
        gpu_vram_gb = float(form_data.get('gpu_vram_gb', 0))
        warranty_years = int(form_data.get('warranty_years', 1))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid numeric input: {e}")

    if display_size_inch <= 0:
        raise ValueError("Display size must be greater than 0")
    if display_width_px <= 0 or display_height_px <= 0:
        raise ValueError("Display resolution must be positive")
    if cpu_core_count <= 0:
        raise ValueError("CPU core count must be positive")

    data = {
        'brand': str(form_data.get('brand', '')).strip().lower(),
        'device_category': str(form_data.get('device_category', '')),
        'cpu_brand': str(form_data.get('cpu_brand', '')),
        'cpu_family': str(form_data.get('cpu_family', '')),
        'cpu_series': str(form_data.get('cpu_series', '')),
        'cpu_model': str(form_data.get('cpu_model', '')),
        'cpu_suffix': str(form_data.get('cpu_suffix', '')),
        'cpu_core_count': cpu_core_count,
        'cpu_thread_count': cpu_thread_count,
        'cpu_p_cores': cpu_p_cores,
        'cpu_e_cores': cpu_e_cores,
        'cpu_lp_e_cores': cpu_lp_e_cores,
        'ram_gb': ram_gb,
        'storage_gb': storage_gb,
        'display_width_px': display_width_px,
        'display_height_px': display_height_px,
        'display_size_inch': display_size_inch,
        'gpu_brand': str(form_data.get('gpu_brand', '')),
        'gpu_series': str(form_data.get('gpu_series', '')),
        'gpu_model': str(form_data.get('gpu_model', '')),
        'gpu_vram_gb': gpu_vram_gb,
        'gpu_type': str(form_data.get('gpu_type', '')),
        'os_name': str(form_data.get('os_name', '')),
        'warranty_years': warranty_years,
    }
    data = engineer_features(data)

    df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    model = load_model(model_path)
    prediction = float(model.predict(df)[0])
    prediction = max(0, prediction)  # floor at 0

    return {
        'predicted_price': round(prediction, 2),
        'formatted_price': f"₹{prediction:,.0f}",
        'pixels_mp': round(data['pixels_mp'], 3),
        'ppi': round(data['ppi'], 1),
        'has_dgpu': data['has_dgpu'],
    }
