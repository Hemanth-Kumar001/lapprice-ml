import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import math
from pathlib import Path
import joblib
import pandas as pd

_MODEL = None
_MODEL_PATH = None

FEATURE_COLUMNS = [
    "brand", "device_category", "cpu_brand", "cpu_family", "cpu_series",
    "cpu_model", "cpu_suffix", "cpu_core_count", "cpu_thread_count",
    "cpu_p_cores", "cpu_e_cores", "cpu_lp_e_cores", "ram_gb", "storage_gb",
    "display_width_px", "display_height_px", "display_size_inch",
    "gpu_brand", "gpu_series", "gpu_model", "gpu_vram_gb", "gpu_type",
    "os_name", "warranty_years", "pixels_mp", "ppi", "has_dgpu"
]

def load_model(model_path=None):
    global _MODEL, _MODEL_PATH
    path = str(Path(model_path) if model_path else Path(__file__).resolve().parent.parent / "laptop_price_prediction_model.pkl")
    if _MODEL is None or path != _MODEL_PATH:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Trained model file not found: {path}")
        _MODEL = joblib.load(path)
        _MODEL_PATH = path
    return _MODEL

def engineer_features(data):
    w, h, d = float(data["display_width_px"]), float(data["display_height_px"]), float(data["display_size_inch"])
    data["pixels_mp"] = (w * h) / 1_000_000
    data["ppi"] = math.sqrt(w**2 + h**2) / d
    data["has_dgpu"] = 1 if str(data.get("gpu_type", "")).strip().lower() == "dedicated" else 0
    return data

def predict_price(form_data, model_path=None):
    try:
        numeric = {
            "cpu_core_count": int(form_data.get("cpu_core_count", 0)),
            "cpu_thread_count": int(form_data.get("cpu_thread_count", 0)),
            "cpu_p_cores": int(form_data.get("cpu_p_cores", 0)),
            "cpu_e_cores": int(form_data.get("cpu_e_cores", 0)),
            "cpu_lp_e_cores": int(form_data.get("cpu_lp_e_cores", 0)),
            "ram_gb": int(form_data.get("ram_gb", 0)),
            "storage_gb": int(form_data.get("storage_gb", 0)),
            "display_width_px": int(form_data.get("display_width_px", 0)),
            "display_height_px": int(form_data.get("display_height_px", 0)),
            "display_size_inch": float(form_data.get("display_size_inch", 0)),
            "gpu_vram_gb": float(form_data.get("gpu_vram_gb", 0)),
            "warranty_years": int(form_data.get("warranty_years", 1)),
        }
    except (ValueError, TypeError) as exc:
        raise ValueError("Please enter valid numbers in all hardware fields.") from exc
    if numeric["display_size_inch"] <= 0:
        raise ValueError("Display size must be greater than zero.")
    if numeric["display_width_px"] <= 0 or numeric["display_height_px"] <= 0:
        raise ValueError("Display resolution must be positive.")
    if numeric["cpu_core_count"] <= 0:
        raise ValueError("CPU core count must be greater than zero.")
    if numeric["ram_gb"] <= 0 or numeric["storage_gb"] <= 0:
        raise ValueError("RAM and storage must be greater than zero.")

    data = {
        "brand": str(form_data.get("brand", "Other")).strip().lower(),
        "device_category": str(form_data.get("device_category", "Notebook")),
        "cpu_brand": str(form_data.get("cpu_brand", "Intel")),
        "cpu_family": str(form_data.get("cpu_family", "Core")),
        "cpu_series": str(form_data.get("cpu_series", "Core i5")),
        "cpu_model": str(form_data.get("cpu_model", "Unknown")),
        "cpu_suffix": str(form_data.get("cpu_suffix", "")),
        "gpu_brand": str(form_data.get("gpu_brand", "Intel")),
        "gpu_series": str(form_data.get("gpu_series", "Integrated")),
        "gpu_model": str(form_data.get("gpu_model", "Integrated Graphics")),
        "gpu_type": str(form_data.get("gpu_type", "Integrated")),
        "os_name": str(form_data.get("os_name", "Windows")),
        **numeric,
    }
    data = engineer_features(data)
    frame = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    prediction = max(0.0, float(load_model(model_path).predict(frame)[0]))
    return {
        "predicted_price": round(prediction, 2),
        "formatted_price": f"₹{prediction:,.0f}",
        "pixels_mp": round(data["pixels_mp"], 3),
        "ppi": round(data["ppi"], 1),
        "has_dgpu": data["has_dgpu"],
    }
