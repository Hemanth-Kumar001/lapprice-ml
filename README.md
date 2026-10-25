# LapPrice AI — Laptop Price Prediction

🌐 **Live Website:** [Open LapPrice AI](https://lapprice-ml.onrender.com) · 💻 **Source Code:** [GitHub Repository](https://github.com/Hemanth-Kumar001/lapprice-ml)

An AI-powered web application for predicting laptop prices in the Indian market using a Tuned Gradient Boosting model with **92.8% R² accuracy**.

## Overview

This Flask application uses a scikit-learn Gradient Boosting pipeline trained on 992 real laptop specifications from the Indian market. Enter laptop hardware specs and get an instant price estimate in INR (₹).

**Key stats:**
- Model: Tuned Gradient Boosting Regressor
- R² Score: 92.8% (test set)
- MAE: ₹11,075
- Training data: 793 laptops
- Features: 27 (including 3 engineered)

---

## Prerequisites

- **Python 3.11** (recommended via Anaconda/Miniconda)
- **scikit-learn 1.6.1** (must match the training environment for model compatibility)

---

## Installation

### 1. Clone / Navigate to project

```bash
cd "c:\New folder\Laptop-Price-Prediction"
```

### 2. Create conda environment

```bash
conda create -n laptop-price python=3.11 -y
conda activate laptop-price
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

The app will start at **http://localhost:5000**

To run on a specific port:
```bash
python app.py  # default port 5000
```

For production use:
```bash
flask --app app run --host=0.0.0.0 --port=5000
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: 10 tests passing covering model loading, predictions, validation, and all routes.

```
tests/test_app.py::test_model_loads           PASSED
tests/test_app.py::test_valid_prediction       PASSED
tests/test_app.py::test_invalid_display_size   PASSED
tests/test_app.py::test_invalid_non_numeric    PASSED
tests/test_app.py::test_home_route             PASSED
tests/test_app.py::test_predict_get_route      PASSED
tests/test_app.py::test_predict_post_valid     PASSED
tests/test_app.py::test_dashboard_route        PASSED
tests/test_app.py::test_about_route            PASSED
tests/test_app.py::test_api_dashboard          PASSED
tests/test_app.py::test_api_predict            PASSED
```

---

## Project Structure

```
Laptop-Price-Prediction/
├── app/                            # Flask application package
│   ├── __init__.py                 # App factory (create_app)
│   ├── routes.py                   # URL routes and view functions
│   ├── predictor.py                # Model loading & prediction logic
│   ├── static/
│   │   ├── assets/                 # Helios HTML5UP theme assets
│   │   │   ├── css/
│   │   │   ├── js/
│   │   │   └── webfonts/
│   │   ├── css/
│   │   │   └── custom.css          # Custom styles
│   │   ├── js/
│   │   │   ├── predict.js          # Prediction form logic
│   │   │   └── dashboard.js        # Chart.js chart initialization
│   │   └── images/
│   │       └── laptop_ml_results.png
│   └── templates/
│       ├── base.html               # Base Jinja2 template (Helios nav/footer)
│       ├── home.html               # Landing page with stats & features
│       ├── predict.html            # Prediction form & results
│       ├── dashboard.html          # ML metrics, charts, model comparison
│       └── about.html              # Project info, pipeline, tech stack
├── tests/
│   ├── __init__.py
│   └── test_app.py                 # Pytest test suite (11 tests)
├── app.py                          # Entry point (sets cwd, runs Flask)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore
├── laptop_price_prediction_model.pkl  # Trained sklearn pipeline
├── model_metrics.json              # Model performance metrics
├── model_comparison.csv            # 8-model comparison results
├── feature_importance.csv          # Feature importance scores
├── prediction_test_results.csv     # Actual vs predicted test pairs
└── Indian Laptop Specifications Dataset.csv
```

---

## Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Landing page with model stats, feature overview, CTA |
| Predict | `/predict` | Prediction form with all 27 inputs; shows price result |
| Dashboard | `/dashboard` | Charts, model comparison table, feature importance |
| About | `/about` | Dataset info, pipeline, feature engineering, tech stack |

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/` | Home page |
| `GET/POST` | `/predict` | Prediction form (POST returns result) |
| `POST` | `/api/predict` | JSON API: accepts spec JSON, returns predicted price |
| `GET` | `/dashboard` | Dashboard page |
| `GET` | `/api/dashboard` | JSON API: returns all dashboard data |
| `GET` | `/about` | About page |

### API Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "lenovo",
    "device_category": "Thin & Light",
    "cpu_brand": "Intel",
    "cpu_family": "Core",
    "cpu_series": "Core I5",
    "cpu_model": "13420H",
    "cpu_suffix": "H",
    "cpu_core_count": "8",
    "cpu_thread_count": "12",
    "cpu_p_cores": "4",
    "cpu_e_cores": "4",
    "cpu_lp_e_cores": "0",
    "ram_gb": "16",
    "storage_gb": "512",
    "display_width_px": "1920",
    "display_height_px": "1080",
    "display_size_inch": "15.6",
    "gpu_brand": "Intel",
    "gpu_series": "UHD",
    "gpu_model": "UHD Graphics",
    "gpu_vram_gb": "0",
    "gpu_type": "Integrated",
    "os_name": "Windows",
    "warranty_years": "1"
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "predicted_price": 68500.0,
    "formatted_price": "₹68,500",
    "pixels_mp": 2.074,
    "ppi": 141.2,
    "has_dgpu": 0
  }
}
```

---

## Engineered Features

| Feature | Formula | Importance |
|---------|---------|------------|
| `pixels_mp` | `(width × height) / 1,000,000` | 12.0% |
| `ppi` | `√(width² + height²) / size_inch` | 4.4% |
| `has_dgpu` | `1 if gpu_type == "Dedicated" else 0` | ~5% |

---

## License

MIT License. Helios theme by [HTML5 UP](https://html5up.net) (CCA 3.0).
