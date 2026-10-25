"""
Automated tests for the Laptop Price Prediction Flask application.
Tests: model loading, valid prediction, invalid inputs, all routes, dashboard data.
"""
import pytest
import json
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def app():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app import create_app
    application = create_app()
    application.config['TESTING'] = True
    return application

@pytest.fixture
def client(app):
    return app.test_client()

# 1. Model loading test
def test_model_loads():
    from app.predictor import load_model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'laptop_price_prediction_model.pkl')
    model = load_model(model_path)
    assert model is not None

# 2. Valid prediction test
def test_valid_prediction():
    from app.predictor import predict_price
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'laptop_price_prediction_model.pkl')
    form_data = {
        'brand': 'lenovo',
        'device_category': 'Thin & Light',
        'cpu_brand': 'Intel',
        'cpu_family': 'Core',
        'cpu_series': 'Core I5',
        'cpu_model': '13420H',
        'cpu_suffix': 'H',
        'cpu_core_count': '8',
        'cpu_thread_count': '12',
        'cpu_p_cores': '4',
        'cpu_e_cores': '4',
        'cpu_lp_e_cores': '0',
        'ram_gb': '16',
        'storage_gb': '512',
        'display_width_px': '1920',
        'display_height_px': '1080',
        'display_size_inch': '15.6',
        'gpu_brand': 'Intel',
        'gpu_series': 'UHD',
        'gpu_model': 'UHD Graphics',
        'gpu_vram_gb': '0',
        'gpu_type': 'Integrated',
        'os_name': 'Windows',
        'warranty_years': '1',
    }
    result = predict_price(form_data, model_path)
    assert 'predicted_price' in result
    assert result['predicted_price'] > 0
    assert 'formatted_price' in result
    assert '₹' in result['formatted_price']
    assert 'pixels_mp' in result
    assert 'ppi' in result

# 3. Invalid input test - zero display size
def test_invalid_display_size():
    from app.predictor import predict_price
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'laptop_price_prediction_model.pkl')
    form_data = {
        'brand': 'lenovo', 'device_category': 'Thin & Light',
        'cpu_brand': 'Intel', 'cpu_family': 'Core', 'cpu_series': 'Core I5',
        'cpu_model': '13420H', 'cpu_suffix': 'H',
        'cpu_core_count': '8', 'cpu_thread_count': '12',
        'cpu_p_cores': '4', 'cpu_e_cores': '4', 'cpu_lp_e_cores': '0',
        'ram_gb': '16', 'storage_gb': '512',
        'display_width_px': '1920', 'display_height_px': '1080',
        'display_size_inch': '0',  # invalid
        'gpu_brand': 'Intel', 'gpu_series': 'UHD', 'gpu_model': 'UHD Graphics',
        'gpu_vram_gb': '0', 'gpu_type': 'Integrated',
        'os_name': 'Windows', 'warranty_years': '1',
    }
    with pytest.raises(ValueError):
        predict_price(form_data, model_path)

# 4. Invalid input test - non-numeric
def test_invalid_non_numeric():
    from app.predictor import predict_price
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'laptop_price_prediction_model.pkl')
    form_data = {
        'brand': 'lenovo', 'device_category': 'Thin & Light',
        'cpu_brand': 'Intel', 'cpu_family': 'Core', 'cpu_series': 'Core I5',
        'cpu_model': '13420H', 'cpu_suffix': 'H',
        'cpu_core_count': 'abc',  # invalid
        'cpu_thread_count': '12',
        'cpu_p_cores': '4', 'cpu_e_cores': '4', 'cpu_lp_e_cores': '0',
        'ram_gb': '16', 'storage_gb': '512',
        'display_width_px': '1920', 'display_height_px': '1080',
        'display_size_inch': '15.6',
        'gpu_brand': 'Intel', 'gpu_series': 'UHD', 'gpu_model': 'UHD Graphics',
        'gpu_vram_gb': '0', 'gpu_type': 'Integrated',
        'os_name': 'Windows', 'warranty_years': '1',
    }
    with pytest.raises(ValueError):
        predict_price(form_data, model_path)

# 5. Route tests
def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Laptop' in response.data

def test_predict_get_route(client):
    response = client.get('/predict')
    assert response.status_code == 200

def test_predict_post_valid(client):
    response = client.post('/predict', data={
        'brand': 'lenovo',
        'device_category': 'Thin & Light',
        'cpu_brand': 'Intel',
        'cpu_family': 'Core',
        'cpu_series': 'Core I5',
        'cpu_model': '13420H',
        'cpu_suffix': 'H',
        'cpu_core_count': '8',
        'cpu_thread_count': '12',
        'cpu_p_cores': '4',
        'cpu_e_cores': '4',
        'cpu_lp_e_cores': '0',
        'ram_gb': '16',
        'storage_gb': '512',
        'display_width_px': '1920',
        'display_height_px': '1080',
        'display_size_inch': '15.6',
        'gpu_brand': 'Intel',
        'gpu_series': 'UHD',
        'gpu_model': 'UHD Graphics',
        'gpu_vram_gb': '0',
        'gpu_type': 'Integrated',
        'os_name': 'Windows',
        'warranty_years': '1',
    })
    assert response.status_code == 200

def test_dashboard_route(client):
    response = client.get('/dashboard')
    assert response.status_code == 200

def test_about_route(client):
    response = client.get('/about')
    assert response.status_code == 200

# 6. Dashboard API test
def test_api_dashboard(client):
    response = client.get('/api/dashboard')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'metrics' in data
    assert 'model_comparison' in data
    assert 'feature_importance' in data
    assert 'predictions' in data

# 7. API predict test
def test_api_predict(client):
    response = client.post('/api/predict', 
        json={
            'brand': 'lenovo',
            'device_category': 'Thin & Light',
            'cpu_brand': 'Intel',
            'cpu_family': 'Core',
            'cpu_series': 'Core I5',
            'cpu_model': '13420H',
            'cpu_suffix': 'H',
            'cpu_core_count': '8',
            'cpu_thread_count': '12',
            'cpu_p_cores': '4',
            'cpu_e_cores': '4',
            'cpu_lp_e_cores': '0',
            'ram_gb': '16',
            'storage_gb': '512',
            'display_width_px': '1920',
            'display_height_px': '1080',
            'display_size_inch': '15.6',
            'gpu_brand': 'Intel',
            'gpu_series': 'UHD',
            'gpu_model': 'UHD Graphics',
            'gpu_vram_gb': '0',
            'gpu_type': 'Integrated',
            'os_name': 'Windows',
            'warranty_years': '1',
        })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True
