from flask import Flask

def create_app():
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    app.config['MODEL_PATH'] = 'laptop_price_prediction_model.pkl'
    app.config['SECRET_KEY'] = 'laptop-price-pred-2024'
    
    from .routes import main
    app.register_blueprint(main)
    
    return app
