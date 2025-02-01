from flask import Flask

app = Flask(__name__)

from app.routes import main  # Import the routes after initializing the app
app.register_blueprint(main)  # Register the blueprint
