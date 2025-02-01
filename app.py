from flask import Flask
from flask_cors import CORS
from app.routes import main  # Assuming you have this for your routes

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app)

app.register_blueprint(main)

if __name__ == "__main__":
    app.run(debug=True)
