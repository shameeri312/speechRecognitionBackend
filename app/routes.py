from flask import Blueprint, request, jsonify
import numpy as np
from werkzeug.utils import secure_filename
import os
from .models import extract_features, load_model

# Initialize the Flask Blueprint
main = Blueprint("main", __name__)

# Ensure uploads folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create 'uploads/' if not exists

# Load the trained model
model = load_model("data/model.h5")

# Emotion labels
emotions = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]


# API route for emotion prediction
@main.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the file temporarily to extract features
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)  # Use the defined UPLOAD_FOLDER
    file.save(file_path)

    # Extract features from the uploaded file
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # Reshape for model input

    # Make prediction
    prediction = model.predict(features)
    print("--->", prediction)

    # Get the predicted emotion
    result = emotions[np.argmax(prediction)]

    return jsonify({"emotion": result})
