import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model('data/model.h5')

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Extract MFCC, Chroma, and Mel features from audio file
def extract_features(audio_path):
    # Load the audio file with librosa
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)
    
    # Extract MFCC features (Mel-frequency cepstral coefficients)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    
    # Extract Chroma features (chroma short-time Fourier transform)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    
    # Extract Mel Spectrogram features
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    
    # Combine all features into a single array
    return np.hstack([mfcc, chroma, mel])

# Predict emotion from the audio file
def predict_emotion(audio_path):
    # Extract features from the audio
    features = extract_features(audio_path)
    
    # Reshape for prediction
    features = features.reshape(1, -1)  # Reshaping for prediction

    # Predict the emotion using the trained model
    prediction = model.predict(features)
    
    # Return the emotion with the highest probability
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion
