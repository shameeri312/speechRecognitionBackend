import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Rename last column if needed (replace "happy" with "emotion")
df = df.rename(columns={"happy": "emotion"})

# Print first few rows to verify correct column names
print(df.head())


# Function to load data
def load_data():
    # Extract features (all columns except 'emotion')
    X = df.drop(columns=["emotion"]).values

    # Extract labels (convert to categorical)
    y = df["emotion"].values
    y = to_categorical(y, num_classes=8)  # Assuming 8 emotion classes

    return np.array(X), np.array(y)


# Load and prepare the data
X, y = load_data()

# Reshape X for LSTM input
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape to (samples, time_steps, features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define LSTM model
model = Sequential(
    [
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dense(32, activation="relu"),
        Dense(8, activation="softmax"),  # 8 emotion classes
    ]
)

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Print the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Save the trained model
model.save("data/model.h5")  # Save model for future use
