import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
columns = ['Area', 'Perimeter', 'Compactness', 'Length of Kernel', 'Width of Kernel', 'Asymmetry Coefficient', 'Length of Kernel Groove', 'Class']
data = pd.read_csv(url, delim_whitespace=True, names=columns)

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert classes to 0, 1, 2
y_categorical = to_categorical(y_encoded, num_classes=3)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the feedforward neural network
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Input layer with 64 neurons
    Dense(32, activation='relu'),                              # Hidden layer with 32 neurons
    Dense(3, activation='softmax')                             # Output layer with 3 neurons (3 classes)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict a new sample
new_sample = np.array([[15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22]])
new_sample_scaled = scaler.transform(new_sample)
predicted_class = label_encoder.inverse_transform([np.argmax(model.predict(new_sample_scaled))])
print(f"Predicted Class for New Sample: {predicted_class[0]}")
