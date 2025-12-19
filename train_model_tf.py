import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load Data
print("Loading data...")
try:
    data = pd.read_csv("hand_data_3d.csv") # Make sure to use your 3D data!
except FileNotFoundError:
    print("Error: csv not found.")
    exit()

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 2. Encode Labels (A -> 0, B -> 1, etc.)
# Neural Networks can't understand strings like "A", they need numbers.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the encoder so the App knows that 0 = "A"
with open("label_encoder.p", "wb") as f:
    pickle.dump(label_encoder, f)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Build the Neural Network (The "Brain")
model = tf.keras.models.Sequential([
    # Input Layer: 63 coordinates
    tf.keras.layers.Input(shape=(63,)),
    
    # Hidden Layer 1: 128 Neurons with ReLU activation
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Prevents overfitting
    
    # Hidden Layer 2: 64 Neurons
    tf.keras.layers.Dense(64, activation='relu'),
    
    # Output Layer: One neuron per letter (softmax for probability)
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# 5. Compile and Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training Neural Network...")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 6. Save
model.save("asl_model_tf.keras")
print("Model saved as 'asl_model_tf.keras'")