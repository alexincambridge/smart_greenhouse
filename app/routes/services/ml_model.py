import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("data/models/pest_detection.h5")

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))  # Resize to match model input
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize
    return img

# Perform prediction
def predict_pest(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return "Infected" if prediction > 0.5 else "Healthy"

# Test with a new image
result = predict_pest("data/images/test.jpg")
print(f"🚨 Pest Detection Result: {result}")
