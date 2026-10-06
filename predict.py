import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/baseline_v1/Best_Crack_Detection_Model.keras"

IMAGE_SIZE = 128


# ============================================================
# Load trained model
# ============================================================

print("Loading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# Prediction function
# ============================================================

def predict_image(image_path):

    if not os.path.exists(image_path):
        print("ERROR: Image not found:")
        print(image_path)
        return

    # Load image
    image = load_img(
        image_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert image to array
    image_array = img_to_array(image)

    # Normalize pixels
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array, verbose=0)

    prediction_value = float(prediction[0][0])

    # Your model class mapping:
    # Crack = 0
    # NoCrack = 1

    if prediction_value >= 0.5:

        result = "No Crack"
        confidence = prediction_value * 100

    else:

        result = "Crack Detected"
        confidence = (1 - prediction_value) * 100


    # Display result
    print("=" * 60)
    print("       SURFACE CRACK DETECTION")
    print("=" * 60)

    print("Image       :", image_path)
    print("Prediction  :", result)
    print("Confidence  :", f"{confidence:.2f}%")

    print("=" * 60)


    # Display image
    plt.figure(figsize=(7, 5))

    plt.imshow(image)

    plt.title(
        f"{result} - Confidence: {confidence:.2f}%"
    )

    plt.axis("off")

    plt.show()


# ============================================================
# Test image
# ============================================================

image_path = input(
    "\nEnter image path: "
)

predict_image(image_path)