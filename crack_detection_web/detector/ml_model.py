from functools import lru_cache

import numpy as np

from django.conf import settings

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMAGE_SIZE = 128


# ============================================================
# LOAD EXISTING CNN MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_model():

    model_path = (
        settings.BASE_DIR.parent
        / "models"
        / "baseline_v1"
        / "Best_Crack_Detection_Model.keras"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Crack detection model not found at:\n{model_path}"
        )

    print("=" * 60)
    print("Loading Crack Detection CNN model...")
    print("=" * 60)

    model = load_model(model_path)

    print("Crack Detection CNN model loaded successfully.")
    print("=" * 60)

    return model


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image_path):

    # Load image
    image = load_img(
        image_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert image to NumPy array
    image_array = img_to_array(image)

    # Normalize pixel values
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Load model
    model = get_model()

    # Make prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    prediction_value = float(
        prediction[0][0]
    )

    # ========================================================
    # MODEL CLASS MAPPING
    #
    # Crack   = 0
    # NoCrack = 1
    # ========================================================

    if prediction_value >= 0.5:

        result = "No Crack"

        confidence = (
            prediction_value * 100
        )

    else:

        result = "Crack Detected"

        confidence = (
            (1 - prediction_value) * 100
        )

    return {
        "result": result,
        "confidence": round(confidence, 2),
        "raw_prediction": prediction_value,
    }