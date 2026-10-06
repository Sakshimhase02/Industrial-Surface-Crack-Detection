import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/baseline_v1/Best_Crack_Detection_Model.keras"

IMAGE_SIZE = 128

LAST_CONV_LAYER = "conv2d_3"

OUTPUT_DIR = "results/gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

def load_and_prepare_image(image_path):

    image = load_img(
        image_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = img_to_array(image)

    image_array = image_array / 255.0

    input_image = np.expand_dims(
        image_array,
        axis=0
    ).astype(np.float32)

    return image, input_image


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_conv_layer():

    print()
    print("Searching for convolutional layer...")

    for layer in reversed(model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            print(
                "Grad-CAM layer:",
                layer.name
            )

            return layer.name

    raise ValueError(
        "No Conv2D layer found in the model."
    )


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(image_path):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image, input_image = load_and_prepare_image(
        image_path
    )


    # --------------------------------------------------------
    # Find target Conv2D layer
    # --------------------------------------------------------

    target_layer_name = find_conv_layer()


    # --------------------------------------------------------
    # Gradient Tape
    #
    # We manually execute every layer of the original model.
    # This keeps the convolution output connected directly
    # to the final prediction.
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        x = tf.convert_to_tensor(
            input_image,
            dtype=tf.float32
        )

        conv_output = None


        for layer in model.layers:

            # Skip InputLayer
            if isinstance(
                layer,
                tf.keras.layers.InputLayer
            ):
                continue


            # Run layer
            x = layer(
                x,
                training=False
            )


            # Save output of target Conv2D layer
            if layer.name == target_layer_name:

                conv_output = x


        # Final model prediction
        predictions = x


        # ----------------------------------------------------
        # Check target layer
        # ----------------------------------------------------

        if conv_output is None:

            raise RuntimeError(
                "Could not obtain output from "
                + target_layer_name
            )


        # ----------------------------------------------------
        # Model mapping:
        #
        # Crack   = 0
        # NoCrack = 1
        #
        # Sigmoid output = NoCrack probability
        # ----------------------------------------------------

        prediction_value = predictions[:, 0]


        if float(prediction_value[0]) >= 0.5:

            result = "No Crack"

            # Target NoCrack probability
            target = prediction_value

        else:

            result = "Crack Detected"

            # Crack probability
            target = 1.0 - prediction_value


    # ========================================================
    # CALCULATE GRADIENTS
    # ========================================================

    gradients = tape.gradient(
        target,
        conv_output
    )


    if gradients is None:

        raise RuntimeError(
            "Gradients are still None. "
            "Grad-CAM could not calculate gradients."
        )


    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )


    # Remove batch dimension
    conv_output = conv_output[0]


    # ========================================================
    # CREATE HEATMAP
    # ========================================================

    heatmap = tf.reduce_sum(
        conv_output * pooled_gradients,
        axis=-1
    )


    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )


    # Normalize
    max_value = tf.reduce_max(
        heatmap
    )


    if float(max_value) > 0:

        heatmap = heatmap / max_value


    heatmap = heatmap.numpy()


    # ========================================================
    # CONFIDENCE
    # ========================================================

    prediction_value = float(
        predictions[0][0]
    )


    if prediction_value >= 0.5:

        confidence = prediction_value * 100

    else:

        confidence = (
            1 - prediction_value
        ) * 100


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("              GRAD-CAM ANALYSIS")
    print("=" * 60)

    print(
        "Image      :",
        image_path
    )

    print(
        "Prediction :",
        result
    )

    print(
        "Confidence :",
        f"{confidence:.2f}%"
    )

    print(
        "Conv Layer :",
        target_layer_name
    )

    print("=" * 60)


    # ========================================================
    # DISPLAY ORIGINAL IMAGE
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )


    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        image
    )

    plt.title(
        f"Original Image\n{result}"
    )

    plt.axis(
        "off"
    )


    # ========================================================
    # DISPLAY GRAD-CAM
    # ========================================================

    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        image
    )

    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.5
    )

    plt.title(
        "Grad-CAM\nImportant Regions"
    )

    plt.axis(
        "off"
    )


    plt.tight_layout()


    # ========================================================
    # SAVE RESULT
    # ========================================================

    filename = os.path.basename(
        image_path
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "gradcam_" + filename
    )


    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )


    print()
    print(
        "Grad-CAM result saved to:"
    )

    print(
        output_path
    )


    # ========================================================
    # SHOW RESULT
    # ========================================================

    plt.show()


# ============================================================
# MAIN
# ============================================================

print()

image_path = input(
    "Enter image path: "
).strip()


if not os.path.exists(image_path):

    print()
    print("ERROR: Image not found:")
    print(image_path)

else:

    make_gradcam(
        image_path
    )