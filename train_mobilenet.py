import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "Processed_CrackDataset"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

MODEL_DIR = "models/mobilenet_v2"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DATA GENERATORS
# ============================================================

print()
print("Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42
)


validation_generator = validation_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "validation"),
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


print()
print("Class indices:")
print(train_generator.class_indices)

print()
print("Training images   :", train_generator.samples)
print("Validation images :", validation_generator.samples)
print("Test images       :", test_generator.samples)


# ============================================================
# LOAD MOBILENETV2
# ============================================================

print()
print("Loading MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

print("MobileNetV2 loaded successfully.")


# ============================================================
# FREEZE PRE-TRAINED MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 60)
print("MOBILENETV2 MODEL")
print("=" * 60)

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "Best_MobileNetV2_Crack_Detection.keras"
)


callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.00001
    )
]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING MOBILENETV2 TRAINING")
print("=" * 60)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "Final_MobileNetV2_Crack_Detection.keras"
)


model.save(
    final_model_path
)


print()
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    "Best model :",
    best_model_path
)

print(
    "Final model:",
    final_model_path
)


# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("=" * 60)
print("EVALUATING ON TEST DATA")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print()
print(
    "Test Loss     :",
    test_loss
)

print(
    "Test Accuracy :",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "MobileNetV2 Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "mobilenet_accuracy.png"
    ),
    dpi=150
)

plt.show()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "MobileNetV2 Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "mobilenet_loss.png"
    ),
    dpi=150
)

plt.show()


print()
print("=" * 60)
print("MOBILENETV2 TRAINING COMPLETED")
print("=" * 60)