
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

POTATO_DATASET = BASE_DIR / "dataset" / "PlantVillage"

# CSV is inside dataset/PlantVillage
CROP_DATASET = BASE_DIR / "dataset" / "PlantVillage" / "Crop_recommendation.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 10

layers = tf.keras.layers
models = tf.keras.models

MobileNetV2 = tf.keras.applications.MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


# ==========================================
# 2. POTATO DISEASE DETECTION
# ==========================================

print("\n================================")
print("POTATO DISEASE MODEL TRAINING")
print("================================")

if not POTATO_DATASET.exists():
    raise FileNotFoundError(
        f"Potato dataset not found: {POTATO_DATASET}"
    )

allowed_classes = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight"
]

actual_classes = {
    folder.name
    for folder in POTATO_DATASET.iterdir()
    if folder.is_dir()
}

if actual_classes != set(allowed_classes):
    raise ValueError(
        f"Expected classes: {allowed_classes}\n"
        f"Found classes: {sorted(actual_classes)}"
    )


# Load training images

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(POTATO_DATASET),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=allowed_classes
)


# Load validation images

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(POTATO_DATASET),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=allowed_classes
)

class_names = train_ds.class_names

print("Potato classes:", class_names)


# Save class names

with open(MODEL_DIR / "potato_classes.json", "w") as file:
    json.dump(class_names, file)


# Optimize dataset loading

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# Data augmentation

augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# Load pretrained MobileNetV2

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# Build model

inputs = layers.Input(shape=(224, 224, 3))

x = augmentation(inputs)
x = layers.Lambda(preprocess_input)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    3,
    activation="softmax"
)(x)

potato_model = models.Model(inputs, outputs)


# Compile model

potato_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Callbacks

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "best_potato_model.keras"),
        monitor="val_accuracy",
        save_best_only=True
    )
]


# Train potato model

history = potato_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# Save final model

potato_model.save(
    str(MODEL_DIR / "potato_disease_model.keras")
)


# Evaluate model

loss, accuracy = potato_model.evaluate(val_ds)

print("\nPotato Disease Validation Accuracy:",
      round(accuracy * 100, 2), "%")


# Save accuracy graph

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Potato Disease Detection Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(BASE_DIR / "potato_accuracy.png")
plt.close()

print("Potato disease model saved successfully!")


# ==========================================
# 3. CROP RECOMMENDATION MODEL
# ==========================================

print("\n================================")
print("CROP RECOMMENDATION MODEL TRAINING")
print("================================")


# Check CSV path

print("\nChecking CSV path:")
print(CROP_DATASET)

print("File exists:", CROP_DATASET.is_file())


if not CROP_DATASET.is_file():
    raise FileNotFoundError(
        f"""
Crop recommendation CSV not found!

Expected location:
{CROP_DATASET}

Please place Crop_recommendation.csv
inside dataset/PlantVillage/
"""
    )


# Load CSV only once

df = pd.read_csv(CROP_DATASET)

print("\nCSV loaded successfully!")
print("Dataset shape:", df.shape)
print("\nCSV columns:")
print(df.columns.tolist())


# Required features

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

required_columns = features + ["label"]


# Check required columns

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV: {missing_columns}"
    )


# Clean dataset

df = df.dropna(subset=required_columns).copy()

for col in features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(subset=features + ["label"])

if df.empty:
    raise ValueError("Crop dataset contains no valid rows!")

print("\nClean dataset shape:", df.shape)

print("\nAvailable crops:")
print(df["label"].unique())


# Input and output

X = df[features]
y = df["label"]


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Random Forest Classifier

crop_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# Train crop recommendation model

print("\nTraining crop recommendation model...")

crop_model.fit(X_train, y_train)


# Predictions

predictions = crop_model.predict(X_test)


# Evaluate model

crop_accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nCrop Recommendation Accuracy:",
      round(crop_accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# Save crop model

joblib.dump(
    crop_model,
    MODEL_DIR / "crop_recommendation_model.pkl"
)

joblib.dump(
    features,
    MODEL_DIR / "crop_features.pkl"
)


print("\nCrop recommendation model saved successfully!")


# ==========================================
# 4. TRAINING COMPLETED
# ==========================================

print("\n================================")
print("BOTH MODELS TRAINED SUCCESSFULLY!")
print("================================")

print("1. Potato Disease Detection")
print("2. Crop Recommendation")

print("\nSaved models inside:", MODEL_DIR)

print("\nGenerated files:")

for file in MODEL_DIR.iterdir():
    print("-", file.name)

print("\nALL TRAINING COMPLETED SUCCESSFULLY!")
