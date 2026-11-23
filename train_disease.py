
import os
import json
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

MobileNetV2 = tf.keras.applications.MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# 1. Dataset path
DATASET_PATH = "dataset/potato_disease"

# 2. Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 10

# 3. Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# 4. Load validation dataset
validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# 5. Save class names
class_names = train_data.class_names

print("Disease Classes:", class_names)

os.makedirs("models", exist_ok=True)

with open("models/disease_classes.json", "w") as file:
    json.dump(class_names, file)

# 6. Improve dataset performance
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(
    buffer_size=AUTOTUNE
)

validation_data = validation_data.prefetch(
    buffer_size=AUTOTUNE
)

# 7. Load pretrained MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# 8. Build disease detection model
inputs = layers.Input(shape=(224, 224, 3))

x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)

x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# 9. Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 10. Train model
print("Training Potato Disease Model...")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# 11. Evaluate model
loss, accuracy = model.evaluate(validation_data)

print("Disease Model Accuracy:", accuracy)

# 12. Save model
model.save("models/potato_disease_model.keras")

print("Potato Disease Model Saved Successfully!")
