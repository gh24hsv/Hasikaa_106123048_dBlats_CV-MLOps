import os
import ssl
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping #type: ignore

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify() 
        return True
    except (IOError, SyntaxError, ValueError):
        return False

base_dir = "/Users/hsv/.cache/kagglehub/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/versions/1/PetImages"


cat_dir = os.path.join(base_dir, "Cat")
dog_dir = os.path.join(base_dir, "Dog")


cat_filenames = [os.path.join("Cat", filename) for filename in os.listdir(cat_dir) if is_valid_image(os.path.join(cat_dir, filename))][:1000]
dog_filenames = [os.path.join("Dog", filename) for filename in os.listdir(dog_dir) if is_valid_image(os.path.join(dog_dir, filename))][:1000]


cat_data = pd.DataFrame({"filename": cat_filenames, "label": "Cat"})
dog_data = pd.DataFrame({"filename": dog_filenames, "label": "Dog"})


data = pd.concat([cat_data, dog_data], ignore_index=True)


train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data["label"], shuffle=True, random_state=42)
test_data, val_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["label"], random_state=42)


train_data["filename"] = train_data["filename"].apply(lambda x: os.path.join(base_dir, x))
val_data["filename"] = val_data["filename"].apply(lambda x: os.path.join(base_dir, x))
test_data["filename"] = test_data["filename"].apply(lambda x: os.path.join(base_dir, x))


size = (224, 224) 
batch_size = 32

# Create ImageDataGenerator
idg = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)


train_idg = idg.flow_from_dataframe(
    train_data,
    x_col="filename",
    y_col="label",
    batch_size=batch_size,
    target_size=size,
    class_mode="categorical"
)

val_idg = idg.flow_from_dataframe(
    val_data,
    x_col="filename",
    y_col="label",
    batch_size=batch_size,
    target_size=size,
    shuffle=False,
    class_mode="categorical"
)

test_idg = idg.flow_from_dataframe(
    test_data,
    x_col="filename",
    y_col="label",
    batch_size=batch_size,
    target_size=size,
    shuffle=False,
    class_mode="categorical"
)


ssl._create_default_https_context = ssl._create_unverified_context
efficientnet_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights="imagenet")


for layer in efficientnet_model.layers:
    layer.trainable = False


flat = tf.keras.layers.Flatten()(efficientnet_model.output)
dense1 = tf.keras.layers.Dense(128, activation="relu")(flat)
dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
output = tf.keras.layers.Dense(2, activation="softmax")(dropout1)

final_model = tf.keras.models.Model(inputs=efficientnet_model.input, outputs=output)


final_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience=2, factor=0.5, min_lr=0.00001, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)


history = final_model.fit(
    train_idg,
    validation_data=val_idg,
    epochs=10,
    callbacks=[learning_rate_reduction, early_stopping],
    verbose=1
)


loss, acc = final_model.evaluate(test_idg, batch_size=batch_size, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")


save_dir = "./saved_model_efficientnet/"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "efficientnet_model.h5")
final_model.save(model_path)

print(f"Model saved to: {model_path}")

# Accuracy recorded = 98+%
