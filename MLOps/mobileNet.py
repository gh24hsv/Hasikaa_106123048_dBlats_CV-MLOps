import os
import ssl
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping #type: ignore

#pre-verifying image input using Pillow because noted interruption/exit of code mid of training model otherwise
def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()  
        return True
    except (IOError, SyntaxError, ValueError):
        return False

# Path to the dataset extracted from Kaggle (12500 images for cat and dog available each 
# but noted that around 1000 enough for high accuracy of model)
base_dir = "/Users/hsv/.cache/kagglehub/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/versions/1/PetImages"

# Directories for cats and dogs
cat_dir = os.path.join(base_dir, "Cat")
dog_dir = os.path.join(base_dir, "Dog")

# List of filenames for cats and dogs (using a subset for faster training of model)
cat_filenames = [os.path.join("Cat", filename) for filename in os.listdir(cat_dir) if is_valid_image(os.path.join(cat_dir, filename))][:1000]
dog_filenames = [os.path.join("Dog", filename) for filename in os.listdir(dog_dir) if is_valid_image(os.path.join(dog_dir, filename))][:1000]

# Create a DataFrame for cats and dogs
cat_data = pd.DataFrame({"filename": cat_filenames, "label": "Cat"})
dog_data = pd.DataFrame({"filename": dog_filenames, "label": "Dog"})

# Combine the data
data = pd.concat([cat_data, dog_data], ignore_index=True)

# Split the data into train, validation, and test sets
# Stratify: to maintain proportion of provided dataset during train/test/val split
# Shuffle=True to shuffle all the image files cuz input dataframe created by concatenation of two categories dataframe
train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data["label"], shuffle=True, random_state=42)
test_data, val_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["label"], random_state=42)

# Update file paths to absolute paths to give proper input while creating ImageDataGenerators
train_data["filename"] = train_data["filename"].apply(lambda x: os.path.join(base_dir, x))
val_data["filename"] = val_data["filename"].apply(lambda x: os.path.join(base_dir, x))
test_data["filename"] = test_data["filename"].apply(lambda x: os.path.join(base_dir, x))

# Set image size and batch size
size = (224, 224)  # MobileNet's default input size (also if too large size given, initially tried with image median as size,
#                    noted significant increase in model training time, especially for heavier ones like VGG16)
batch_size = 32

# Create ImageDataGenerator
# ImageDataGenerator: used for images augmentation and pre-processing
idg = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

# Create ImageDataGenerators for training, validation, and testing
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

# Load MobileNet model
ssl._create_default_https_context = ssl._create_unverified_context # used for debugging for HTTPS
mobilenet_model = tf.keras.applications.MobileNet(include_top=False, input_shape=(224, 224, 3), weights="imagenet") # pre-trained model

# Freeze layers: ensuring pre-trained weights are maintained/frozen 
for layer in mobilenet_model.layers:
    layer.trainable = False

# Added custom layers on top of MobileNet
flat = tf.keras.layers.Flatten()(mobilenet_model.output)
dense1 = tf.keras.layers.Dense(128, activation="relu")(flat) # dense layer to learn task-specific patterns
dropout1 = tf.keras.layers.Dropout(0.2)(dense1) # dropouts used for reducing overfitting 
output = tf.keras.layers.Dense(2, activation="softmax")(dropout1) # output layer customized for binary classification

# Combiining the base and custom layers
final_model = tf.keras.models.Model(inputs=mobilenet_model.input, outputs=output)

# Compile the model: configuring the learning process (kind og like compiling a program before running the exec. program)
# Optimizer adjusts the learning rate during training
# categorical_crossentropy is the loss func used in classification problems
final_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks: used to monitor and modify the training process dynamically
# Explaining the various arguments in the callbacks:-
# monitor: what metric it monitors
# patience: waits for how many epochs without improvement
# factor: by how many times it changes the lr
# verbose: how it displays the info in logs
# Note>> Verbose level: 0 => silent, 1 => progress bar, 2 => one line per epoch
learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience=2, factor=0.5, min_lr=0.00001, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

# Train the model
history = final_model.fit(
    train_idg,
    validation_data=val_idg,
    epochs=10,
    callbacks=[learning_rate_reduction, early_stopping],
    verbose=1
)

# Evaluate the model
loss, acc = final_model.evaluate(test_idg, batch_size=batch_size, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# Save the model
save_dir = "./saved_model_mobilenet/"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "mobilenet_model.h5")
final_model.save(model_path)

print(f"Model saved to: {model_path}")

# Accuracy recorded = 97+%