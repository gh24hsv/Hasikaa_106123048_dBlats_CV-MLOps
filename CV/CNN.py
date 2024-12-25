import os
import numpy as np #type: ignore
from skimage.io import imread #type: ignore
from skimage.transform import resize #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import accuracy_score #type: ignore

# Input directory and categories
input_dir = '/Users/hsv/Desktop/DB Ind/clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []

# Load and preprocess the dataset
for cat_idx, cat in enumerate(categories):
    files = os.listdir(os.path.join(input_dir, cat))
    for file in files:
        img_path = os.path.join(input_dir, cat, file)
        img = imread(img_path)
        img = resize(img, (15, 15))  # Resize image to 15x15
        data.append(img)
        labels.append(cat_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# Normalize the data
data = data / 255.0

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=len(categories))

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(15, 15, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(categories), activation='softmax')  # For binary classification with one-hot labels
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Predictions and accuracy calculation
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
final_accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f'Final Accuracy: {final_accuracy * 100:.2f}%')

# Save the model
model.save('./parking_lot_cnn_model.h5')
