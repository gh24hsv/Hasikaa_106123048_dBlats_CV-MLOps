# 🐱 Cat vs Dog Classifier 🐶

This project is a binary image classification task to distinguish between images of cats and dogs. It incorporates a pre-trained mobileNet and efficientNet arch. and a user-friendly **Streamlit** app for interactive predictions.

## Features

### **1. Streamlit App**
The project includes an interactive Streamlit app for:
- **Image Classification:**
  - Upload an image of a cat or dog and get predictions with confidence scores.
- **Webcam Integration:**
  - Real-time classification using webcam live feed.
- **Video Classification:**
  - Upload a video file to classify frames as "Cat" or "Dog."
- **Model Selection:**
  - Choose between different pre-trained models (e.g., MobileNet, EfficientNet).

### **2. ML Model**
- **Transfer Learning:**
  - MobileNet and EfficientNet pre-trained on ImageNet for feature extraction.
- **Custom Layers:**
  - Adapted the models with additional layers for binary classification.
- **Callbacks:**
  - ReduceLROnPlateau and EarlyStopping to optimize the training process.

### **3. Input Types**
- **Image Upload:** Classify a single image as "Cat" or "Dog."
- **Webcam Feed:** Real-time classification using live video.
- **Video File:** Process and classify video files frame-by-frame.

### **4. Dataset**
- **Dataset:** [Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)
- A subset of the dataset is used for training and testing for faster development.

## Prerequisites

- Python 3.7+
- Required Python Libraries:
  ```bash
  pip install streamlit tensorflow numpy opencv-python pillow
