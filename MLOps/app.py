import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Styling using CSS
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        color: #333333;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Model selection feature
model_options = {
    "MobileNet": "./saved_model_mobilenet/mobilenet_model.h5",
    "EfficientNet": "./saved_model_efficientnet/efficientnet_model.h5",
}

st.sidebar.title("Model Selection")
selected_model_name = st.sidebar.selectbox("Select a model", list(model_options.keys()))
model_path = model_options[selected_model_name]

# Loading the selected model
st.sidebar.write(f"Loading model: {selected_model_name}")
model = tf.keras.models.load_model(model_path)

# Defining labels
labels = ["Cat", "Dog"]

# Function to pre-process the input image/frame
def preprocess_image(image, size=(224, 224)):
    image = cv2.resize(image, size)  # Resize to the input size of the model
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    if selected_model_name == "MobileNet":
        image = tf.keras.applications.mobilenet.preprocess_input(image)
    elif selected_model_name == "EfficientNet":
        image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image

# Function to predict whether thr input image has cat or dog
def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    label = labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return label, confidence

st.title("🐱 Cat vs Dog Classifier 🐶")

# Choosing input type feature
option = st.sidebar.radio("Choose input type", ["Image", "Webcam", "Video"])

if option == "Image":
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Load and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_np = np.array(image)

        if st.button("Predict"):
            label, confidence = predict(image_np)
            st.write(f"**Prediction:** {label} (Confidence: {confidence*100:.2f}%)")

elif option == "Webcam":
    st.header("Webcam Live Feed")
    st.write("Click 'Start' to open the webcam feed and 'Stop' to exit.")
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    if start_button:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to access webcam")
                break

            # Flip and preprocess frame
            frame = cv2.flip(frame, 1)
            prediction_label, prediction_confidence = predict(frame)

            # Display prediction on the frame itself
            cv2.putText(frame, f"{prediction_label} ({prediction_confidence*100:.2f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display frame in Streamlit
            stframe.image(frame, channels="BGR")

            # Stop button to exit webcam
            if stop_button:
                break

        cap.release()

elif option == "Video":
    st.header("Upload a Video")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            temp_video.write(uploaded_video.read())
            video_path = temp_video.name

        if st.button("Process Video"):
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("Could not open video. Ensure the format is supported.")
            else:
                stframe = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    prediction_label, prediction_confidence = predict(frame)

                    cv2.putText(frame, f"{prediction_label} ({prediction_confidence*100:.2f}%)", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    stframe.image(frame, channels="BGR")

                cap.release()

st.write("Thank you for using the 🐾 Cat vs Dog Classifier!")
