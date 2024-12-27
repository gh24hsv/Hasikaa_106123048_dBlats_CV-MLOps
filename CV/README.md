# 🚗 Parking Lot Monitoring System 🅿️

This project implements a **Parking Lot Monitoring System** using computer vision (CV) and machine learning (ML) techniques. The system detects vehicles in parking spaces, classifies their statuses (empty or occupied), and calculates the total count of available parking spots in real-time. It supports both image and video inputs.

Additionally, the project includes a comparative analysis of various machine learning models and neural network architectures, providing insights into their performance and suitability for parking lot monitoring.

---

## **Features**

- **Vehicle Detection:** Detects vehicles in parking spots using bounding boxes.
- **Status Classification:** Classifies parking spots as empty or occupied based on vehicle detection.
- **Total Count:** Outputs the total number of available parking spots in real-time.
- **Input Support:** Works seamlessly with both image and video inputs.
- **Model Comparisons:** Evaluates multiple machine learning models and neural network architectures to identify the most efficient solution.

---

## **Limitations**
- Customized for one specific parking lot arrangement.

---

## **Dataset**

- The dataset can be downloaded [here](https://drive.google.com/drive/folders/1CjEFWihRqTLNUnYRwHXxGAVwSXF2k8QC).
- **Mask Image:** Created using tools like CVAT or LabelImg to mark bounding boxes and generate a binary mask image for parking spots.

---

## **Model Comparisons**

The project explores multiple models and their performances. Here are the results:

| **Model**            | **Accuracy** | **Remarks**                                                                 |
|-----------------------|--------------|------------------------------------------------------------------------------|
| Logistic Regression   | 99.6%       | Simple to implement, easy to understand hyperparameters.                    |
| Support Vector Classifier (SVC) | 99.92%      | Very effective for the dataset used; best combination of `C` and `gamma` achieved using GridSearchCV. |
| Multi-Layer Perceptron (MLP)    | 100%        | Handles non-linearities with multiple hidden layers; may be overkill for this task. |
| Convolutional Neural Network (CNN) | 99.4%       | Architecture: 2 Convolutional layers, MaxPooling, Dense layers; good performance. |

**General Conclusion:**
- All models achieved high accuracy due to the clear distinction between the two categories (empty and occupied).
- A lightweight model (e.g., Logistic Regression or SVC) is recommended for deployment, as handling complex classification cases is unnecessary for this dataset.

---

## **Optimizations Used**

1. **Preprocessing:**
   - Resized images to uniform dimensions (15x15 for ML models).
   - Applied a mask to extract parking spot regions from the input.

2. **Motion-Based Detection:**
   - Used differences in mean pixel values to detect motion in parking spots.
   - Checked only affected spots, improving efficiency.

3. **Hyperparameter Tuning:**
   - Utilized GridSearchCV for optimizing hyperparameters in ML models (e.g., `C` and `gamma` for SVC).

4. **CNN Architectures:**
   - Experimented with various architectures and activation functions to maximize performance.

5. **Real-Time Optimization:**
   - Skipped frames (step = 30) for video inputs to enhance real-time performance.

---