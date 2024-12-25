Parking Lot Monitoring System

Overview

This project implements a Parking Lot Monitoring System using computer vision (CV) and machine learning (ML) techniques. The system detects vehicles in parking spaces, identifies their statuses (empty or occupied), and outputs the total count of available parking spots. The solution supports both image and video inputs. Additionally, the project includes a comparison of various machine learning models and neural network architectures, analyzing their performances and identifying the best approach for parking lot monitoring.

Features

Vehicle Detection: Detects vehicles in parking spots using bounding boxes.
Status Classification: Identifies parking spots as empty or occupied based on whether vehicle detected.
Total Count: Outputs the total number of available parking spots in real-time.
Input Support: Works with both image and video inputs.
*Model Comparisons: Evaluates multiple models and compares their performances.*

Limitations:
Customized to one parking lot arrangement

DataSet: https://drive.google.com/drive/folders/1CjEFWihRqTLNUnYRwHXxGAVwSXF2k8QC
mask image: using CVAT/labelmg (mark bboxes and get binary image)

Model Comparisons

General: All the models gave really high accuracy scores probabl due to the clear distinction between the two catgories in the classification (MLP gave 100% accuracy score). Hence the best choice would be to implement a simple and lightweight model rather than those that could handle complex classification cases as that does not seem to be the problem here.

Logistic Regression: gave 99.6 % accuracy score, simple to implement and easy to understand the hyperparametrs 
SVC: gave 99.92 % accuracy score, very effective for the dataset used, best combo of C and gamma chosen using gridsearch
MLP: gave 100 % accuracy, multiple hidden layers to handle non-linearities
CNN: gave 99.4 % accuracy, architecture used: 2 Conv, maxpooling, dense layers

Optmizations Used:
Resized images to uniform dimensions (15x15 for ML model)
Mask applied to extract parking spot regions
Differences in mean pixel values used for motion-based detection (only those spots checked)
Hyperparameter tuning using GridSearchCV for ML models
Experimented with various CNN architectures and activation functions
Skipped frames (step = 30) to optimize real-time performance for videos