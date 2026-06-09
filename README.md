# MS-ASL Sign Language Recognition System

## Overview

This project is a Sign Language Recognition System built using the Microsoft Sign Language (MS-ASL) dataset. The system detects hand landmarks from sign language videos using MediaPipe, extracts meaningful features, and predicts the corresponding sign using a machine learning model. A fuzzy rule-based interpretation layer is included to provide explainable predictions.

## Features

* Video-based sign language recognition
* Hand landmark extraction using MediaPipe
* Feature engineering from hand gestures
* Random Forest based classification
* Fuzzy rule-based confidence interpretation
* Flask web application for easy interaction
* Support for uploading and analyzing sign language videos

## Dataset

The project uses the Microsoft Sign Language Dataset (MS-ASL), a large-scale benchmark dataset containing American Sign Language videos.

Dataset Information:

* 1000 sign classes
* Training, validation, and test splits
* Real-world signer variations
* Large vocabulary sign recognition benchmark

Reference:
Vaezi Joze, H. and Koller, O. (2019). MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language.

## Tech Stack

* Python
* Flask
* OpenCV
* MediaPipe
* NumPy
* Scikit-Learn
* Joblib


## Working

1. Upload a sign language video.
2. MediaPipe extracts hand landmarks.
3. Features are generated from landmark coordinates.
4. Random Forest predicts the sign class.
5. Fuzzy rules generate confidence explanations.
6. Results are displayed through the Flask interface.

## Future Improvements

* Deep Learning based recognition
* Real-time webcam prediction
* Sentence-level sign recognition
* Model deployment on cloud platforms
* Improved gesture tracking


https://github.com/AashrithaSura
