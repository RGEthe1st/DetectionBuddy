# Real-Time Face & Emotion Analytics

 Real-time face detection, tracking, and emotion analysis system developed in Python. It leverages advanced deep-learning and computer vision libraries to deliver an interactive video feed where detected faces are dynamically numbered (e.g., if only one face is present, it appears as **ID 1**; if another enters, the faces are renumbered accordingly). The system utilizes MTCNN for accurate face detection, OpenCV's CSRT tracker for reliable tracking, and DeepFace for real-time emotion recognition.

## Features

- **Accurate Face Detection:**  
  Utilizes MTCNN to detect faces with high accuracy and filters out false positives by applying confidence and size thresholds.

- **Reliable Face Tracking:**  
  Employs OpenCV's CSRT tracker to maintain face locations between detection intervals, minimizing drift and ensuring smooth tracking.

- **Real-Time Emotion Analysis:**  
  Integrates DeepFace (with the MTCNN backend) to analyze facial expressions and determine the dominant emotion of each detected face.

- **Dynamic Display IDs:**  
  Reassigns display IDs dynamically based on the number of faces present. For instance, if only one person is detected, they will be labeled as **ID 1**. As faces appear and disappear, the numbering updates automatically.

- **Performance Optimizations:**  
  Resizes video frames and selectively performs heavy computations (e.g., emotion analysis every few frames) to maintain a higher FPS.

## Requirements

- **Python 3.x**
- **Libraries:**
  - [opencv-contrib-python](https://pypi.org/project/opencv-contrib-python/)
  - [deepface](https://pypi.org/project/deepface/)
  - [numpy](https://pypi.org/project/numpy/)
  - [mtcnn](https://pypi.org/project/mtcnn/)
