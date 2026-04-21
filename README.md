# Real-Time Object Detection API & Webcam Vision with YOLO

This project was built as a hands-on study to explore **Computer Vision**, **pre-trained AI models**, and **real-time object detection** using Python.

The main goal was to learn how to integrate a pre-trained object detection model into both:

- a **REST API** using FastAPI
- a **real-time webcam application** using OpenCV

The project uses a YOLO pre-trained model from Ultralytics for inference.

---

## Tech Stack

- Python
- FastAPI
- OpenCV
- Ultralytics YOLO
- Uvicorn

---

## Features

### Image Detection API
- Upload image files via HTTP
- Detect objects using YOLO
- Return detected objects as JSON
- Confidence score for each object

### Real-Time Webcam Detection
- Webcam live stream processing
- Bounding boxes drawn manually
- Object label + confidence
- Real-time FPS display

---

## Project Structure

```text
object-detection-project/
│
├── api/
│   └── app.py
│
├── webcam/
│   └── realtime_detection.py
│
├── images/
├── requirements.txt
└── README.md
