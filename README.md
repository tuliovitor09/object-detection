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
```
## Installation

### Create virtual environment

```
python -m venv venv
```

### Activate virtual environment

#### Windows (CMD)

```
venv\Scripts\activate
```

#### PowerShell

```
.\venv\Scripts\Activate.ps1
```

---

## Install dependencies

```
pip install fastapi uvicorn ultralytics opencv-python python-multipart
```

Or using requirements file:

```
pip install -r requirements.txt
```

---

## Running the API

Start the FastAPI server:

```
uvicorn app:app --reload
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### POST `/detect`

Upload an image and receive detected objects.

### Request

`multipart/form-data`

Field name:

```
file
```

---

## Example Response

```
{
  "filename": "image.jpg",
  "detections": [
    {
      "object": "person",
      "confidence": 0.94
    },
    {
      "object": "cell phone",
      "confidence": 0.88
    }
  ]
}
```

---

## Running Real-Time Webcam Detection

```
python webcam_detection.py
```

The webcam will open and start detecting objects in real time.

Press:

```
Q
```

to exit.

---

## Example Features in Webcam Mode

* live bounding boxes
* object label
* confidence score
* FPS counter

Example:

```
person 0.94
FPS: 22.37
```

---

## Learning Concepts Covered

This project was created with a strong focus on learning the concepts behind the implementation.

### Artificial Intelligence

* pre-trained model usage
* inference pipeline
* confidence thresholds
* object detection

### Computer Vision

* frame processing
* image coordinates
* bounding boxes (`x1, y1, x2, y2`)
* real-time rendering

### Backend / API

* asynchronous endpoints
* file upload handling
* JSON response structure
* FastAPI routing

### Performance

* frame processing time
* FPS calculation
* real-time inference optimization

---

## Future Improvements

Planned next steps for this project:

* object counting
* object tracking
* video file detection
* model fine-tuning
* cloud deployment
* Docker containerization
* streaming via WebSocket

---

## Purpose of This Project

This repository is part of my transition and continuous learning journey into:

* **Artificial Intelligence**
* **Machine Learning Engineering**
* **Python backend development**
* **Computer Vision**

The goal is to build real projects while deeply understanding the concepts behind them.

---

## Author

**Tulio Vitor Sousa**

Software Engineer | Transitioning into AI, Python and Machine Learning Engineering
