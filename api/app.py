from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

model = YOLO("yolo26l.pt")


# exemplo selecionando imagem no computador
@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):

    file_path = f"images/{file.filename}"
    os.makedirs("images", exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(file_path, conf=0.15, imgsz=1280)

    detections = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({"object": model.names[cls], "confidence": round(conf, 2)})

    os.makedirs("outputs", exist_ok=True)
    results[0].save("outputs/result.jpg")
    results[0].show()

    return {"filename": file.filename, "detections": detections}
