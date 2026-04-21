import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo26m.pt")
cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.20)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Webcam detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
