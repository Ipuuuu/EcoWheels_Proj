import cv2
import time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO

# ================= CONFIG =================
DROIDCAM_URL = "http://192.168.5.131:4747/video"  # change this
MODEL_PATH = "/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/runs/detect/train2/weights/best.pt"
# ==========================================

app = FastAPI()
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(DROIDCAM_URL)

if not cap.isOpened():
    raise RuntimeError("Could not open DroidCam stream")


def generate_frames():
    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        _, buffer = cv2.imencode(".jpg", annotated)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

        time.sleep(max(0, 0.1 - (time.time() - start)))  # ~10 FPS



@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
