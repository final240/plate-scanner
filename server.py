import cv2
import numpy as np
import re
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import easyocr

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['ko', 'en'], gpu=False)

def extract_plate_text(crop_img):
    results = reader.readtext(crop_img)
    for bbox, text, prob in results:
        cleaned = re.sub(r'[^0-9가-힣]', '', text)
        match = re.search(r'(\d{2,3}[가-힣]\d{4}|[가-힣]{2}\d{2}[가-힣]\d{4})', cleaned)
        if match:
            return match.group(0), prob
    return None, 0

@app.post("/detect")
async def detect_license_plate(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"success": False, "results": []}

    results = model(frame, verbose=False)[0]
    detected_plates = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_plate = frame[y1:y2, x1:x2]
        if cropped_plate.size == 0:
            continue

        plate_text, confidence = extract_plate_text(cropped_plate)
        if plate_text:
            detected_plates.append({
                "plate": plate_text,
                "confidence": float(confidence),
                "bbox": [x1, y1, x2 - x1, y2 - y1]
            })

    return {"success": True, "results": detected_plates}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)