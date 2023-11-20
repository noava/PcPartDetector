import cv2
import torch
import math
from flask import render_template, request, Response
from app import app
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("C:/Users/adria/ikt213g23h/assignments/solutions/PCPartDetector/train106/weights/best.pt").to(device)
classNames = ["hdd", "ssd", "ram", "gpu", "hdd_L"]


def detect_objects():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                cv2.putText(img, classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
