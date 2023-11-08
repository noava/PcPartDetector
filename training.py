import torch
from ultralytics import YOLO

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO("yolov8n.yaml").to(device)

    results = model.train(data="pcparts.yaml", epochs=300, batch=32, resume=True)
