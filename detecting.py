import cv2
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("C:/Users/nikla/ikt213g23h/assignments/solutions/PCPartDetector/runs/detect/train8/weights/best.pt").to(device)

    # Detect the image

    # Load the image
    image = cv2.imread("C:/Users/nikla/ikt213g23h/assignments/solutions/PCPartDetector/TestImg/ram.jpg")

    # Camera:
    results = model.predict(source="1", show=True)

    # Image:
    # results = model.predict(image, show=True, save=True, save_dir="C:/Users/nikla/ikt213g23h/assignments/solutions/PCPartDetector/TestImg/")

    # Wait for keyboard press
    cv2.waitKey(0)

    # Close the image window
    cv2.destroyAllWindows()