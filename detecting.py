import cv2
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("./runs/detect/train8/weights/best.pt").to(device)


    # Load the image
    image = cv2.imread("./TestImg/ram.jpg")

    # Image:
    results = model.predict(image, show=True, save=True)

    # Camera:
    results = model.predict(source="1", show=True)



    cv2.waitKey(0)

    cv2.destroyAllWindows()