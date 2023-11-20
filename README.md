# PC-Part Detector

## Overview

The PC Part Recognition Application is a project aimed at simplifying the sorting and recycling of PC components in a facility. The application utilizes the YOLO (You Only Look Once) model for real-time object detection to identify and categorize PC parts such as hard drives (HDD), solid-state drives (SSD), RAM, graphics processing units (GPU), and large HDDs (HDD_L).

## Project Goals
1. Develop an application for recognizing PC parts in real-time.
2. Create a high-quality dataset to train the YOLO model with good precision.
3. Reduce the risk of using incorrect components during the sorting and recycling of PC components.

## Getting Started

1. **Training the YOLO Model:**
   - Run `training.py` to train the YOLO model.
   - Configure dataset paths and class names in `pcparts.yaml`.

2. **Flask Web Application:**
   - Execute `main.py` to start the Flask web application.
   - Access the main page (`/`) for project information.

3. **Live Object Detection:**
   - Navigate to the detection page (`/detect`) for a live video feed with PC part detection.
   - Click "Detect" to initiate the real-time detection.

## Usage

- Access the web interface through a web browser.
- Click on "Detect" to initiate the live video feed with PC part detection.
- Detected PC parts will be highlighted with bounding boxes and labels.

### Dataset

The data is structured as follows:
   - data
       - train
           - images
           - labels
       - validate
           - images
           - labels

## System Architecture

The project consists of the following components:

1. **Training Module (`training.py`):**
   - Uses the YOLO model to train on a specified dataset.
   - Supports GPU acceleration.
   - Generates trained weights for PC part recognition.

2. **Dataset Configuration (`pcparts.yaml`):**
   - Defines paths for training and validation images.
   - Specifies class names for PC parts.

3. **Flask Web Application (`main.py`, `routes.py`):**
   - `main.py`: Entry point for starting the Flask application.
   - `routes.py`: The Flask app routes and integrates the trained YOLO model for object detection.
   - Provides routes for the main page, detection page, and video feed.


4. **Web Interface (`index.html` and `detect.html`):**
   - `index.html`: Main page with project information.
   - `detect.html`: Displays live video feed with PC part detection.
   - Uses simple HTML templates and CSS for styling.



## Dependencies

- Python 3.x
- PyTorch
- Ultralytics YOLO
- Flask
- OpenCV

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- This project was developed as part of a Machine Vision Course.
