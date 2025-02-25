# Shoplifting Detection with YOLOv8

## Overview
This project implements a shoplifting detection system using the YOLOv8 model. The system detects objects in video frames and identifies potential shoplifting activities in retail environments. The model is trained on a custom dataset, and detection results are shown as annotated video frames with bounding boxes around the detected objects.
## Features
#### Object Detection: 
Detects various objects like products and people in a retail environment.
#### Shoplifting Detection:
Automatically highlights suspicious activities that could indicate shoplifting.
#### Video Output: 
Annotated video frames are saved with bounding boxes and class labels for detected objects.
#### Real-Time Detection:
Performs frame-by-frame detection using YOLOv8, displaying results in real-time.
## Installation
To run this project, you need Python 3.7 or later and the necessary libraries. You can set up the environment using the following steps.

1. Clone the Repository (if using a GitHub repository)

git clone https://github.com/sujigunasekar/shoplifting-detection.git
cd shoplifting-detection
2. Install Required Dependencies
Use pip to install the necessary packages. You can create a virtual environment to manage dependencies:
```
pip install -r requirements.txt
```
If you don’t have the requirements.txt, you can install the core dependencies manually:
```
pip install ultralytics opencv-python
```
## Dataset
This project uses a custom dataset for shoplifting detection. You can use your own video footage and annotations for training.

The dataset includes:

#### Images: 
Frames extracted from video footage.
#### Labels:
YOLO-compatible annotation files containing bounding box information for each object.
### Training the Model
If you want to train the model with your own dataset, follow these steps:

### Prepare your dataset in the YOLO format:

Organize images in the /train/images/ directory.

Create a corresponding label file in /train/labels/ for each image, containing class information in YOLO format.

Create a data.yaml file to configure the dataset:

### Train the model:
```
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model (nano version)
model.train(data="data.yaml", epochs=20, batch=8, imgsz=640)  # Customize as needed
Running Inference on Video
Once the model is trained, use it for inference on a video to detect shoplifting behavior.

from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO("/content/runs/detect/train/weights/best.pt")  # Use your trained model's path
```
### Path to the input video
```
video_path = "/content/shoplifting_dataset_video.mp4"

cap = cv2.VideoCapture(video_path)
```
### Get video properties
```
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
```
### Set up output video writer
```
output_path = "/content/shoplifting_detection_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on the current frame
    results = model(frame)

    # Annotate frame with bounding boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results[0].names[int(box.cls)]
        confidence = box.conf[0].item()

        if confidence > 0.5:  # Confidence threshold
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the annotated frame to output video
    out.write(frame)
```
### Release resources
```
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Output video saved at: {output_path}")
```

### Model Performance Summary:
The YOLOv8 model trained for shoplifting detection demonstrates promising results with the following performance metrics:

Precision: 73.2%

This indicates that 73.2% of the predicted shoplifting events were correct, minimizing false positive detections.

Recall: 76.9%

The model successfully identified 76.9% of all the actual shoplifting instances, minimizing false negatives.

F1-Score: 75.0%

The F1-score, which balances precision and recall, is approximately 75%, reflecting the model's overall effectiveness in detecting shoplifting events.


These results suggest that the model is well-suited for identifying shoplifting in video footage, with a good balance between false positives and false negatives. Further refinement and tuning may improve performance, especially by adjusting the confidence threshold and experimenting with more diverse datasets.
## Output
![image](https://github.com/user-attachments/assets/9fb86b77-a972-441f-be50-f010de2f3761)

### Detection Video: 
The output video will show the annotated frames, with bounding boxes around detected objects, including class labels and confidence scores.




