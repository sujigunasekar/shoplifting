# Shoplifting Detection with YOLOv8

## Overview
This project implements a shoplifting detection system using the YOLOv8 model. The system detects objects in video frames and identifies potential shoplifting activities in retail environments. The model is trained on a custom dataset, and detection results are shown as annotated video frames with bounding boxes around the detected objects.
## Features
### Object Detection: 
Detects various objects like products and people in a retail environment.
### Shoplifting Detection: Automatically highlights suspicious activities that could indicate shoplifting.
### Video Output: Annotated video frames are saved with bounding boxes and class labels for detected objects.
### Real-Time Detection: Performs frame-by-frame detection using YOLOv8, displaying results in real-time.
## Installation
To run this project, you need Python 3.7 or later and the necessary libraries. You can set up the environment using the following steps.

1. Clone the Repository (if using a GitHub repository)
bash
Copy
Edit
git clone https://github.com/yourusername/shoplifting-detection.git
cd shoplifting-detection
2. Install Required Dependencies
Use pip to install the necessary packages. You can create a virtual environment to manage dependencies:

bash
Copy
Edit
pip install -r requirements.txt
If you don’t have the requirements.txt, you can install the core dependencies manually:

bash
Copy
Edit
pip install ultralytics opencv-python
## Dataset
This project uses a custom dataset for shoplifting detection. You can use your own video footage and annotations for training.

The dataset includes:

Images: Frames extracted from video footage.
Labels: YOLO-compatible annotation files containing bounding box information for each object.
Training the Model
If you want to train the model with your own dataset, follow these steps:

## Prepare your dataset in the YOLO format:

Organize images in the /train/images/ directory.
Create a corresponding label file in /train/labels/ for each image, containing class information in YOLO format.
Create a data.yaml file to configure the dataset:

yaml
Copy
Edit
train: /path/to/train/images
val: /path/to/val/images
nc: <number_of_classes>
names: ['class1', 'class2', ...]  # Replace with your class names
Train the model:

python
Copy
Edit
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model (nano version)
model.train(data="data.yaml", epochs=20, batch=8, imgsz=640)  # Customize as needed
Running Inference on Video
Once the model is trained, use it for inference on a video to detect shoplifting behavior.

from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO("/content/runs/detect/train/weights/best.pt")  # Use your trained model's path

# Path to the input video
video_path = "/content/shoplifting_dataset_video.mp4"

cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up output video writer
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

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Output video saved at: {output_path}")
## Output
### Detection Video: The output video will show the annotated frames, with bounding boxes around detected objects, including class labels and confidence scores.
### Accuracy and Precision: The accuracy (and possibly other metrics) is logged during training and can be used to evaluate model performance.
Troubleshooting

