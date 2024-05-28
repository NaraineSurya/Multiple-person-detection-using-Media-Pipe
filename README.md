# YOLOv5 and MediaPipe Pose Estimation for Multi-Person Staircase Rail Detection

## About

This project demonstrates a real-time system that utilizes YOLOv5 for person detection and MediaPipe Pose for simultaneous pose estimation of multiple people to monitor fall hazards on staircases. The system identifies people within a designated region of interest (ROI) and then estimates their poses to determine if their hands are touching the staircase rail. This information can be valuable in preventing falls, especially for those who require assistance while using stairs.

## Sample Video

Here's a sample video demonstrating the system: 
<video width="640" height="480" controls>
  <source src="vid/sample_vid.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Technologies

- **YOLOv5:** A powerful object detection model pre-trained on the COCO dataset for various object classes, including people.
- **MediaPipe Pose:** A machine learning framework that tracks human body keypoints in real-time, enabling simultaneous pose estimation for multiple people.
- **OpenCV (cv2):** A popular library for real-time computer vision tasks and image processing.

## Setup

1. **Install Dependencies:** Ensure you have Python and the following libraries installed:
   - torch
   - mediapipe
   - opencv-python
   - ultralytics (for YOLOv5)

2. **Download YOLOv5 Model:** Download the pre-trained YOLOv5 model weights (e.g., 'yolov5s.pt') and place it in the specified path within the code (`model_path`).

3. **Define Video Path:** Update the `video_path` variable in the code to point to the video file you want to process.

4. **Run the Script:** Execute the Python script to start the real-time video analysis.

**Note:**

- This is a basic implementation and can be further enhanced with fall detection algorithms based on pose analysis.
- The colors used for visualization (bounding boxes, rails, landmarks) can be customized within the code.
