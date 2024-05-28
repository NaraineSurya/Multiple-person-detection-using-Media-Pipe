import torch
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

display_size = (1280, 720)

# Define the coordinates of the rectangle region (x1, y1) - top-left, (x2, y2) - bottom-right
region_coords = [(300, 0), (670, 470)]

# Define the coordinates of the staircase rail region (x1, y1) - top-left, (x2, y2) - bottom-right
staircase_coords = [(515, 0), (535, 0), (575, 270), (545, 270)]

def is_point_in_polygon(point, polygon):
    result = cv2.pointPolygonTest(np.array(polygon), point, False)
    return result >= 0

# Load the YOLOv5 model
model_path = r"C:\Users\naray\Downloads\yolov5s.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Path to the video file
video_path = r"C:\Users\naray\OneDrive\Pictures\demo2.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_path = 'output_detection_yolo_pose.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (display_size[0], display_size[1]))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        # Resize the frame to fit the display size
        frame = cv2.resize(frame, display_size)

        # Get the region of interest for detection
        x1, y1 = region_coords[0]
        x2, y2 = region_coords[1]
        roi = frame[y1:y2, x1:x2]

        # Convert the ROI to RGB (YOLOv5 expects RGB input)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Perform inference on the ROI
        results = model(roi_rgb)

        # Extract information from the results
        results_df = results.pandas().xyxy[0]

        # Draw the range rectangle on the frame
        region_color = (0, 255, 255)  # Yellow by default
        cv2.rectangle(frame, (x1, y1), (x2, y2), region_color, 2)

        # Draw the staircase rail polygon
        rail_color = (0, 0, 255)  # Red by default
        cv2.polylines(frame, [np.array(staircase_coords)], isClosed=True, color=rail_color, thickness=2)

        # Perform pose estimation for each detected person
        for i in range(len(results_df)):
            x1_det, y1_det, x2_det, y2_det, conf, cls = results_df.iloc[i, :6]
            if model.names[int(cls)] == 'person':  # Check if the detected class is 'person'
                # Adjust detection coordinates to match the original frame
                x1_det += x1
                y1_det += y1
                x2_det += x1
                y2_det += y1
                
                if x1_det >= x1 and y1_det >= y1 and x2_det <= x2 and y2_det <= y2:
                    
                    person_roi = frame[int(y1_det):int(y2_det), int(x1_det):int(x2_det)]

                    # Convert the person ROI to RGB for processing with MediaPipe
                    person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

                    # Perform pose estimation on the person ROI
                    pose_results = pose.process(person_roi_rgb)
                    
                    hand_landmark_color = (0, 0, 255)  # Red by default

                    if pose_results.pose_landmarks:
                        hand_touching = False
                        # Calculate the offset of the ROI in the original image
                        offset_x, offset_y = int(x1_det), int(y1_det)

                        # Get wrist coordinates relative to the original image
                        left_wrist_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                        left_wrist_x = int(left_wrist_landmark.x * (x2_det - x1_det) + offset_x)
                        left_wrist_y = int(left_wrist_landmark.y * (y2_det - y1_det) + offset_y)

                        right_wrist_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                        right_wrist_x = int(right_wrist_landmark.x * (x2_det - x1_det) + offset_x)
                        right_wrist_y = int(right_wrist_landmark.y * (y2_det - y1_det) + offset_y)

                        # Check if any wrist landmark is touching the staircase rail
                        if is_point_in_polygon((right_wrist_x, right_wrist_y), staircase_coords) or is_point_in_polygon((left_wrist_x, left_wrist_y), staircase_coords):
                            hand_touching = True
                            hand_landmark_color = (0, 255, 0)  # Green if touching
                            rail_color = (0, 255, 0)  # Change rail color to green

                        # Draw the updated staircase rail polygon
                        cv2.polylines(frame, [np.array(staircase_coords)], isClosed=True, color=rail_color, thickness=2)

                        # Draw the pose estimation with the updated landmark color
                        mp_drawing.draw_landmarks(
                            person_roi,
                            pose_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=hand_landmark_color, thickness=2, circle_radius=2))

                # Draw bounding box around the detected person
                cv2.rectangle(frame, (int(x1_det), int(y1_det)), (int(x2_det), int(y2_det)), hand_landmark_color, 2)
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.putText(frame, label, (int(x1_det), int(y1_det) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video file
        out.write(frame)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('YOLOv5 + MediaPipe Pose', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
