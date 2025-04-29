import cv2
import mediapipe as mp
import numpy as np
import json

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the video
cap = cv2.VideoCapture("VideosForTesting/crick01.mp4")

paused = False
last_bbox = None
frame_id = 0
output_data = []

def bbox_to_corners(bbox):
    x_min, y_min, x_max, y_max = bbox
    return [
        [x_min, y_min, 0],
        [x_max, y_min, 0],
        [x_max, y_max, 0],
        [x_min, y_max, 0]
    ]

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        bbox_found = False

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Draw landmarks and lines
            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            connections = mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                cv2.line(frame,
                         (int(start.x * width), int(start.y * height)),
                         (int(end.x * width), int(end.y * height)),
                         (0, 255, 0), 2)

            # Get right knee and ankle
            knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            knee_point = (int(knee.x * width), int(knee.y * height))
            ankle_point = (int(ankle.x * width), int(ankle.y * height))

            leg_height = abs(knee_point[1] - ankle_point[1])
            leg_width = abs(knee_point[0] - ankle_point[0])
            proportional_factor = 1.5

            x_min = max(0, min(knee_point[0], ankle_point[0]) - int(leg_width * proportional_factor))
            y_min = max(0, min(knee_point[1], ankle_point[1]) - int(leg_height * proportional_factor))
            x_max = min(width, max(knee_point[0], ankle_point[0]) + int(leg_width * proportional_factor))
            y_max = min(height, max(knee_point[1], ankle_point[1]) + int(leg_height * proportional_factor))

            leg_region = frame[y_min:y_max, x_min:x_max]
            hsv_leg_region = cv2.cvtColor(leg_region, cv2.COLOR_BGR2HSV)

            lower_color = np.array([0, 48, 80])
            upper_color = np.array([20, 255, 255])
            mask = cv2.inRange(hsv_leg_region, lower_color, upper_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    abs_x_min = x_min + x
                    abs_y_min = y_min + y
                    abs_x_max = abs_x_min + w
                    abs_y_max = abs_y_min + h
                    last_bbox = (abs_x_min, abs_y_min, abs_x_max, abs_y_max)
                    bbox_found = True
                    cv2.rectangle(frame, (abs_x_min, abs_y_min), (abs_x_max, abs_y_max), (0, 255, 0), 2)

        # Store data (use last known bbox if detection failed)
        if last_bbox:
            corners = bbox_to_corners(last_bbox)
            output_data.append({
                "frame_id": frame_id,
                "leg": {
                    "corners": corners
                }
            })

        # Display frame
        frame_resized = cv2.resize(frame, (640, 360))
        cv2.imshow("Full Frame with Bounding Boxes and Skeleton", frame_resized)

        frame_id += 1

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:
        paused = not paused

# Save all frame data to JSON file
with open("leg_bboxes.json", "w") as f:
    json.dump(output_data, f, indent=2)

cap.release()
cv2.destroyAllWindows()
