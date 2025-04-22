import cv2
import numpy as np
import json
from filterpy.kalman import KalmanFilter
from collections import deque

# ---- Kalman Filter Setup ----
def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0, 0, 0, 0])
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 1000.
    kf.R = np.array([[10, 0], [0, 10]])
    kf.Q = np.eye(4)
    return kf

# ---- Load Input Video ----
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

# ---- Video Properties for Output ----
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('detections_overlaid.mp4', fourcc, fps, (width, height))

# ---- Tracking State ----
kalman = create_kalman_filter()
track_history = deque(maxlen=1000)
frame_number = 0
output_data = []

# ---- Frame-by-Frame Processing ----
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=15, minRadius=3, maxRadius=15)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        detected = True
        ball_x, ball_y = circles[0][0], circles[0][1]
        kalman.update(np.array([ball_x, ball_y]))
        cv2.circle(frame, (ball_x, ball_y), 10, (0, 0, 255), 2)
    else:
        detected = False
        kalman.predict()

    pred_x, pred_y = int(kalman.x[0]), int(kalman.x[1])
    track_history.append((pred_x, pred_y))

    for i in range(1, len(track_history)):
        cv2.line(frame, track_history[i - 1], track_history[i], (0, 255, 255), 2)

    cv2.circle(frame, (pred_x, pred_y), 5, (255, 0, 0), -1)

    output_data.append({
        'frame': frame_number,
        'predicted_position': [pred_x, pred_y],
        'detected': detected
    })

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

with open("tracked_output.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("Video saved as detections_overlaid.mp4")