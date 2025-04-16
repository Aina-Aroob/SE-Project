import cv2
import json

# Load trajectory coordinates
with open('ball_trajectory.json', 'r') as f:
    data = json.load(f)
trajectory = data["trajectory"]

# Open input video
cap = cv2.VideoCapture('input_video.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(trajectory):
        break

    # Get current point and draw a blue circle
    point = trajectory[frame_idx]
    cv2.circle(frame, (point['x'], point['y']), 5, (255, 0, 0), -1)  # blue dot

    # Draw trajectory line up to current point
    for i in range(1, frame_idx + 1):
        pt1 = (trajectory[i - 1]['x'], trajectory[i - 1]['y'])
        pt2 = (trajectory[i]['x'], trajectory[i]['y'])
        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # blue line

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("✅ Output video with trajectory saved as output_video.mp4")
