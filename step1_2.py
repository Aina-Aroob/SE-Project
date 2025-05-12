import cv2
import time

# Path to your cricket video
video_path = 'input1.mp4'  

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Get original frames per second (FPS) of the video
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate new delay for 0.75x speed
delay = int((1000 / original_fps) / 0.75)  # in milliseconds

while True:
    ret, frame = cap.read()

    if not ret:
        print("Finished reading the video.")
        break

    # Pre-processing steps:

    # 1. Resize the frame
    frame_resized = cv2.resize(frame, (640, 360))  # Resize to 640x360

    # 2. Apply Gaussian blur to reduce noise
    frame_blurred = cv2.GaussianBlur(frame_resized, (5, 5), 0)

    # 3. Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)

    # Show the grayscale resized frame
    cv2.imshow('Grayscale Slow Motion Video', frame_gray)

    # Wait with adjusted delay
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
