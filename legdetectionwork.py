import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the video
cap = cv2.VideoCapture("crick01.mp4")

paused = False  # Variable to track pause state

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Draw the skeleton on the frame
            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw the connections between landmarks (skeletal lines)
            connections = mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start_landmark = landmarks[start_idx]
                end_landmark = landmarks[end_idx]

                start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
                end_point = (int(end_landmark.x * width), int(end_landmark.y * height))

                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

            # Get coordinates for right knee and ankle
            knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            knee_point = (int(knee.x * width), int(knee.y * height))
            ankle_point = (int(ankle.x * width), int(ankle.y * height))

            # Calculate the dimensions of the cropped region based on the distance between knee and ankle
            leg_height = abs(knee_point[1] - ankle_point[1])
            leg_width = abs(knee_point[0] - ankle_point[0])

            # Define a proportional factor (you can adjust this to make the crop smaller or larger)
            proportional_factor = 1.5  # Adjust the size of the cropped region

            # Calculate new coordinates for cropping
            x_min = max(0, min(knee_point[0], ankle_point[0]) - int(leg_width * proportional_factor))
            y_min = max(0, min(knee_point[1], ankle_point[1]) - int(leg_height * proportional_factor))
            x_max = min(width, max(knee_point[0], ankle_point[0]) + int(leg_width * proportional_factor))
            y_max = min(height, max(knee_point[1], ankle_point[1]) + int(leg_height * proportional_factor))

            # Convert the region of interest to HSV color space for easier color detection
            leg_region = frame[y_min:y_max, x_min:x_max]
            hsv_leg_region = cv2.cvtColor(leg_region, cv2.COLOR_BGR2HSV)

            # Define the color range for detection (example: detecting skin tone)
            lower_color = np.array([0, 48, 80])  # Lower bound of the color range (example)
            upper_color = np.array([20, 255, 255])  # Upper bound of the color range (example)

            # Create a mask for the color range
            mask = cv2.inRange(hsv_leg_region, lower_color, upper_color)
            
            # Find contours on the masked region to draw bounding boxes
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:  # Filter out small areas to avoid noise
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Draw the bounding box on the full frame (main window)
                    cv2.rectangle(frame, (x_min + x, y_min + y), (x_min + x + w, y_min + y + h), (0, 255, 0), 2)

        # Resize full frame for display
        frame_resized = cv2.resize(frame, (640, 360))
        cv2.imshow("Full Frame with Bounding Boxes and Skeleton", frame_resized)

    # Wait for key press
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # Spacebar key
        paused = not paused

cap.release()
cv2.destroyAllWindows()