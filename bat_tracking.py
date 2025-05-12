import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def detect_bat(frame, region, wrist_center):
    """
    Improved bat detection using wrist position, motion, and color filtering.
    """
    x, y, w, h = region
    roi = frame[y:h, x:w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Bat's color filtering (assumes bat is darker than gloves and pads)
    lower_bat = np.array([0, 0, 50])   # Lower bound for bat (dark colors)
    upper_bat = np.array([179, 255, 180])  # Upper bound for bat (lighter colors)
    bat_mask = cv2.inRange(hsv, lower_bat, upper_bat)

    # Mask gloves and pads (assuming white color for gloves/pads)
    lower_white = np.array([0, 0, 200])  # White detection
    upper_white = np.array([179, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    bat_mask = cv2.bitwise_and(bat_mask, ~white_mask)  # Remove white gloves/pads

    contours, _ = cv2.findContours(bat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_bat = None
    min_distance = float('inf')

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            aspect_ratio = h_c / float(w_c)
            
            if 2 < aspect_ratio < 10:
                # Center of contour in full frame
                cx = x + x_c + w_c // 2
                cy = y + y_c + h_c // 2

                # Distance to wrist
                dist = np.sqrt((cx - wrist_center[0])**2 + (cy - wrist_center[1])**2)

                if dist < min_distance and dist < 250:  # Avoid far away bats (e.g. wicketkeeper)
                    min_distance = dist
                    best_bat = (x_c, y_c, w_c, h_c, contour)

    if best_bat:
        x_c, y_c, w_c, h_c, contour = best_bat
        cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(roi, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 255, 0), 2)


cap = cv2.VideoCapture('crick01.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create a window
cv2.namedWindow('Bat Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break
        
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    h, w, _ = frame.shape
    
    if results.pose_landmarks:
        # Get wrist coordinates
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        lx, ly = int(left_wrist.x * w), int(left_wrist.y * h)
        rx, ry = int(right_wrist.x * w), int(right_wrist.y * h)
        
        # Draw circles on wrists
        cv2.circle(frame, (lx, ly), 8, (255, 0, 0), -1)
        cv2.circle(frame, (rx, ry), 8, (0, 0, 255), -1)
        
        # Define bat detection region based on wrist positions
        top_y = min(ly, ry)
        bot_y = min(h-1, int(max(ly, ry) + 0.4 * h))
        left_x = min(lx, rx) - 30
        right_x = max(lx, rx) + 30
        left_x = max(0, left_x)
        right_x = min(w-1, right_x)
        
        # Draw rectangle for bat detection region
        cv2.rectangle(frame, (left_x, top_y), (right_x, bot_y), (0, 255, 255), 2)
        
        # Detect bat in the region
        bat_region = (left_x, top_y, right_x, bot_y)
        detect_bat(frame, bat_region, (lx, ly))  # Use left wrist as reference

    # Display the frame
    cv2.imshow('Bat Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
