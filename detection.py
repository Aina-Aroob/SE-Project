import cv2
import numpy as np
import json
import csv # Import the csv module

# Define the lower and upper boundaries for the red color in HSV space
lower_red1 = np.array([0, 100, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 50])
upper_red2 = np.array([180, 255, 255])

# --- Input/Output Files ---
input_video_path = "Video3.mp4"
output_video_path = 'detections_overlaid.mp4'
output_json_path = 'ball_detections_all_frames.json' # Changed filename for clarity
output_csv_path = 'tracked_positions.csv'
# --------------------------

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {input_video_path}")
    exit()

# Video properties for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Kalman filter setup
kalman = None
kalman_initialized = False
tracked_path = []

# --- Data Storage for Output ---
json_output_data = [] # For JSON file (ALL frames)
csv_output_data = []  # For CSV file (tracked position every frame)
# -----------------------------

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or error reading frame.")
        break

    # Preprocess frame for detection
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # --- Detection Logic ---
    detected = False
    detection_center_x, detection_center_y = -1, -1 # Store detected center for Kalman
    detection_info_for_json = None # Store the specific detection details for JSON this frame

    # Try Hough Circles first
    mask_gray = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(mask_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(height/4), # Adjusted minDist
                               param1=100, param2=15, minRadius=5, maxRadius=60) # Adjusted radius

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if len(circles) > 0:
            largest_circle = max(circles, key=lambda c: c[2]) # Get largest circle
            x, y, r = largest_circle

            cv2.circle(frame, (x, y), r, (255, 0, 0), 2) # Draw detection (Blue)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), 3)
            detected = True
            detection_center_x, detection_center_y = x, y

            # Prepare detection info for JSON (without "type")
            detection_info_for_json = {
                "center": [int(x), int(y)],
                "radius": float(r)
            }

    # If no circle detected, try contours/ellipses
    if not detected:
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    cv2.ellipse(frame, ellipse, (0, 255, 0), 2) # Draw detection (Green)
                    detected = True
                    (x, y), (MA, ma), angle = ellipse
                    detection_center_x, detection_center_y = int(x), int(y)

                    # Calculate average radius and prepare info for JSON (without "type")
                    avg_radius = (float(MA) + float(ma)) / 2.0
                    detection_info_for_json = {
                        "center": [int(x), int(y)],
                        "radius": avg_radius
                    }
    # --- End Detection ---

    # --- Prepare JSON Entry for *this* frame ---
    # Append entry for the current frame, using detection_info_for_json if detected, otherwise None
    json_entry = {
        "frame_id": frame_id,
        "detection": detection_info_for_json # This will be the dict or None
    }
    json_output_data.append(json_entry)
    # ------------------------------------------


    # --- Kalman Filter Processing ---
    current_tracked_pos = None # Position for this frame from Kalman Filter

    if kalman_initialized:
        prediction = kalman.predict()
        pred_x, pred_y = prediction[0][0], prediction[1][0]

        if detected: # Use the actual detection if available
            measurement = np.array([[np.float32(detection_center_x)], [np.float32(detection_center_y)]])
            kalman.correct(measurement)
            current_tracked_pos = (int(kalman.statePost[0][0]), int(kalman.statePost[1][0]))
        else: # Use prediction if no detection
            current_tracked_pos = (int(pred_x), int(pred_y))

        tracked_path.append(current_tracked_pos)

    elif detected: # First detection - Initialize Kalman Filter
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        kalman.statePost = np.array([[np.float32(detection_center_x)], [np.float32(detection_center_y)], [0.], [0.]], dtype=np.float32)
        kalman_initialized = True
        current_tracked_pos = (detection_center_x, detection_center_y) # Use first detection coords
        tracked_path.append(current_tracked_pos)
    # --- End Kalman Filter ---


    # --- Prepare CSV Data for *this* frame ---
    if current_tracked_pos is not None:
        csv_row = [frame_id, current_tracked_pos[0], current_tracked_pos[1]]
    else:
        # Before Kalman is initialized (no detection yet)
        csv_row = [frame_id, -1, -1]
    csv_output_data.append(csv_row)
    # -----------------------------------------


    # --- Draw Tracked Path and Current Position ---
    if current_tracked_pos:
        cv2.circle(frame, current_tracked_pos, 5, (0, 0, 255), -1) # Kalman position (Red)

    if len(tracked_path) >= 2:
        for i in range(1, len(tracked_path)):
            if tracked_path[i-1] is not None and tracked_path[i] is not None:
                pt1_valid = 0 <= tracked_path[i-1][0] < width and 0 <= tracked_path[i-1][1] < height
                pt2_valid = 0 <= tracked_path[i][0] < width and 0 <= tracked_path[i][1] < height
                if pt1_valid and pt2_valid:
                     cv2.line(frame, tracked_path[i-1], tracked_path[i], (0, 255, 255), 2) # Path (Yellow)
    # --- End Drawing ---


    # Write frame to output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Use waitKey(1) for faster playback
        break

    frame_id += 1

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed {frame_id} frames.")
print(f"Output video saved to: {output_video_path}")


# --- Save JSON Data ---
# Custom JSON encoder to handle potential numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let the base class default method raise the TypeError for other types
        return super(NumpyEncoder, self).default(obj)

try:
    # Ensure None is handled correctly (json.dump does this by default)
    with open(output_json_path, 'w') as f:
        json.dump(json_output_data, f, indent=2, cls=NumpyEncoder)
    print(f"Detection data for all frames saved to: {output_json_path}")
except Exception as e:
    print(f"Error saving JSON file: {e}")
# ---------------------


# --- Save CSV Data ---
try:
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'x', 'y']) # Write header
        writer.writerows(csv_output_data)   # Write data rows
    print(f"Tracked positions saved to: {output_csv_path}")
except Exception as e:
    print(f"Error saving CSV file: {e}")
# ---------------------