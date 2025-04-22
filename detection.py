import cv2
import numpy as np
import json

# Define the lower and upper boundaries for the red color in HSV space
lower_red1 = np.array([0, 100, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 50])
upper_red2 = np.array([180, 255, 255])

cap = cv2.VideoCapture("frontslow-cropped.mp4")

# Video properties for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('detections_overlaid.mp4', fourcc, fps, (width, height))

# Kalman filter setup
kalman = None
kalman_initialized = False
tracked_path = []

# Detection tracking
first_detection_frame = None
last_detection_frame = None
output_data = []
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for detection
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Detect circles or ellipses
    mask_gray = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(mask_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=15, minRadius=10, maxRadius=100)
    detected = False
    detection_info = None
    x, y = -1, -1

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        largest_circle = max(circles, key=lambda c: c[2])
        x, y, r = largest_circle
        cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
        cv2.circle(frame, (x, y), 2, (255, 0, 0), 3)
        detected = True
        detection_info = {"type": "circle", "center": [int(x), int(y)], "radius": int(r)}
    else:
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) >= 5 and cv2.contourArea(largest_contour) > 100:
                ellipse = cv2.fitEllipse(largest_contour)
                cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                detected = True
                (x, y), (MA, ma), angle = ellipse
                detection_info = {
                    "type": "ellipse",
                    "center": [int(x), int(y)],
                    "major_axis": float(MA),  # Convert to float for JSON
                    "minor_axis": float(ma),  # Convert to float for JSON
                    "angle": float(angle)     # Convert to float for JSON
                }

    # Update first and last detection frames
    if detected:
        if first_detection_frame is None:
            first_detection_frame = frame_id
        last_detection_frame = frame_id

    # Kalman filter processing
    current_pos = None
    if kalman_initialized:
        # Predict
        prediction = kalman.predict()
        pred_x, pred_y = prediction[0][0], prediction[1][0]

        if detected:
            # Correct with measurement
            measurement = np.array([[x], [y]], dtype=np.float32)
            kalman.correct(measurement)
            current_x, current_y = kalman.statePost[0][0], kalman.statePost[1][0]
        else:
            current_x, current_y = pred_x, pred_y

        current_pos = (int(round(current_x)), int(round(current_y)))  # Ensure integer coordinates
        tracked_path.append(current_pos)
        # Draw Kalman prediction
        cv2.circle(frame, current_pos, 5, (0, 0, 255), -1)

    elif detected:
        # Initialize Kalman filter
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0.95, 0],
            [0, 0, 0, 0.95]
        ], dtype=np.float32)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        kalman_initialized = True
        current_pos = (int(x), int(y))
        tracked_path.append(current_pos)
        cv2.circle(frame, current_pos, 5, (0, 0, 255), -1)

    # Draw tracked path
    if len(tracked_path) >= 2:
        for i in range(1, len(tracked_path)):
            if tracked_path[i-1] is not None and tracked_path[i] is not None:
                cv2.line(frame, tracked_path[i-1], tracked_path[i], (0, 255, 255), 2)

    # Prepare JSON entry
    entry = {
        "frame_id": frame_id,
        "detection": detection_info if detected else "nothing detected",
        "position": [int(current_pos[0]), int(current_pos[1])] if current_pos is not None else None
    }
    output_data.append(entry)

    # Write frame to output video
    out.write(frame)

    cv2.imshow("Ball Tracking", frame)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Collect and combine circle and ellipse detections in a single file
combined_detections = []

for entry in output_data:
    det = entry.get('detection', {})
    
    if isinstance(det, dict) and det.get('type') == "circle":
        combined_detections.append({
            "frame_id": entry['frame_id'],
            "detection": {
                "type": "circle",
                "center": det['center'],
                "radius": det['radius']
            }
        })

    elif isinstance(det, dict) and det.get('type') == "ellipse":
        combined_detections.append({
            "frame_id": entry['frame_id'],
            "detection": {
                "type": "ellipse",
                "center": det['center'],
                "major_axis": det['major_axis'],
                "minor_axis": det['minor_axis'],
                "angle": det['angle']
            }
        })

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Save combined detections to a single JSON file
with open('combined_detections.json', 'w') as f:
    json.dump(combined_detections, f, indent=2, cls=NumpyEncoder)