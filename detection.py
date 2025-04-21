import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("sideslow.mp4")

# Video writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("sideslow_better_output.mp4", fourcc, fps, (width, height))

# For motion tracking
last_position = None
MAX_MOVE_DIST = 100  # pixels

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # PREPROCESS: Sharpen and enhance
    blur = cv2.GaussianBlur(frame, (0, 0), 3)
    sharp = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)

    # Convert to HSV
    hsv = cv2.cvtColor(sharp, cv2.COLOR_BGR2HSV)

    # Red mask (2 ranges)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Optional: Morphology to clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_candidate = None
    best_distance = float('inf')

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > 5000:  # Skip tiny/huge areas
            continue

        # Circularity filter
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.6:
            continue

        # Get center
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))

        # Track closest to last position
        if last_position is not None:
            dist = np.linalg.norm(np.array(center) - np.array(last_position))
            if dist < best_distance and dist < MAX_MOVE_DIST:
                best_distance = dist
                best_candidate = (center, int(radius))
        else:
            best_candidate = (center, int(radius))

    # Draw detection
    if best_candidate:
        center, radius = best_candidate
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.putText(frame, f"Ball: {center}", (center[0]+10, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        last_position = center

    # Save frame
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
