import cv2
import numpy as np
import json

# Define the lower and upper boundaries for the red color in HSV space
lower_red1 = np.array([0, 80, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 80, 50])
upper_red2 = np.array([180, 255, 255])

cap = cv2.VideoCapture("sideslow-chairhidden.mp4")

# Get video properties for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or 'mp4v'
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('detections_overlaid.mp4', fourcc, fps, (width, height))

paused = False
frame_id = 0
output_data = []

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        mask_gray = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(mask_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=100, param2=15, minRadius=10, maxRadius=100)

        detected = False
        detection_info = None

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            largest_circle = max(circles, key=lambda c: c[2])
            x, y, r = largest_circle
            cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), 3)
            detected = True
            detection_info = {"type": "circle", "center": [int(x), int(y)], "radius": int(r)}

        if not detected:
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
                        "major_axis": float(MA),
                        "minor_axis": float(ma),
                        "angle": float(angle)
                    }

        # Write JSON entry for this frame
        if detected:
            output_data.append({
                "frame_id": frame_id,
                "detection": detection_info
            })
        else:
            output_data.append({
                "frame_id": frame_id,
                "detection": "nothing detected"
            })

        # --- Write the processed frame to output video ---
        out.write(frame)

        frame_id += 1

    cv2.imshow("Ball Tracking", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(500) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused

cap.release()
out.release()  # Don't forget to release the VideoWriter!
cv2.destroyAllWindows()

with open('ball_tracking_output.json', 'w') as f:
    json.dump(output_data, f, indent=2)
