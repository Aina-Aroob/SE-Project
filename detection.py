import cv2
import numpy as np
import json
import os
import sys
import urllib.request
from datetime import datetime
from collections import deque

# ── CONFIG FOR BALL TRACKING ───────────────────────────────────────────────────
INPUT_VIDEO         = r"F:\STORAGE DOWNLOADS\SE PROJECT\New folder\crick01 (2).mp4" # Use your video path
OUTPUT_DIR          = "ball_only_tracking_output"      # Output dir name for ball tracking
YOLO_CONFIDENCE_THRESH = 0.4  # Lowered to catch more potential ball detections
FALLBACK_CONFIDENCE = 0.3    # Even lower threshold used after ball is lost

# Memory settings for trajectory prediction
TRAJECTORY_MEMORY = 8        # Increased to have more history for better trajectory modeling
MAX_FRAMES_LOST = 15         # Maximum frames to continue predicting without detection

# YOLO Files
YOLO_CFG_URL        = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
YOLO_WEIGHTS_URL    = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
YOLO_CFG_FILE       = "yolov4.cfg"
YOLO_WEIGHTS_FILE   = "yolov4.weights"

# Assumed real-world sizes for Z estimation
BALL_DIAMETER_MM = 73
FOCAL_LENGTH_FACTOR = 1.2

# Physics modeling constants
GRAVITY_PX = 2.0  # Gravity effect in pixels per frame squared (positive = downward)
BOUNCE_FACTOR = 0.7  # Energy retained after bounce
# ── END CONFIG ────────────────────────────────────────────────────────────────────

def ensure_yolo_files():
    """Download YOLO cfg/weights if they're not present."""
    if not os.path.exists(YOLO_CFG_FILE):
        print(f"Downloading {YOLO_CFG_FILE}...")
        urllib.request.urlretrieve(YOLO_CFG_URL, YOLO_CFG_FILE)
    if not os.path.exists(YOLO_WEIGHTS_FILE):
        print(f"Downloading {YOLO_WEIGHTS_FILE} (this can take a while)...")
        urllib.request.urlretrieve(YOLO_WEIGHTS_URL, YOLO_WEIGHTS_FILE)

class YoloBallTracker:
    def __init__(self, video_path, output_dir,
                 yolo_confidence_thresh=YOLO_CONFIDENCE_THRESH,
                 ball_diameter_mm=BALL_DIAMETER_MM,
                 focal_length_factor=FOCAL_LENGTH_FACTOR):
        self.video_path = video_path
        self.output_dir = output_dir
        self.yolo_conf_thresh = yolo_confidence_thresh
        self.fallback_conf_thresh = FALLBACK_CONFIDENCE
        self.ball_diameter_mm = ball_diameter_mm
        self.focal_length_factor = focal_length_factor
        
        # Motion tracking variables
        self.ball_history = deque(maxlen=TRAJECTORY_MEMORY)
        self.frames_since_detection = 0
        self.last_ball_velocity = (0, 0)  # (vx, vy)
        self.last_reliable_ball = None
        
        # Physics modeling variables
        self.bounce_detected = False
        self.bounce_frame = -1
        self.gravity = GRAVITY_PX
        self.bounce_factor = BOUNCE_FACTOR
        self.frame_height = 0  # Will be set when video is opened
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.out_video_path = os.path.join(self.output_dir, "ball_annotated_video.mp4")
        self.out_json_path  = os.path.join(self.output_dir, "ball_tracking_data.json")

        ensure_yolo_files()
        self._load_yolo_model()
        
    def _load_yolo_model(self):
        """Loads the YOLOv4 model."""
        try:
            self.net = cv2.dnn.readNetFromDarknet(YOLO_CFG_FILE, YOLO_WEIGHTS_FILE)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # Use CPU for wider compatibility

            layers = self.net.getLayerNames()
            unconnected_out_layers_indices = self.net.getUnconnectedOutLayers()

            if isinstance(unconnected_out_layers_indices, np.ndarray) and unconnected_out_layers_indices.ndim > 1:
                self.out_layers_names = [layers[i[0] - 1] for i in unconnected_out_layers_indices]
            elif isinstance(unconnected_out_layers_indices, (list, np.ndarray)):
                 self.out_layers_names = [layers[i - 1] for i in unconnected_out_layers_indices]
            else:
                try:
                    self.out_layers_names = [layers[i[0] - 1] for i in unconnected_out_layers_indices]
                except (TypeError, IndexError):
                    self.out_layers_names = [layers[i - 1] for i in unconnected_out_layers_indices]

            print("YOLOv4 loaded for ball detection.")
        except cv2.error as e:
            print(f"Error loading YOLO model or files: {e}")
            print(f"Ensure '{YOLO_CFG_FILE}' and '{YOLO_WEIGHTS_FILE}' are present.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during YOLO model loading: {e}")
            sys.exit(1)

    def _estimate_focal_length_pixels(self, frame_width):
        """Estimates focal length in pixels based on frame width and a factor."""
        return frame_width * self.focal_length_factor

    def _get_3d_z_meters(self, apparent_size_px, real_size_mm, focal_length_px):
        """Estimates Z-depth in meters."""
        if apparent_size_px > 0 and real_size_mm > 0 and focal_length_px > 0:
            z_mm = (focal_length_px * real_size_mm) / apparent_size_px
            return min(z_mm / 1000.0, 50.0) # Convert to meters, cap at 50m
        return 50.0 # Default large Z if calculation is not possible

    def _detect_ball_yolo(self, frame, use_fallback=False):
        """Detects the most confident 'sports ball' using YOLO with options for fallback."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.out_layers_names)

        boxes = []
        confidences = []
        class_ids = []
        
        # Use fallback confidence if needed
        threshold = self.fallback_conf_thresh if use_fallback else self.yolo_conf_thresh

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence > threshold and class_id == 32: # COCO ID 32 is "sports ball"
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    box_w = int(detection[2] * w)
                    box_h = int(detection[3] * h)
                    x = int(center_x - box_w / 2)
                    y = int(center_y - box_h / 2)

                    boxes.append([x, y, box_w, box_h])
                    confidences.append(confidence)
                    class_ids.append(class_id) # Should always be 32 here

        indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.4) # Non-Max Suppression

        detected_balls_info = []
        final_indices = []
        if indices is not None:
            if hasattr(indices, 'flatten'):
                 final_indices = indices.flatten().tolist()
            elif isinstance(indices, list) and all(isinstance(item, list) and len(item)==1 for item in indices):
                final_indices = [item[0] for item in indices]
            elif isinstance(indices, list):
                 final_indices = indices
            else:
                 final_indices = list(indices)

        if final_indices:
            for i in final_indices:
                x, y, bw, bh = boxes[i]
                conf = confidences[i]
                cx, cy = x + bw // 2, y + bh // 2
                apparent_size_px = (bw + bh) / 2.0 # Average of width and height for Z estimation

                detected_balls_info.append({
                    "bbox_2d": (x, y, bw, bh),
                    "center_2d": (cx, cy),
                    "apparent_size_px": apparent_size_px,
                    "confidence": conf
                })

        best_ball = None
        if detected_balls_info:
            # If we have ball history, prioritize balls that are close to predicted location
            if self.ball_history and len(self.ball_history) >= 2:
                # Calculate predicted position based on velocity
                last_cx, last_cy = self.ball_history[-1]["center_2d"]
                
                # Score each detection by proximity to predicted position + confidence
                for ball in detected_balls_info:
                    cx, cy = ball["center_2d"]
                    predicted_cx = last_cx + self.last_ball_velocity[0]
                    # Apply gravity to predicted y-position
                    predicted_cy = last_cy + self.last_ball_velocity[1] + self.gravity
                    
                    # Calculate distance to predicted position
                    dist = np.sqrt((cx - predicted_cx)**2 + (cy - predicted_cy)**2)
                    
                    # Calculate a combined score (lower distance is better)
                    max_reasonable_distance = 100  # Maximum reasonable movement between frames
                    distance_factor = max(0, 1 - (dist / max_reasonable_distance))
                    ball["tracking_score"] = (ball["confidence"] * 0.7) + (distance_factor * 0.3)
                
                # Get ball with best combined score
                best_ball = max(detected_balls_info, key=lambda b: b['tracking_score'])
            else:
                # Without history, just use the highest confidence detection
                best_ball = max(detected_balls_info, key=lambda b: b['confidence'])

        return best_ball
    
    def _detect_bounce(self, frame_idx):
        """Detects if a bounce has occurred by analyzing velocity changes."""
        if len(self.ball_history) < 3:
            return False
            
        # Get the three most recent positions to analyze direction changes
        point_minus2 = self.ball_history[-3]["center_2d"][1]  # y-coordinate from 2 frames ago
        point_minus1 = self.ball_history[-2]["center_2d"][1]  # y-coordinate from 1 frame ago
        point_current = self.ball_history[-1]["center_2d"][1]  # current y-coordinate
        
        # Calculate direction changes - in screen coordinates, lower y values are higher physically
        velocity_prev = point_minus1 - point_minus2  # Positive = moving down
        velocity_current = point_current - point_minus1  # Positive = moving down
        
        # Check for a bounce (velocity was positive/down, now negative/up)
        # This means the ball was falling and is now rising - a bounce!
        if velocity_prev > 2 and velocity_current < -2:  # Thresholds to avoid detecting small noise
            # Only detect a bounce if we haven't detected one recently (at least 5 frames ago)
            if frame_idx - self.bounce_frame > 5:
                self.bounce_frame = frame_idx
                print(f"Bounce detected at frame {frame_idx}")
                
                # When a bounce occurs, invert the y velocity and apply the bounce factor
                vx, vy = self.last_ball_velocity
                self.last_ball_velocity = (vx, -vy * self.bounce_factor)
                
                return True
                
        return False

    def _predict_ball_position(self, frame_idx=0):
        """Predicts ball position based on recent trajectory with physics."""
        # Need at least 2 points to estimate velocity
        if len(self.ball_history) < 2:
            return None
            
        # Get the two most recent positions
        prev_ball = self.ball_history[-2]
        last_ball = self.ball_history[-1]
        
        # Calculate velocity (movement per frame)
        prev_cx, prev_cy = prev_ball["center_2d"]
        last_cx, last_cy = last_ball["center_2d"]
        vx = last_cx - prev_cx
        vy = last_cy - prev_cy
        
        # Check for a bounce before updating velocity
        bounce_occurred = self._detect_bounce(frame_idx)
        
        if not bounce_occurred:
            # Apply gravity to vertical velocity (only if we didn't just bounce)
            vy += self.gravity
            
            # Update velocity with some smoothing if no bounce occurred
            self.last_ball_velocity = (
                0.7 * vx + 0.3 * self.last_ball_velocity[0],
                0.7 * vy + 0.3 * self.last_ball_velocity[1]
            )
        
        # Calculate predicted position
        pred_cx = int(last_cx + self.last_ball_velocity[0])
        pred_cy = int(last_cy + self.last_ball_velocity[1])
        
        # Boundary check - if we hit the ground (approximated by frame bottom - 50 pixels)
        ground_y = self.frame_height - 50
        if pred_cy > ground_y and self.last_ball_velocity[1] > 0:
            # Reflect off the ground - bounce!
            pred_cy = ground_y - (pred_cy - ground_y)  # Mirror around ground level
            self.last_ball_velocity = (
                self.last_ball_velocity[0],  # x-velocity stays the same
                -self.last_ball_velocity[1] * self.bounce_factor  # y-velocity inverts and reduces
            )
            print(f"Predicted floor bounce at frame {frame_idx} (pred_y:{pred_cy}, ground:{ground_y})")
        
        # Create predicted ball info (copy most attributes from last detection)
        predicted_ball = last_ball.copy()
        predicted_ball["center_2d"] = (pred_cx, pred_cy)
        
        # Adjust bounding box based on new center
        old_x, old_y, bw, bh = predicted_ball["bbox_2d"]
        new_x = int(pred_cx - bw/2)
        new_y = int(pred_cy - bh/2)
        predicted_ball["bbox_2d"] = (new_x, new_y, bw, bh)
        
        # Mark as predicted
        predicted_ball["confidence"] = predicted_ball["confidence"] * 0.9  # Decay confidence
        predicted_ball["is_predicted"] = True
        
        return predicted_ball

    def _select_best_candidate(self, detected_ball, predicted_ball):
        """Selects the best candidate between detection and prediction."""
        # If no detection, use prediction
        if detected_ball is None:
            return predicted_ball
            
        # If no prediction, use detection
        if predicted_ball is None:
            return detected_ball
            
        # If detection has good confidence, use it
        if detected_ball["confidence"] > 0.6:
            return detected_ball
            
        # Calculate distance between detection and prediction
        d_cx, d_cy = detected_ball["center_2d"]
        p_cx, p_cy = predicted_ball["center_2d"]
        distance = np.sqrt((d_cx - p_cx)**2 + (d_cy - p_cy)**2)
        
        # If detection is far from prediction, be cautious
        if distance > 100:  # Threshold distance in pixels
            # If detection is confident enough, trust it as a new detection
            if detected_ball["confidence"] > 0.5:
                return detected_ball
            else:
                # Otherwise trust the prediction
                return predicted_ball
        else:
            # Close enough - use detection
            return detected_ball

    def process(self):
        """Processes the video to detect and track the ball."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video: {self.video_path}")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Store frame height for bounce detection
        self.frame_height = frame_h

        if frame_w == 0 or frame_h == 0:
            print(f"Error: Video frame dimensions are zero for {self.video_path}. Cannot proceed.")
            cap.release()
            return

        focal_length_px = self._estimate_focal_length_pixels(frame_w)
        print(f"Estimated focal length: {focal_length_px:.2f} pixels (for frame width {frame_w})")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.out_video_path, fourcc, fps, (frame_w, frame_h))

        frame_data_list = []
        frame_idx = 0

        # For trajectory visualization
        trajectory_points = []
        predicted_trajectory = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Try to detect the ball with normal confidence first
            ball_info = self._detect_ball_yolo(frame, use_fallback=False)
            
            # If not found with normal confidence, try fallback confidence if we've recently seen the ball
            if ball_info is None and self.frames_since_detection < 10:
                ball_info = self._detect_ball_yolo(frame, use_fallback=True)
                
            # If still not found, try to predict position based on history and physics
            predicted_ball = None
            if self.frames_since_detection < MAX_FRAMES_LOST:
                predicted_ball = self._predict_ball_position(frame_idx)
                
            # If we have both detection and prediction, select the best candidate
            if ball_info is not None and predicted_ball is not None:
                ball_info = self._select_best_candidate(ball_info, predicted_ball)
            elif ball_info is None and predicted_ball is not None:
                ball_info = predicted_ball
                print(f"Frame {frame_idx}: Using predicted position")
            
            current_frame_ball_data = None  # Initialize for JSON

            if ball_info:
                # Update tracking state
                if ball_info.get("is_predicted", False):
                    self.frames_since_detection += 1
                else:
                    self.frames_since_detection = 0
                    self.last_reliable_ball = ball_info

                # Add to history
                self.ball_history.append(ball_info)
                
                # Extract data
                bcx, bcy = ball_info["center_2d"]
                ball_apparent_size_px = ball_info["apparent_size_px"]
                z_ball_m = self._get_3d_z_meters(ball_apparent_size_px, self.ball_diameter_mm, focal_length_px)

                current_frame_ball_data = {
                    "center_3d": [int(bcx), int(bcy), round(z_ball_m, 3)],
                    "radius": round(ball_apparent_size_px / 2.0, 1),  # Apparent radius in pixels
                    #"confidence": round(ball_info["confidence"], 3),
                   # "is_predicted": ball_info.get("is_predicted", False)
                }

                # Save current position for trajectory drawing
                trajectory_points.append((int(bcx), int(bcy)))
                
                # Drawing on the frame
                bx_draw, by_draw, bw_draw, bh_draw = ball_info["bbox_2d"]
                
                # Use different colors for detected vs predicted
                box_color = (0, 255, 0) if not ball_info.get("is_predicted", False) else (0, 165, 255)
                
                cv2.rectangle(frame, (bx_draw, by_draw), (bx_draw + bw_draw, by_draw + bh_draw), box_color, 2)
                cv2.circle(frame, (int(bcx), int(bcy)), 5, (0, 0, 255), -1)  # Center point
                
                # Display text with detection/prediction status
                status_text = "Detected" if not ball_info.get("is_predicted", False) else "Predicted"
                cv2.putText(frame, f"Ball {status_text}", (bx_draw, by_draw - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                cv2.putText(frame, f"Conf: {ball_info['confidence']:.2f}", (bx_draw, by_draw - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                            
                # Draw the actual trajectory (smooth curve)
                if len(trajectory_points) >= 2:
                    for i in range(1, len(trajectory_points)):
                        cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (255, 0, 0), 2)
                        
                # Draw the predicted future trajectory
                if len(self.ball_history) >= 2:
                    # Create a copy of the current ball info for prediction
                    temp_ball = ball_info.copy()
                    temp_vx, temp_vy = self.last_ball_velocity
                    
                    # Predict 10 frames ahead
                    predicted_trajectory = [(int(bcx), int(bcy))]  # Start with current position
                    for i in range(10):
                        # Apply current velocity
                        next_x = predicted_trajectory[-1][0] + temp_vx
                        next_y = predicted_trajectory[-1][1] + temp_vy
                        
                        # Apply gravity to y velocity
                        temp_vy += self.gravity
                        
                        # Check for ground bounce (simplified)
                        ground_y = self.frame_height - 50
                        if next_y > ground_y and temp_vy > 0:
                            # Bounce!
                            next_y = ground_y - (next_y - ground_y)
                            temp_vy = -temp_vy * self.bounce_factor
                            
                        predicted_trajectory.append((int(next_x), int(next_y)))
                    
                    # Draw predicted trajectory with dotted line
                    for i in range(1, len(predicted_trajectory)):
                        cv2.circle(frame, predicted_trajectory[i], 3, (255, 165, 0), -1)  # Orange dots
                
                # Draw velocity vector
                cv2.arrowedLine(
                    frame, 
                    (int(bcx), int(bcy)),
                    (int(bcx + self.last_ball_velocity[0] * 3), int(bcy + self.last_ball_velocity[1] * 3)),
                    (0, 255, 255), 2
                )
                
                # Display if bounce was detected
                if frame_idx - self.bounce_frame < 15:  # Show bounce indicator for 15 frames
                    cv2.putText(frame, "BOUNCE DETECTED!", (frame_w//2 - 100, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Add data for the current frame to the list
            frame_data_list.append({
                "frame_id": frame_idx,
                "ball": current_frame_ball_data  # This will be the dict or None
            })

            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames for ball tracking...")

        cap.release()
        writer.release()

        # Save all ball tracking data to JSON
        with open(self.out_json_path, "w") as jf:
            json.dump(frame_data_list, jf, indent=2)

        print("\nBall tracking processing complete!")
        print(f"Annotated video (ball only) saved to: {self.out_video_path}")
        print(f"Ball tracking data saved to: {self.out_json_path}")


if __name__ == "__main__":
    if not os.path.exists(INPUT_VIDEO):
        print(f"FATAL: Input video not found at {INPUT_VIDEO}")
        print("Please update the INPUT_VIDEO path in the script.")
        sys.exit(1)

    # Create and run the ball tracker
    ball_tracker = YoloBallTracker(
        video_path=INPUT_VIDEO,
        output_dir=OUTPUT_DIR,
        yolo_confidence_thresh=YOLO_CONFIDENCE_THRESH,
        ball_diameter_mm=BALL_DIAMETER_MM,
        focal_length_factor=FOCAL_LENGTH_FACTOR
    )
    ball_tracker.process()