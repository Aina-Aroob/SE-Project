import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2
import mediapipe as mp
import numpy as np
import json
import sys
import argparse
from datetime import datetime

# ── CONFIG DEFAULTS ────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR      = "ballTrackingIntermediate"
BAT_WIDTH_MM            = 108
FOCAL_LENGTH_FACTOR     = 1.2
POSE_CONFIDENCE_THRESH  = 0.5
BAT_ASPECT_MIN          = 2.0
BAT_ASPECT_MAX          = 10.0
BAT_MIN_AREA            = 300
WRIST_PROXIMITY_THRESH  = 250
# ── END CONFIG DEFAULTS ───────────────────────────────────────────────────────

class BatTracker:
    def __init__(self, video_path,
                 output_dir=DEFAULT_OUTPUT_DIR,
                 bat_width_mm=BAT_WIDTH_MM,
                 focal_length_factor=FOCAL_LENGTH_FACTOR,
                 pose_confidence=POSE_CONFIDENCE_THRESH,
                 bat_aspect_min=BAT_ASPECT_MIN,
                 bat_aspect_max=BAT_ASPECT_MAX,
                 bat_min_area=BAT_MIN_AREA,
                 wrist_proximity_thresh=WRIST_PROXIMITY_THRESH):
                 
        self.video_path = video_path
        self.output_dir = output_dir
        self.bat_width_mm = bat_width_mm
        self.focal_length_factor = focal_length_factor
        self.pose_confidence = pose_confidence
        self.bat_aspect_min = bat_aspect_min
        self.bat_aspect_max = bat_aspect_max
        self.bat_min_area = bat_min_area
        self.wrist_proximity_thresh = wrist_proximity_thresh
        
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_video_filename = os.path.basename(video_path).split('.')[0]
        self.out_json_path = os.path.join(self.output_dir, f"bat_data_{safe_video_filename}_{timestamp}.json")

        self._load_pose_model()

    def _load_pose_model(self):
        """Loads the MediaPipe Pose model."""
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False, 
                min_detection_confidence=self.pose_confidence,
                min_tracking_confidence=self.pose_confidence
            )
            print("MediaPipe Pose loaded for bat tracking.")
        except Exception as e:
            print(f"Error loading MediaPipe Pose model: {e}")
            raise ImportError(f"Failed to load MediaPipe Pose: {e}")

    # --- Z Estimation Functions ---
    def _estimate_focal_length_pixels(self, frame_width):
        return frame_width * self.focal_length_factor

    def _get_3d_z_meters(self, apparent_size_px, real_size_mm, focal_length_px):
        if apparent_size_px > 0 and real_size_mm > 0 and focal_length_px > 0:
            z_mm = (focal_length_px * real_size_mm) / apparent_size_px
            return min(z_mm / 1000.0, 50.0)
        return 50.0
    # --- End Z Estimation Functions ---

    # Modified detect_bat function
    def _detect_bat(self, frame, region, wrist_center, focal_length_px):
        """
        Detects the bat within a region based on color, shape, and proximity to the wrist.
        Returns the 3D corners [x,y,z] of the bat's bounding box relative to the full frame.
        """
        x, y, w, h = region
        y_end = min(y + h, frame.shape[0])
        x_end = min(x + w, frame.shape[1])
        roi = frame[y:y_end, x:x_end]
        
        if roi.size == 0: 
            return None
            
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        lower_bat = np.array([0, 0, 50])
        upper_bat = np.array([179, 255, 180])
        bat_mask = cv2.inRange(hsv, lower_bat, upper_bat)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([179, 60, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        bat_mask = cv2.bitwise_and(bat_mask, ~white_mask)

        contours, _ = cv2.findContours(bat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_bat_data = None
        min_distance = float('inf')

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.bat_min_area:
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                if w_c == 0: continue
                aspect_ratio = h_c / float(w_c)
                
                if self.bat_aspect_min < aspect_ratio < self.bat_aspect_max:
                    cx_roi = x_c + w_c // 2
                    cy_roi = y_c + h_c // 2
                    cx_frame = x + cx_roi
                    cy_frame = y + cy_roi

                    dist = np.sqrt((cx_frame - wrist_center[0])**2 + (cy_frame - wrist_center[1])**2)

                    if dist < min_distance and dist < self.wrist_proximity_thresh:
                        min_distance = dist
                        
                        apparent_width_px = w_c
                        z_bat_m = self._get_3d_z_meters(apparent_width_px, self.bat_width_mm, focal_length_px)
                        
                        bbox_frame_x = x + x_c
                        bbox_frame_y = y + y_c
                        bbox_frame_w = w_c
                        bbox_frame_h = h_c
                        
                        best_bat_data = (bbox_frame_x, bbox_frame_y, bbox_frame_w, bbox_frame_h, z_bat_m)

        if best_bat_data:
            bx, by, bw, bh, bz = best_bat_data
            corners_3d = [
                [int(bx), int(by), round(bz, 3)],
                [int(bx + bw), int(by), round(bz, 3)],
                [int(bx + bw), int(by + bh), round(bz, 3)],
                [int(bx), int(by + bh), round(bz, 3)]
            ]
            return corners_3d
        else:
            return None

    def process(self):
        """Processes the video, detects bat, and saves 3D corners to JSON."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if frame_w == 0 or frame_h == 0 or fps is None or fps == 0:
            error_msg = f"Video properties invalid (w={frame_w}, h={frame_h}, fps={fps})"
            cap.release()
            if hasattr(self, 'pose') and self.pose: self.pose.close()
            raise ValueError(error_msg)
            
        focal_length_px = self._estimate_focal_length_pixels(frame_w)
        print(f"Estimated focal length: {focal_length_px:.2f} pixels")
        
        output_data = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            current_bat_data = None
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                    right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    
                    if left_wrist.visibility > self.pose_confidence and right_wrist.visibility > self.pose_confidence:
                        lx, ly = int(left_wrist.x * frame_w), int(left_wrist.y * frame_h)
                        rx, ry = int(right_wrist.x * frame_w), int(right_wrist.y * frame_h)
                        
                        top_y = min(ly, ry) - int(0.1 * frame_h)
                        bot_y = max(ly, ry) + int(0.4 * frame_h)
                        left_x = min(lx, rx) - int(0.1 * frame_w)
                        right_x = max(lx, rx) + int(0.1 * frame_w)
                        
                        top_y = max(0, top_y)
                        bot_y = min(frame_h, bot_y)
                        left_x = max(0, left_x)
                        right_x = min(frame_w, right_x)
                        region_w = right_x - left_x
                        region_h = bot_y - top_y

                        if region_w > 0 and region_h > 0:
                            bat_region = (left_x, top_y, region_w, region_h)
                            
                            ref_wrist_center = (lx, ly) if ly > ry else (rx, ry) 
                            
                            bat_corners = self._detect_bat(frame, bat_region, ref_wrist_center, focal_length_px)
                            
                            if bat_corners:
                                current_bat_data = {"corners": bat_corners}
                                
                except (IndexError, AttributeError):
                    pass

            frame_object = {
                "frame_id": frame_idx,
                "ball": None, 
                "bat": current_bat_data,
                "leg": None, 
                "stumps": None,
                "batsman_orientation": None
            }
            output_data.append(frame_object)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames for bat tracking...", end='\r')

        cap.release()
        if hasattr(self, 'pose') and self.pose: self.pose.close()
        print(f"\nFinished processing {frame_idx} frames.")

        with open(self.out_json_path, "w") as jf:
            json.dump(output_data, jf, indent=2)
            
        print(f"\nBat tracking processing complete!")
        print(f"Bat tracking data saved to: {self.out_json_path}")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect bat using MediaPipe Pose and save 3D corner data to JSON.')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, 
                        help=f'Directory to save output JSON file (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--pose-conf', type=float, default=POSE_CONFIDENCE_THRESH,
                        help='MediaPipe Pose confidence threshold')

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"FATAL: Input video not found at {args.video_path}")
        sys.exit(1)
        
    print("Starting Bat Tracking...")
    try:
        tracker = BatTracker(
            video_path=args.video_path,
            output_dir=args.output_dir,
            pose_confidence=args.pose_conf
        )
        tracker.process()
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    print("Script finished.")
