import cv2
import numpy as np
import json
import os
import sys
import argparse # Added for command-line parsing
from datetime import datetime
import mediapipe as mp

# ── CONFIG DEFAULTS ────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR          = "ballTrackingIntermediate" # Changed default output dir
POSE_CONFIDENCE_THRESH = 0.5
ORIENTATION_X_DIFF_THRESH = 0.03
LEGS_REGION_WIDTH_MM = 450
FOCAL_LENGTH_FACTOR = 1.2
# ── END CONFIG DEFAULTS ───────────────────────────────────────────────────────

class PoseLegTracker:
    # Updated constructor to accept parameters and set defaults
    def __init__(self, video_path, 
                 output_dir=DEFAULT_OUTPUT_DIR,
                 pose_confidence_thresh=POSE_CONFIDENCE_THRESH,
                 legs_region_width_mm=LEGS_REGION_WIDTH_MM,
                 focal_length_factor=FOCAL_LENGTH_FACTOR,
                 orientation_x_diff_thresh=ORIENTATION_X_DIFF_THRESH):

        self.video_path = video_path
        self.output_dir = output_dir
        self.pose_conf_thresh = pose_confidence_thresh
        self.legs_region_width_mm = legs_region_width_mm
        self.focal_length_factor = focal_length_factor
        self.orientation_x_diff_thresh = orientation_x_diff_thresh

        os.makedirs(self.output_dir, exist_ok=True)
        # Removed self.out_video_path
        # Construct JSON path dynamically
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_video_filename = os.path.basename(video_path).split('.')[0]
        self.out_json_path  = os.path.join(self.output_dir, f"leg_pose_data_{safe_video_filename}_{timestamp}.json")

        self._load_pose_model()

    def _load_pose_model(self):
        """Loads the MediaPipe Pose model."""
        try:
            self.mp_pose = mp.solutions.pose
            # Added static_image_mode=False for video processing
            self.pose = self.mp_pose.Pose(
                static_image_mode=False, 
                min_detection_confidence=self.pose_conf_thresh,
                min_tracking_confidence=self.pose_conf_thresh,
                model_complexity=1
            )
            # Removed mp_drawing as it's no longer needed
            # self.mp_drawing = mp.solutions.drawing_utils 
            print("MediaPipe Pose loaded.")
        except Exception as e:
            print(f"Error loading MediaPipe Pose model: {e}")
            print("Ensure MediaPipe is installed correctly (`pip install mediapipe`).")
            # Re-raise or handle appropriately for programmatic use
            raise ImportError(f"Failed to load MediaPipe Pose: {e}") 

    def _estimate_focal_length_pixels(self, frame_width):
        """Estimates focal length in pixels."""
        return frame_width * self.focal_length_factor

    def _get_3d_z_meters(self, apparent_size_px, real_size_mm, focal_length_px):
        """Estimates Z-depth in meters."""
        if apparent_size_px > 0 and real_size_mm > 0 and focal_length_px > 0:
            z_mm = (focal_length_px * real_size_mm) / apparent_size_px
            return min(z_mm / 1000.0, 50.0) # Cap depth at 50m
        return 50.0 # Default large Z if calculation is not possible

    def _determine_batsman_orientation(self, pose_landmarks_list):
        """Determines batsman orientation (L/R/U) based on shoulder and hip X-coordinates."""
        try:
            left_shoulder = pose_landmarks_list[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = pose_landmarks_list[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = pose_landmarks_list[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = pose_landmarks_list[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

            min_landmark_visibility = self.pose_conf_thresh
            if not (left_shoulder.visibility > min_landmark_visibility and
                    right_shoulder.visibility > min_landmark_visibility and
                    left_hip.visibility > min_landmark_visibility and
                    right_hip.visibility > min_landmark_visibility):
                return "U"

            ls_x, rs_x = left_shoulder.x, right_shoulder.x
            lh_x, rh_x = left_hip.x, right_hip.x

            r_score, l_score = 0, 0
            shoulder_x_diff = ls_x - rs_x
            if shoulder_x_diff < -self.orientation_x_diff_thresh: r_score += 1
            elif shoulder_x_diff > self.orientation_x_diff_thresh: l_score += 1

            hip_x_diff = lh_x - rh_x
            if hip_x_diff < -self.orientation_x_diff_thresh: r_score += 1
            elif hip_x_diff > self.orientation_x_diff_thresh: l_score += 1

            if r_score > l_score and r_score > 0: return "R"
            if l_score > r_score and l_score > 0: return "L"
            return "U"
        except (IndexError, AttributeError, TypeError):
            return "U"

    # Removed frame_h, frame_w from signature as they are not needed without drawing
    def _detect_legs_pose(self, frame_rgb): 
        """Detects leg region and batsman orientation using MediaPipe Pose."""
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        legs_data_output = None 
        batsman_orientation = "U"
        processed_landmarks = None # Store landmarks if pose detected

        if results.pose_landmarks:
            processed_landmarks = results.pose_landmarks.landmark # Use landmark attribute
            batsman_orientation = self._determine_batsman_orientation(processed_landmarks)

            # Define keypoints for bounding box calculation (pixel coords needed internally here)
            frame_h, frame_w = frame_rgb.shape[:2] # Get dimensions needed for pixel coords
            keypoints_indices_for_bbox = {
                "left_hip": self.mp_pose.PoseLandmark.LEFT_HIP.value,
                "right_hip": self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                "left_knee": self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                "right_knee": self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                "left_ankle": self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
                "right_ankle": self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
            }

            min_x_px, min_y_px = float('inf'), float('inf')
            max_x_px, max_y_px = float('-inf'), float('-inf')
            relevant_kpts_count = 0

            for name, idx in keypoints_indices_for_bbox.items():
                # Check visibility before using landmarks
                if idx < len(processed_landmarks) and processed_landmarks[idx].visibility > self.pose_conf_thresh:
                    kpt = processed_landmarks[idx]
                    px, py = int(kpt.x * frame_w), int(kpt.y * frame_h)
                    min_x_px, min_y_px = min(min_x_px, px), min(min_y_px, py)
                    max_x_px, max_y_px = max(max_x_px, px), max(max_y_px, py)
                    relevant_kpts_count += 1

            # Calculate bbox and apparent width if enough points are visible
            if relevant_kpts_count >= 2 and max_x_px > min_x_px:
                leg_bbox_px = (min_x_px, min_y_px, max_x_px - min_x_px, max_y_px - min_y_px)
                apparent_leg_region_width_px = leg_bbox_px[2] # Use width

                # Store data needed for JSON output
                legs_data_output = {
                    "bbox_2d_pixels": leg_bbox_px, # Store pixel bbox if needed later
                    "apparent_width_px": apparent_leg_region_width_px
                    # raw_landmarks removed as drawing is removed
                }
        
        # Return the calculated data and the orientation
        return legs_data_output, batsman_orientation 

    def process(self):
        """Processes the video for leg/pose tracking, saving only JSON data."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video: {self.video_path}")
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if frame_w == 0 or frame_h == 0 or fps is None or fps == 0:
            error_msg = f"Video properties (width={frame_w}, height={frame_h}, fps={fps}) are invalid."
            print(f"Error: {error_msg}")
            cap.release()
            if hasattr(self, 'pose') and self.pose: self.pose.close()
            raise ValueError(error_msg)

        focal_length_px = self._estimate_focal_length_pixels(frame_w)
        print(f"Estimated focal length: {focal_length_px:.2f} pixels (for frame width {frame_w})")

        # Removed VideoWriter setup
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # writer = cv2.VideoWriter(self.out_video_path, fourcc, fps, (frame_w, frame_h))

        # Changed to store data in the target format
        output_data = [] 
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get leg BBox info and orientation
            # Removed frame_h, frame_w args as they are calculated inside if needed
            legs_info, batsman_orientation = self._detect_legs_pose(frame_rgb)

            current_leg_data = None # Initialize leg data for this frame's JSON object
            current_orientation = batsman_orientation # Use orientation even if no leg bbox found

            if legs_info:
                # Calculate Z-depth based on apparent width
                leg_apparent_width_px = legs_info["apparent_width_px"]
                z_legs_m = self._get_3d_z_meters(leg_apparent_width_px, self.legs_region_width_mm, focal_length_px)
                
                # Get pixel bounding box for corner calculation
                lx_px, ly_px, lw_px, lh_px = legs_info["bbox_2d_pixels"]

                # Define 3D corners using pixel bbox and calculated Z
                corner_tl_3d = [int(lx_px), int(ly_px), round(z_legs_m, 3)]
                corner_tr_3d = [int(lx_px + lw_px), int(ly_px), round(z_legs_m, 3)]
                corner_br_3d = [int(lx_px + lw_px), int(ly_px + lh_px), round(z_legs_m, 3)]
                corner_bl_3d = [int(lx_px), int(ly_px + lh_px), round(z_legs_m, 3)]

                # Format leg data according to the target structure
                current_leg_data = {
                    # Use "corners" key as requested
                    "corners": [corner_tl_3d, corner_tr_3d, corner_br_3d, corner_bl_3d] 
                }
                
                # --- All Drawing Code Removed --- 
                # cv2.rectangle(...)
                # cv2.putText(...)
                # self.mp_drawing.draw_landmarks(...)

            # Create the frame object for the JSON output list
            frame_object = {
                "frame_id": frame_idx,
                "ball": None, # Placeholder
                "bat": None,  # Placeholder
                "leg": current_leg_data, # Will be the dict or None
                "stumps": None, # Placeholder
                "batsman_orientation": current_orientation # Store orientation (L/R/U)
            }
            output_data.append(frame_object)

            # Removed frame writing: writer.write(frame)
            # Removed frame info text: cv2.putText(frame, f"Frame: {frame_idx}" ...)

            frame_idx += 1
            if frame_idx % 50 == 0:
                # Keep progress update
                print(f"Processed {frame_idx} frames for leg/pose tracking...", end='\r') 

        cap.release()
        # Removed writer.release()
        # Close MediaPipe pose object
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()
        print(f"\nFinished processing {frame_idx} frames.")

        # Save the collected data to JSON in the new format
        with open(self.out_json_path, "w") as jf:
            json.dump(output_data, jf, indent=2)

        print("\nLeg/Pose tracking processing complete!")
        # Updated print message
        # print(f"Annotated video (leg/pose only) saved to: {self.out_video_path}")
        print(f"Leg/Pose tracking data saved to: {self.out_json_path}")


# Modified __main__ block for command-line execution and importability
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track leg region and batsman orientation using MediaPipe Pose, saving results to JSON.')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, 
                        help=f'Directory to save output JSON file (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--pose-conf', type=float, default=POSE_CONFIDENCE_THRESH,
                        help='MediaPipe Pose detection/tracking confidence threshold')
    parser.add_argument('--orient-thresh', type=float, default=ORIENTATION_X_DIFF_THRESH,
                         help='Normalized X difference threshold for orientation detection')
    # Add more args if needed

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"FATAL: Input video not found at {args.video_path}")
        sys.exit(1)

    print("Starting Leg/Pose Tracking...")
    try:
        tracker = PoseLegTracker(
            video_path=args.video_path,
            output_dir=args.output_dir,
            pose_confidence_thresh=args.pose_conf,
            orientation_x_diff_thresh=args.orient_thresh
            # Pass other args here
        )
        tracker.process()
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    print("Script finished.")