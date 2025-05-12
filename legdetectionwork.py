import cv2
import numpy as np
import json
import os
import sys
# urllib.request is not needed as we are removing YOLO file downloads for this script
from datetime import datetime # Kept for consistency, though not actively used here
import mediapipe as mp # Import MediaPipe

# ── CONFIG FOR LEG/POSE TRACKING ───────────────────────────────────────────────
INPUT_VIDEO         = r"F:\STORAGE DOWNLOADS\SE PROJECT\New folder\crick01 (2).mp4" # Use your video path
OUTPUT_DIR          = "leg_pose_tracking_output"    # Output dir name for leg/pose tracking
POSE_CONFIDENCE_THRESH = 0.5 # MediaPipe Pose confidence for landmarks
ORIENTATION_X_DIFF_THRESH = 0.03 # Normalized X difference threshold for orientation (0.0 to 1.0)

# Assumed real-world sizes for Z estimation
LEGS_REGION_WIDTH_MM = 450 # Used for Z estimation of legs region width
FOCAL_LENGTH_FACTOR = 1.2
# ── END CONFIG ────────────────────────────────────────────────────────────────────

class PoseLegTracker:
    def __init__(self, video_path, output_dir,
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
        self.out_video_path = os.path.join(self.output_dir, "leg_pose_annotated_video.mp4")
        self.out_json_path  = os.path.join(self.output_dir, "leg_pose_tracking_data.json")

        self._load_pose_model()

    def _load_pose_model(self):
        """Loads the MediaPipe Pose model."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.pose_conf_thresh,
            min_tracking_confidence=self.pose_conf_thresh,
            model_complexity=1 # 0 (light), 1 (full), 2 (heavy) - adjust as needed
        )
        self.mp_drawing = mp.solutions.drawing_utils
        print("MediaPipe Pose loaded.")

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

    def _detect_legs_pose(self, frame_rgb, frame_h, frame_w):
        """Detects leg region and batsman orientation using MediaPipe Pose."""
        results = self.pose.process(frame_rgb)
        legs_data_output = None # Renamed to avoid confusion with a potential class member
        batsman_orientation = "U"

        if results.pose_landmarks:
            all_landmarks = results.pose_landmarks.landmark
            batsman_orientation = self._determine_batsman_orientation(all_landmarks)

            keypoints_indices_for_bbox = { # Landmarks for leg bounding box
                "left_hip": self.mp_pose.PoseLandmark.LEFT_HIP.value,
                "right_hip": self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                "left_knee": self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                "right_knee": self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                "left_ankle": self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
                "right_ankle": self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
            }

            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            relevant_kpts_count = 0

            for name, idx in keypoints_indices_for_bbox.items():
                if idx < len(all_landmarks) and all_landmarks[idx].visibility > self.pose_conf_thresh:
                    kpt = all_landmarks[idx]
                    px, py = int(kpt.x * frame_w), int(kpt.y * frame_h)
                    min_x, min_y = min(min_x, px), min(min_y, py)
                    max_x, max_y = max(max_x, px), max(max_y, py)
                    relevant_kpts_count += 1

            if relevant_kpts_count >= 2 and max_x > min_x and max_y > min_y: # Need at least 2 points for a valid region
                leg_bbox_x, leg_bbox_y = min_x, min_y
                leg_bbox_w, leg_bbox_h = max_x - min_x, max_y - min_y
                apparent_leg_region_width_px = leg_bbox_w # Use the width of the bbox for Z

                legs_data_output = {
                    "bbox_2d": (leg_bbox_x, leg_bbox_y, leg_bbox_w, leg_bbox_h),
                    "apparent_width_px": apparent_leg_region_width_px,
                    "raw_landmarks": results.pose_landmarks, # For drawing all pose landmarks
                    "batsman_orientation": batsman_orientation
                }
        return legs_data_output

    def process(self):
        """Processes the video for leg/pose tracking."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video: {self.video_path}")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if frame_w == 0 or frame_h == 0 or fps == 0:
            print("Error: Video properties (width, height, or FPS) are invalid.")
            cap.release()
            return

        focal_length_px = self._estimate_focal_length_pixels(frame_w)
        print(f"Estimated focal length: {focal_length_px:.2f} pixels (for frame width {frame_w})")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.out_video_path, fourcc, fps, (frame_w, frame_h))

        frame_data_list = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Get leg and pose information
            legs_info = self._detect_legs_pose(frame_rgb, frame_h, frame_w)

            current_frame_data = {"frame_id": frame_idx, "leg_pose": None}

            if legs_info:
                lx, ly, lw, lh = legs_info["bbox_2d"]
                leg_apparent_width_px = legs_info["apparent_width_px"]
                z_legs_m = self._get_3d_z_meters(leg_apparent_width_px, self.legs_region_width_mm, focal_length_px)

                # Define 3D corners of the leg bounding box
                corner_tl_3d = [int(lx), int(ly), round(z_legs_m, 3)]
                corner_tr_3d = [int(lx + lw), int(ly), round(z_legs_m, 3)]
                corner_br_3d = [int(lx + lw), int(ly + lh), round(z_legs_m, 3)]
                corner_bl_3d = [int(lx), int(ly + lh), round(z_legs_m, 3)]

                batsman_orientation = legs_info.get("batsman_orientation", "U")

                current_frame_data["leg_pose"] = {
                    "corners_3d": [corner_tl_3d, corner_tr_3d, corner_br_3d, corner_bl_3d],
                    "batsman_orientation": batsman_orientation
                }

                # Drawing on the frame
                cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2) # Blue for leg bbox
                cv2.putText(frame, f"Legs Z: {z_legs_m:.2f}m", (lx, ly - 10 if ly > 20 else ly + lh + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                orientation_text_y = ly - 30 if ly > 40 else ly + lh + 35
                cv2.putText(frame, f"Orient: {batsman_orientation}", (lx, orientation_text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Yellow for orientation

                # Draw all pose landmarks if available
                if legs_info.get("raw_landmarks"):
                    self.mp_drawing.draw_landmarks(
                        frame,
                        legs_info["raw_landmarks"],
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1), # Red dots
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1) # Green lines
                    )

            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            frame_data_list.append(current_frame_data)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx} frames for leg/pose tracking...")

        cap.release()
        writer.release()
        if hasattr(self, 'pose') and self.pose: # Ensure pose object exists before trying to close
            self.pose.close()

        with open(self.out_json_path, "w") as jf:
            json.dump(frame_data_list, jf, indent=2)

        print("\nLeg/Pose tracking processing complete!")
        print(f"Annotated video (leg/pose only) saved to: {self.out_video_path}")
        print(f"Leg/Pose tracking data saved to: {self.out_json_path}")


if __name__ == "__main__":
    if not os.path.exists(INPUT_VIDEO):
        print(f"FATAL: Input video not found at {INPUT_VIDEO}")
        print("Please update the INPUT_VIDEO path in the script.")
        sys.exit(1)

    tracker = PoseLegTracker(
        video_path=INPUT_VIDEO,
        output_dir=OUTPUT_DIR,
        pose_confidence_thresh=POSE_CONFIDENCE_THRESH,
        legs_region_width_mm=LEGS_REGION_WIDTH_MM,
        focal_length_factor=FOCAL_LENGTH_FACTOR,
        orientation_x_diff_thresh=ORIENTATION_X_DIFF_THRESH
    )
    tracker.process()