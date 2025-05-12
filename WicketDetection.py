
##    tHIS IS HOW TO RUN THE CODE 




  ####################       python wicket.py /home/splash/Videos/crick04.mp4





import cv2
import numpy as np
import json
import argparse
import os
from datetime import datetime

class WicketDetector3D:
    def __init__(self, video_path, output_dir="output"):
        
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
    
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    
        self.wicket_positions = []
        
        
        self.lower_wicket = np.array([25,  35, 200])  # H, S, V lower bound
        self.upper_wicket = np.array([35,  80, 255])  # H, S, V upper bound
        
    
        self.last_known_positions = []
        self.position_memory_frames = 30  # Remember positions for this many frames
        
        # Z-axis estimation parameters
        self.reference_wicket_height = 71.1  # Standard cricket stumps height in cm
        self.focal_length = None  # Will be calibrated based on first clear detection
        self.reference_pixel_height = None  # Will be set during first clear detection
        
    def detect_wickets(self, display_live=True, save_video=True):
        
        
        if save_video:
            output_video_path = os.path.join(self.output_dir, 'processed_video.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}/{self.total_frames}")
            
            # Create a copy of the frame for drawing
            display_frame = frame.copy()
            
            # Detect wickets in this frame
            wicket_data = self._detect_wickets_in_frame(frame, display_frame)
            
            # Update persistent tracking
            if wicket_data:
                self.last_known_positions = [(frame_count, data) for data in wicket_data]
            else:
                # Filter out old positions
                self.last_known_positions = [(f, pos) for f, pos in self.last_known_positions 
                                            if frame_count - f <= self.position_memory_frames]
            
            # Draw persistent highlights from memory
            self._draw_persistent_highlights(display_frame, frame_count)
            
            # Store the frame results
            if wicket_data:
                self.wicket_positions.append({
                    "frame_number": frame_count,
                    "timestamp": frame_count / self.fps,
                    "stumps": wicket_data
                })
            
            # Show live visualization if requested
            if display_live:
                # Add frame info
                cv2.putText(display_frame, f"Frame: {frame_count}/{self.total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show the frame
                cv2.imshow('Cricket Wicket 3D Detection', display_frame)
                
                # Break if 'q' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # Write the frame to output video if requested
            if save_video:
                out.write(display_frame)
        
        # Clean up
        self.cap.release()
        if save_video:
            out.release()
        if display_live:
            cv2.destroyAllWindows()
            
        
        self._save_results()
        
        print(f"Processing complete. Detected wickets in {len(self.wicket_positions)} frames.")
        print(f"Results saved to {self.output_dir}")
    
    def _detect_wickets_in_frame(self, frame, display_frame):
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask for potential wicket colors
        mask = cv2.inRange(hsv, self.lower_wicket, self.upper_wicket)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        wicket_data_list = []
        
        for contour in contours:
            # Filter contours by area to eliminate small noise
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum area threshold
                continue
                
           
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w
            
            
            if aspect_ratio > 2.0:  # Adjust this threshold based on videos
                # Calculate center point of the wicket
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Estimate Z coordinate (distance from camera)
                z_coord = self._estimate_z_coordinate(h)
                
                # Calculate the 3D coordinates of the four corners of the wicket
                corners = self._calculate_wicket_corners(x, y, w, h, z_coord)
                
                # Draw active detection with bright green box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Draw filled center dot
                cv2.circle(display_frame, (center_x, center_y), 7, (0, 0, 255), -1)
                cv2.circle(display_frame, (center_x, center_y), 9, (255, 255, 255), 2)  # White outline
                
                # Add text with 3D coordinates
                cv2.putText(display_frame, f"Wicket ({center_x}, {center_y}, {z_coord:.1f}cm)", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Add corner points
                for i, (cx, cy, cz) in enumerate(corners):
                    cv2.circle(display_frame, (int(cx), int(cy)), 4, (0, 255, 255), -1)
                    cv2.putText(display_frame, f"{i+1}", (int(cx)-5, int(cy)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Add to results in the requested format
                wicket_data = {
                    "center": [center_x, center_y, z_coord],
                    "corners": corners,
                    "bound_box": {
                        "top_left": [x, y, z_coord],
                        "width": w,
                        "height": h,
                        "depth": w * 0.2  # Assuming wicket depth is about 20% of width
                    }
                }
                
                wicket_data_list.append(wicket_data)
        
        return wicket_data_list
    
    def _estimate_z_coordinate(self, pixel_height):
        
        if self.focal_length is None:
            # Assume first detection is at a known distance (e.g., 300cm)
            assumed_distance = 300.0  # cm
            self.reference_pixel_height = pixel_height
            self.focal_length = (pixel_height * assumed_distance) / self.reference_wicket_height
            print(f"Calibrated focal length: {self.focal_length:.2f} pixels")
            return assumed_distance
        
        # Calculate distance based on focal length and known wicket height
        z_coord = (self.focal_length * self.reference_wicket_height) / pixel_height
        return z_coord
    
    def _calculate_wicket_corners(self, x, y, w, h, z):
        
        # Calculate the 3D coordinates of the four corners  clockwise order from top-left
        # Top-left, top-right, bottom-right, bottom-left
        corners = [
            [x, y, z],
            [x + w, y, z],
            [x + w, y + h, z],
            [x, y + h, z]
        ]
        return corners
    
    def _draw_persistent_highlights(self, frame, current_frame):
        
        for frame_num, wicket_data in self.last_known_positions:
            # Calculate fading effect based on age
            age = current_frame - frame_num
            alpha = max(0.3, 1.0 - (age / self.position_memory_frames))
            
            center_x, center_y, z_coord = wicket_data["center"]
            bound_box = wicket_data["bound_box"]
            x = bound_box["top_left"][0]
            y = bound_box["top_left"][1]
            w = bound_box["width"]
            h = bound_box["height"]
            
            # Draw semi-transparent rectangle for persistent highlighting
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 165, 255), 2)
            
            # Draw dot at center
            cv2.circle(overlay, (int(center_x), int(center_y)), 5, (0, 165, 255), -1)
            
            # Apply the overlay with transparency
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Add faded text with coordinates
            text_color = (0, 165, 255)
            cv2.putText(frame, f"({int(center_x)}, {int(center_y)}, {z_coord:.1f})", 
                       (x + w + 5, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    def _save_results(self):
        """Save detection results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json = os.path.join(self.output_dir, f'wicket_positions_3d_{timestamp}.json')
        
        # Format the results
        formatted_results = []
        for frame_data in self.wicket_positions:
            stumps_list = []
            for stump in frame_data["stumps"]:
                formatted_stump = {
                    "CORNERS": stump["corners"],
                    "BOUND_BOX": stump["bound_box"]
                }
                stumps_list.append(formatted_stump)
            
            formatted_results.append({
                "frame_number": frame_data["frame_number"],
                "timestamp": frame_data["timestamp"],
                "STUMPS": stumps_list
            })
        
        results = {
            "video_path": self.video_path,
            "video_dimensions": {"width": self.width, "height": self.height},
            "fps": self.fps,
            "total_frames": self.total_frames,
            "wicket_detections": formatted_results
        }
        
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Saved detection results to {output_json}")


def main():
    parser = argparse.ArgumentParser(description='Detect cricket wickets in 3D from a video')
    parser.add_argument('video_path', help='Path to the cricket video file')
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    parser.add_argument('--no-live', action='store_true', help='Disable live visualization')
    parser.add_argument('--no-save-video', action='store_true', help='Do not save the processed video')
    parser.add_argument('--memory-frames', type=int, default=30, 
                        help='Number of frames to remember wicket positions')
    parser.add_argument('--reference-height', type=float, default=71.1,
                        help='Reference wicket height in centimeters (default: 71.1 cm)')
    
    args = parser.parse_args()
    
    try:
        detector = WicketDetector3D(args.video_path, args.output_dir)
        detector.position_memory_frames = args.memory_frames
        detector.reference_wicket_height = args.reference_height
        detector.detect_wickets(display_live=not args.no_live, save_video=not args.no_save_video)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


def run_webcam_detector():
    """Run 3D wicket detection on webcam feed or live video input"""
    # Initialize webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set up HSV color thresholds for wicket detection
    lower_wicket = np.array([25, 35, 200])
    upper_wicket = np.array([35, 80, 255])
    
    # For tracking positions
    last_positions = []
    memory_frames = 30
    frame_count = 0
    
    # Z-axis estimation parameters
    reference_wicket_height = 71.1  # Standard cricket stumps height in cm
    focal_length = None  # Will be calibrated based on first clear detection
    
    print("Live 3D wicket detection started!")
    print("Press 'q' to quit")
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        display_frame = frame.copy()
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask and clean it up
        mask = cv2.inRange(hsv, lower_wicket, upper_wicket)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store current frame detections
        current_detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w
            
            if aspect_ratio > 2.0:
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Estimate Z coordinate
                if focal_length is None:
                    # Calibrate with first detection (assume 300cm distance)
                    assumed_distance = 300.0
                    focal_length = (h * assumed_distance) / reference_wicket_height
                    print(f"Calibrated focal length: {focal_length:.2f} pixels")
                    z_coord = assumed_distance
                else:
                    # Calculate Z based on calibrated focal length
                    z_coord = (focal_length * reference_wicket_height) / h
                
                # Calculate corners
                corners = [
                    [x, y, z_coord],
                    [x + w, y, z_coord],
                    [x + w, y + h, z_coord],
                    [x, y + h, z_coord]
                ]
                
                # Bright green box for active detection
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Prominent red dot with white outline
                cv2.circle(display_frame, (center_x, center_y), 7, (0, 0, 255), -1)
                cv2.circle(display_frame, (center_x, center_y), 9, (255, 255, 255), 2)
                
                # Add text with 3D coordinates
                cv2.putText(display_frame, f"Wicket ({center_x}, {center_y}, {z_coord:.1f}cm)", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw corner points
                for i, (cx, cy, cz) in enumerate(corners):
                    cv2.circle(display_frame, (int(cx), int(cy)), 4, (0, 255, 255), -1)
                    cv2.putText(display_frame, f"{i+1}", (int(cx)-5, int(cy)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Store detection data
                wicket_data = {
                    "center": [center_x, center_y, z_coord],
                    "corners": corners,
                    "bound_box": {
                        "top_left": [x, y, z_coord],
                        "width": w,
                        "height": h,
                        "depth": w * 0.2  # Approximated depth
                    }
                }
                
                current_detections.append(wicket_data)
        
        # Update position history
        if current_detections:
            last_positions = [(frame_count, pos) for pos in current_detections]
        else:
            last_positions = [(f, pos) for f, pos in last_positions 
                            if frame_count - f <= memory_frames]
        
        # Draw persistent highlights
        for f_num, wicket_data in last_positions:
            if f_num != frame_count:  # Don't redraw current detections
                age = frame_count - f_num
                alpha = max(0.3, 1.0 - (age / memory_frames))
                
                center_x, center_y, z_coord = wicket_data["center"]
                bound_box = wicket_data["bound_box"]
                x = bound_box["top_left"][0]
                y = bound_box["top_left"][1]
                w = bound_box["width"]
                h = bound_box["height"]
                
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.circle(overlay, (int(center_x), int(center_y)), 5, (0, 165, 255), -1)
                cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
                
                text_color = (0, 165, 255)
                cv2.putText(display_frame, f"({int(center_x)}, {int(center_y)}, {z_coord:.1f})", 
                           (x + w + 5, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Show mask for debugging
        cv2.imshow("Wicket Mask", mask)
        
        # Show the result frame
        cv2.putText(display_frame, "Live 3D Wicket Detection (Press 'q' to quit)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Cricket Wicket 3D Detection", display_frame)
        
        # Wait for key press - use a small wait time for responsive UI
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Check if any command line arguments were provided
    if len(os.sys.argv) > 1:
        # Run with command line arguments
        exit(main())
    else:
        # No arguments provided, run webcam detector
        print("No video file specified. Starting live webcam 3D detection.")
        run_webcam_detector()