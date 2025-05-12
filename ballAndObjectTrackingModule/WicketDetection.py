##    tHIS IS HOW TO RUN THE CODE 




  ####################       python wicket.py /home/splash/Videos/crick04.mp4





import cv2
import numpy as np
import json
import argparse
import os
from datetime import datetime

class WicketDetector3D:
    def __init__(self, video_path, output_dir="ballTrackingIntermediate"):
        
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
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
        
    def detect_wickets(self):
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}/{self.total_frames}")
            
            wicket_data = self._detect_wickets_in_frame(frame)
            
            # Update persistent tracking
            if wicket_data:
                self.last_known_positions = [(frame_count, data) for data in wicket_data]
            else:
                # Filter out old positions
                self.last_known_positions = [(f, pos) for f, pos in self.last_known_positions 
                                            if frame_count - f <= self.position_memory_frames]
            
            # Store the frame results
            if wicket_data:
                self.wicket_positions.append({
                    "frame_number": frame_count,
                    "timestamp": frame_count / self.fps,
                    "stumps": wicket_data
                })
        
        # Clean up
        self.cap.release()
            
        
        self._save_results()
        
        print(f"Processing complete. Detected wickets in {len(self.wicket_positions)} frames.")
        print(f"Results saved to {self.output_dir}")
    
    def _detect_wickets_in_frame(self, frame):
        
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
            # If height is zero or very small, avoid division by zero and return a default large distance
            if pixel_height <= 0: 
                return 10000.0 # Or some other indicator of invalid distance
            # Assume first detection is at a known distance (e.g., 300cm)
            assumed_distance = 2050.0  # cm
            self.reference_pixel_height = pixel_height
            self.focal_length = (pixel_height * assumed_distance) / self.reference_wicket_height
            print(f"Calibrated focal length: {self.focal_length:.2f} pixels")
            return assumed_distance
        
        # Avoid division by zero if pixel_height is somehow 0
        if pixel_height <= 0:
            return 10000.0 # Or handle as an error/invalid case
            
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
    
    def _save_results(self):
        """Save detection results to a JSON file in the specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json = os.path.join(self.output_dir, f'wicket_positions_3d_{timestamp}.json')
        
        # Prepare results as a list of frame objects
        output_data = []
        for frame_data in self.wicket_positions:
            # Only process frames where stumps were actually detected
            if frame_data["stumps"]:
                # Take the data from the first detected stump for simplicity
                # as the target format shows only one set of stump corners per frame.
                first_stump_data = frame_data["stumps"][0] 
                
                frame_object = {
                    "frame_id": frame_data["frame_number"],
                    "ball": None,  # Placeholder - Not detected by this script
                    "bat": None,   # Placeholder - Not detected by this script
                    "leg": None,   # Placeholder - Not detected by this script
                    "stumps": {
                        # Use lowercase "corners" key as per the example
                        "corners": first_stump_data["corners"] 
                    },
                    "batsman_orientation": None # Placeholder - Not detected by this script
                }
                output_data.append(frame_object)
            # Else: If frame_data["stumps"] was empty (shouldn't happen with current logic),
            # we simply skip adding this frame to the output, as per the plan.

        # Save the list of frame objects as a JSON array
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2) # Using indent=2 for readability like the example
            
        print(f"Saved detection results in the new format to {output_json}")


def main():
    parser = argparse.ArgumentParser(description='Detect cricket wickets in 3D from a video and save results to JSON.')
    parser.add_argument('video_path', help='Path to the cricket video file')
    parser.add_argument('--output-dir', default='ballTrackingIntermediate', 
                        help='Directory to save output JSON file (default: ballTrackingIntermediate)')
    parser.add_argument('--memory-frames', type=int, default=30, 
                        help='Number of frames to remember wicket positions for potential future use (e.g., tracking continuity)')
    parser.add_argument('--reference-height', type=float, default=71.1,
                        help='Reference wicket height in centimeters (default: 71.1 cm)')
    
    args = parser.parse_args()
    
    try:
        detector = WicketDetector3D(args.video_path, args.output_dir)
        detector.position_memory_frames = args.memory_frames
        detector.reference_wicket_height = args.reference_height
        detector.detect_wickets()
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())