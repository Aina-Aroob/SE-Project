import cv2
import numpy as np
import os
import json
import time
from ultralytics import YOLO

class CricketVideoProcessor:
    def __init__(self, video_path, output_dir="output", playback_speed=0.5):
        
        self.video_path = video_path
        self.output_dir = output_dir
        self.playback_speed = playback_speed
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/frames", exist_ok=True)
        os.makedirs(f"{output_dir}/batsman", exist_ok=True)
        os.makedirs(f"{output_dir}/legs", exist_ok=True)
        
        # Load YOLOv8 model for person detection
        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')  # Using the nano model for speed
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Data storage for coordinates
        self.coordinates_data = {
            'batsman': [],
            'pads': []
        }
        
        print(f"Video loaded: {self.width}x{self.height}, {self.fps} FPS, {self.frame_count} frames")
        print(f"Playback speed: {playback_speed}x (slow motion)")
    
    def process_video(self, frame_interval=1, resize_dim=None, save_frames=True):
        
        
        # Calculate the delay between frames based on fps and playback speed
        frame_delay = int((1000 / self.fps) / self.playback_speed)
        
        # Create display windows
        cv2.namedWindow("Cricket Video Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cricket Video Analysis", 1280, 720)
        cv2.namedWindow("Batsman", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Leg Segmentation", cv2.WINDOW_NORMAL)
        
        frame_idx = 0
        processed_frames = 0
        

        start_time = time.time()
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break
                

            if frame_idx % frame_interval == 0:

                processed_frame = self.preprocess_frame(frame, resize_dim)

                result_frame, batsman_crop, leg_segment = self.process_single_frame(processed_frame, frame_idx)
                

                cv2.imshow("Cricket Video Analysis", result_frame)
                

                if batsman_crop is not None and batsman_crop.size > 0:
                    cv2.imshow("Batsman", batsman_crop)
                
                if leg_segment is not None and leg_segment.size > 0:
                    cv2.imshow("Leg Segmentation", leg_segment)
                
                if save_frames:
                    if result_frame is not None:
                        cv2.imwrite(f"{self.output_dir}/frames/frame_{frame_idx:05d}.jpg", result_frame)
                    if batsman_crop is not None and batsman_crop.size > 0:
                        cv2.imwrite(f"{self.output_dir}/batsman/batsman_{frame_idx:05d}.jpg", batsman_crop)
                    if leg_segment is not None and leg_segment.size > 0:
                        cv2.imwrite(f"{self.output_dir}/legs/legs_{frame_idx:05d}.jpg", leg_segment)
                
                processed_frames += 1
                
                if processed_frames % 10 == 0:
                    elapsed_time = time.time() - start_time
                    fps = processed_frames / elapsed_time
                    print(f"Processing frame {frame_idx}/{self.frame_count}, FPS: {fps:.2f}")
                
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Toggle slower/faster playback
                    self.playback_speed = max(0.1, min(2.0, self.playback_speed * 0.5 if self.playback_speed > 0.2 else self.playback_speed * 2))
                    frame_delay = int((1000 / self.fps) / self.playback_speed)
                    print(f"Playback speed changed to {self.playback_speed}x")
            
            frame_idx += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Save coordinates to JSON files
        with open(f"{self.output_dir}/batsman_coordinates.json", 'w') as f:
            json.dump(self.coordinates_data['batsman'], f, indent=4)
            
        with open(f"{self.output_dir}/pad_coordinates.json", 'w') as f:
            json.dump(self.coordinates_data['pads'], f, indent=4)
        
        print(f"Video processing complete! Results saved to '{self.output_dir}' directory")
        print(f"Processed {processed_frames} frames with {len(self.coordinates_data['batsman'])} batsman detections")
        print(f"Detected {len(self.coordinates_data['pads'])} pad instances")
    
    def preprocess_frame(self, frame, resize_dim=None):
        

        if resize_dim:
            frame = cv2.resize(frame, resize_dim)
        
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        
        return blurred
    
    def process_single_frame(self, frame, frame_idx):
        
        # Create copies for processing
        orig_frame = frame.copy()
        annotated_frame = frame.copy()
        batsman_crop = None
        leg_segment = None
        
        batsman_region = (self.width // 2, 0, self.width, self.height)  # (x_start, y_start, x_end, y_end)
        
        detections = self.model.predict(frame, classes=0, conf=0.5)  # Class 0 is person in COCO dataset
        
        if len(detections[0].boxes.data) > 0:
            boxes = detections[0].boxes.data.cpu().numpy()
            
            # Filter for people in the batsman region (typically right side of frame)
            batsman_candidates = []
            for box in boxes:
                x1, y1, x2, y2, conf, class_id = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center of the box
                center_x = (x1 + x2) / 2
                
                # Check if the person is in the batsman region
                if center_x > batsman_region[0]:
                    batsman_candidates.append(box)
            
            # If we found candidates in the batsman region
            if batsman_candidates:
                batsman_candidates = sorted(batsman_candidates, key=lambda x: x[4], reverse=True)
                
                batsman_box = batsman_candidates[0]
                x1, y1, x2, y2, conf, class_id = batsman_box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                
                batsman_data = {
                    'frame_idx': frame_idx,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    'width': int(x2 - x1),
                    'height': int(y2 - y1),
                    'confidence': float(conf)
                }
                self.coordinates_data['batsman'].append(batsman_data)
                
                # Focus more on the batsman by slightly tightening the bounding box
                width = x2 - x1
                new_x1 = int(x1 + width * 0.1)
                new_x2 = int(x2 - width * 0.1)
                
                # Extrat batsman
                batsman_crop = orig_frame[y1:y2, new_x1:new_x2].copy() if y1 < y2 and new_x1 < new_x2 else None
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Batsman {conf:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Segment legs 
                batsman_height = y2 - y1
                leg_start = y1 + int(batsman_height * 0.6)  # Start at 60% down from top
                
                # Extract leg region
                leg_frame = orig_frame[leg_start:y2, new_x1:new_x2].copy() if leg_start < y2 and new_x1 < new_x2 else None
                
                if leg_frame is not None and leg_frame.size > 0:
                    leg_segment, pad_contours = self.segment_legs(leg_frame)
                    
                    # Process pad contours to get coordinates in the original frame
                    if pad_contours:
                        for j, contour in enumerate(pad_contours):
                            # Adjust contour coordinates to match original frame
                            adjusted_contour = contour.copy()
                            adjusted_contour[:, :, 0] += new_x1  # Adjust x coordinate
                            adjusted_contour[:, :, 1] += leg_start  # Adjust y coordinate
                            
                            # Get bounding box for the pad
                            pad_x, pad_y, pad_w, pad_h = cv2.boundingRect(adjusted_contour)
                            
                            # Store pad coordinates
                            pad_data = {
                                'frame_idx': frame_idx,
                                'pad_idx': j,
                                'bbox': [pad_x, pad_y, pad_x + pad_w, pad_y + pad_h],
                                'center': [pad_x + pad_w // 2, pad_y + pad_h // 2],
                                'width': pad_w,
                                'height': pad_h,
                                'contour': adjusted_contour.tolist()  # Convert to list for JSON serialization
                            }
                            self.coordinates_data['pads'].append(pad_data)
                            
                            # Draw pad bounding box on the annotated frame
                            cv2.rectangle(annotated_frame, (pad_x, pad_y), 
                                         (pad_x + pad_w, pad_y + pad_h), (0, 255, 255), 2)
                            cv2.putText(annotated_frame, f"Pad {j+1}", (pad_x, pad_y-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Draw leg area on annotated frame
                    cv2.rectangle(annotated_frame, (new_x1, leg_start), (new_x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, "Legs", (new_x1, leg_start-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add frame information overlay
        cv2.putText(annotated_frame, f"Frame: {frame_idx}/{self.frame_count} | Speed: {self.playback_speed}x", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "Q to quit, S to change speed", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame, batsman_crop, leg_segment
    
    def segment_legs(self, leg_frame):
       
        if leg_frame is None or leg_frame.size == 0:
            return np.array([]), []
            
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(leg_frame, cv2.COLOR_BGR2HSV)
        
        
        
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 30, 255])
        

        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        

        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        

        combined_mask = cv2.bitwise_or(white_mask, blue_mask)
        
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        

        highlighted_legs = cv2.bitwise_and(leg_frame, leg_frame, mask=combined_mask)
        

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  
                significant_contours.append(contour)
        

        result = leg_frame.copy()
        cv2.drawContours(result, significant_contours, -1, (0, 255, 0), 2)
        
    
        vis_frame = np.hstack((leg_frame, highlighted_legs))
        
        return vis_frame, significant_contours


if __name__ == "__main__":
    video_path = "Videos/crick03.mp4" 
    
    
    processor = CricketVideoProcessor(video_path, playback_speed=0.5)
    
    processor.process_video(frame_interval=2)  # Process every second frame for better performance