import cv2
import os
import sys
import json
import time
from ultralytics import YOLO

class CricketVideoProcessor:
    def __init__(self, video_path, output_dir="output", playback_speed=0.5):
        self.video_path = video_path
        self.output_dir = output_dir
        self.playback_speed = playback_speed

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/frames", exist_ok=True)
        os.makedirs(f"{output_dir}/batsman", exist_ok=True)
        os.makedirs(f"{output_dir}/legs", exist_ok=True)

        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.coordinates_data = {'batsman': [], 'pads': []}
        print(f"Video loaded: {self.width}×{self.height}, {self.fps} FPS, {self.frame_count} frames")
        print(f"Playback speed: {playback_speed}x")

    def preprocess_frame(self, frame, resize_dim=None):
        """
        Resize the frame if resize_dim is provided; otherwise return it unchanged.
        """
        return cv2.resize(frame, resize_dim) if resize_dim else frame

    def process_video(self, frame_interval=1, resize_dim=None, save_frames=True):
        frame_delay = int((1000 / self.fps) / self.playback_speed)

        cv2.namedWindow("Cricket Video Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cricket Video Analysis", 1280, 720)
        cv2.namedWindow("Batsman", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Leg Segmentation", cv2.WINDOW_NORMAL)

        frame_idx = 0
        processed = 0
        start = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                proc_frame = self.preprocess_frame(frame, resize_dim)
                result_frame, batsman_crop, leg_seg = self.process_single_frame(proc_frame, frame_idx)

                cv2.imshow("Cricket Video Analysis", result_frame)
                if batsman_crop is not None and batsman_crop.size:
                    cv2.imshow("Batsman", batsman_crop)
                if leg_seg is not None and leg_seg.size:
                    cv2.imshow("Leg Segmentation", leg_seg)

                if save_frames:
                    cv2.imwrite(f"{self.output_dir}/frames/frame_{frame_idx:05d}.jpg", result_frame)
                    if batsman_crop is not None and batsman_crop.size:
                        cv2.imwrite(f"{self.output_dir}/batsman/batsman_{frame_idx:05d}.jpg", batsman_crop)
                    if leg_seg is not None and leg_seg.size:
                        cv2.imwrite(f"{self.output_dir}/legs/legs_{frame_idx:05d}.jpg", leg_seg)

                processed += 1
                if processed % 10 == 0:
                    elapsed = time.time() - start
                    print(f"Frame {frame_idx}/{self.frame_count} | FPS: {processed/elapsed:.2f}")

                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s'):
                    self.playback_speed = max(0.1, min(2.0,
                        self.playback_speed*0.5 if self.playback_speed>0.2 else self.playback_speed*2))
                    frame_delay = int((1000 / self.fps) / self.playback_speed)
                    print(f"Playback speed: {self.playback_speed}x")

            frame_idx += 1

        self.cap.release()
        cv2.destroyAllWindows()

        with open(f"{self.output_dir}/batsman_coordinates.json", 'w') as f:
            json.dump(self.coordinates_data['batsman'], f, indent=4)
        with open(f"{self.output_dir}/pad_coordinates.json", 'w') as f:
            json.dump(self.coordinates_data['pads'], f, indent=4)

        print(f"Done: {processed} frames, {len(self.coordinates_data['batsman'])} batsman, "
              f"{len(self.coordinates_data['pads'])} pads detected.")

    def process_single_frame(self, frame, frame_idx):
        # TODO: add detection logic here; return (frame, batsman_crop, leg_seg)
        return frame, None, None

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    else:
        video_path = "crick01 (1).mp4"
        print(f"No video path provided. Using default: {video_path}")

    processor = CricketVideoProcessor(video_path)
    processor.process_video(
        frame_interval=1,
        resize_dim=(640, 360),  # optional resize
        save_frames=True
    )
