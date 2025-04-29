import cv2
import threading
import time
import os

class CameraManager:
    def _init_(self):
        self.cap = None
        self.is_streaming = False
        self.is_paused = False
        self.lock = threading.Lock()
        self.frames=[]
    def initialize_camera(self, resolution='720p', fps=30):
        resolutions = {
            '480p': (640, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080)
        }
        width, height = resolutions.get(resolution, (1280, 720))

        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)

                if not self.cap.isOpened():
                    raise RuntimeError("Could not open camera")
                print(f"Camera initialized at {width}x{height} @ {fps} FPS")
        return self.cap

    def generate_frames(self):
        while self.is_streaming:
            if self.is_paused:
                continue  # Skip frame generation while paused

            with self.lock:
                if self.cap is None:
                    break
                ret, frame = self.cap.read()

            if not ret:
                print("Failed to capture frame")
                break
            self.frames.append(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

    def start_streaming(self):
        if not self.is_streaming:
            self.is_streaming = True
            self.is_paused = False
            self.frames = []  
            print("Camera streaming started")

    def pause_streaming(self):
        self.is_paused = True
        print("Camera streaming paused")

    def resume_streaming(self):
        if self.is_streaming:
            self.is_paused = False
            print("Camera streaming resumed")

    def stop_streaming(self):
        with self.lock:
            self.is_streaming = False
            self.is_paused = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            self.save_video()
            self.frames = []      
            print("Camera resources released")
    def save_video(self):
        if not self.frames:
            print("No frames captured, skipping save.")
            return

        save_dir = "recordings"
        os.makedirs(save_dir, exist_ok=True)

        filename = time.strftime("%Y%m%d-%H%M%S") + ".avi"
        filepath = os.path.join(save_dir, filename)

        # Assuming all frames are same size
        height, width, _ = self.frames[0].shape
        fps = 30  # You can adjust the fps here

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

        for frame in self.frames:
            out.write(frame)

        out.release()
        print(f"Video saved to: {filepath}")
print("camera_backend.py loaded and camera_manager created")
camera_manager = CameraManager()
