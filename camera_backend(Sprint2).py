import cv2
import threading
import time
import os

class CameraManager:
    def __init__(self):
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