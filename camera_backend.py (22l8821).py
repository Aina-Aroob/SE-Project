import cv2
import threading

class CameraManager:
    def __init__(self):
        self.cap = None
        self.is_streaming = False
        self.lock = threading.Lock()

    def generate_frames(self):
        """Generator function for video streaming"""
        while self.is_streaming:
            with self.lock:
                if self.cap is None:
                    break
                ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to capture frame")
                break

            # Process frame if needed (e.g., face detection)
            # frame = self._process_frame(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   buffer.tobytes() + b'\r\n')

    def start_streaming(self):
        """Start the video stream"""
        if not self.is_streaming:
            self.is_streaming = True
            print("Camera streaming started")

    def stop_streaming(self):
        """Stop the video stream and release resources"""
        with self.lock:
            self.is_streaming = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            print("Camera resources released")


# Singleton instance
camera_manager = CameraManager()