class CameraManager:
    def __init__(self):
        """
        Initializes the CameraManager object with default settings.
        Sets up lock, frame storage, and streaming flags.
        """
        self.cap = None
        self.is_streaming = False
        self.is_paused = False
        self.lock = threading.Lock()
        self.frames = []

    def initialize_camera(self, resolution='720p', fps=30):
        """
        Initializes the camera with given resolution and fps.
        Opens the camera and sets its properties.
        """
        import cv2

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
        """
        Continuously captures frames from the camera while streaming is active.
        Yields encoded JPEG frames (for web streaming or display).
        Stores frames for saving video later.
        """
        import cv2
        import time

        while self.is_streaming:
            if self.is_paused:
                time.sleep(0.1)  # Avoid high CPU usage when paused
                continue

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
        """
        Starts the camera streaming.
        Resets pause flag and clears previous frames.
        """
        if not self.is_streaming:
            self.is_streaming = True
            self.is_paused = False
            self.frames = []
            print("Camera streaming started")

    def pause_streaming(self):
        """
        Pauses the camera streaming.
        Streaming loop will skip capturing frames while paused.
        """
        self.is_paused = True
        print("Camera streaming paused")

    def resume_streaming(self):
        """
        Resumes the camera streaming if it was paused.
        """
        if self.is_streaming:
            self.is_paused = False
            print("Camera streaming resumed")

    def stop_streaming(self):
        """
        Stops the camera streaming.
        Releases camera resource, saves captured frames to video file.
        """
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
        """
        Saves the captured frames into a video file (.avi format).
        Stores video inside a 'recordings' folder with timestamped filename.
        """
        import cv2
        import os
        import time

        if not self.frames:
            print("No frames captured, skipping save.")
            return

        save_dir = "recordings"
        os.makedirs(save_dir, exist_ok=True)

        filename = time.strftime("%Y%m%d-%H%M%S") + ".avi"
        filepath = os.path.join(save_dir, filename)

        # Assuming all frames are same size
        height, width, _ = self.frames[0].shape
        fps = 30  # Adjust FPS here

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

        for frame in self.frames:
            out.write(frame)

        out.release()
        print(f"Video saved to: {filepath}")

# Inform that the module loaded
print("camera_backend.py loaded and camera_manager created")

# Create CameraManager instance
camera_manager = CameraManager()
