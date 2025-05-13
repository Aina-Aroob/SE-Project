from flask import Flask, Response, request
from camera_backend import camera_manager
import time

app = Flask(_name_)

@app.route('/video')
def video():
    def generate():
        try:
            camera_manager.start_streaming()
            for frame in camera_manager.generate_frames():
                yield frame
        finally:
            # This runs when client disconnects
            camera_manager.stop_streaming()
            print("Client disconnected - camera stopped")

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )