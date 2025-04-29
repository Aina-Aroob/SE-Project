from flask import Flask, Response
import webbrowser
import threading
import time
from camera_backend import camera_manager


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

@app.route('/start')
def start():
    camera_manager.start_streaming()
    return "Camera started"

@app.route('/pause')
def pause():
    camera_manager.pause_streaming()
    return "Camera paused"

@app.route('/resume')
def resume():
    camera_manager.resume_streaming()
    return "Camera resumed"

@app.route('/stop')
def stop():
    camera_manager.stop_streaming()
    return "Camera stopped"

if _name_ == '_main_':
    try:
        
        camera_manager.initialize_camera()

        # Open default web browser to the video stream after a short delay
       # threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000/video')).start()

        app.run(host='0.0.0.0', port=5000)
    finally:
        camera_manager.stop_streaming()