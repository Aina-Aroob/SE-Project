'''
Small explanation of how APIs work: 

Communication happens between two hosts, a server and a client. 
The client sends a request to the server, which then responds to it. 
It is kind of like ordering food in a restaurant where you (client) place an order (request) and the waiter (server) brings you food (response.) 



How our API works: 
We are essentially creating a server that exposes an endpoint called "stream-overlay". 
Module 1 will send a request to it with the following data: 
1. Video file 
2. Bounce + Impact coordinates
3. Predicted trajectory coordinates + Decision
Our server will process the video file and return a new video output.mp4


Explanation of the code below:
1. We create a temporary Directory and use it to create paths for video_path, bounce_path, trajectory_path, and output_path. 
   These paths will be used to store the uploaded files and the final output video.
2. We save the files we have received in the request (from Mod 1) to these paths on the server (our) disk.
3. We create an object of TrajectoryOverlayRenderer and send the paths to it (which is what it expects)
4. We call the run() function 
5. We send a response with the augemented video which will get downloaded 
6. The temporary directly will automatically get deleted later so we won't be saving any unnecessary files
'''


import os
import tempfile
from flask import Flask, request, send_file
from stream_overlay import TrajectoryOverlayRenderer

app = Flask(__name__)

@app.route('/stream-overlay', methods=['POST'])
def augment_video():
    try:
        # Ensure required files are in the request
        if 'video' not in request.files or \
           'module5_output_json' not in request.files or \
           'module4_output_json' not in request.files:
            return "Missing required files", 400

        '''
        Use a temporary directory to store intermediate files
        '''
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, 'input.avi')
            bounce_path = os.path.join(temp_dir, 'module5_output.json')
            trajectory_path = os.path.join(temp_dir, 'module4_output.json')
            output_path = os.path.join(temp_dir, 'output.avi')

            # Save uploaded files
            request.files['video'].save(video_path)
            request.files['module5_output_json'].save(bounce_path)
            request.files['module4_output_json'].save(trajectory_path)

            # Run the overlay renderer
            renderer = TrajectoryOverlayRenderer(
                video_path=video_path,
                module4_json=trajectory_path, 
                module5_json=bounce_path,
                output_path=output_path,
                slow_factor=3
            )
            renderer.run()

            # Return the processed video
            return send_file(output_path, mimetype='video/x-msvideo', 
                             as_attachment=True, download_name='augmented_video.avi')
        # mimetype='video/x-msvideo' --> this specifies that our video is of type .avi

    except KeyError:
        return "Missing required files", 400
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
