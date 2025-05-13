# Cricket Object Tracking

## Overview

This project provides a suite of tools for detecting and tracking key objects in cricket videos. It identifies the positions of wickets, the ball, the batsman's legs (approximated via pose estimation), and the bat, outputting their estimated 3D coordinates frame-by-frame into a combined JSON file. It also determines the batsman's orientation (Left/Right handed stance).

## Features

*   **Multi-Object Detection:** Detects and tracks:
    *   Cricket Wickets (Stumps)
    *   Cricket Ball
    *   Batsman's Legs (using pose estimation)
    *   Cricket Bat
*   **3D Coordinate Estimation:** Estimates the `[x, y, z]` coordinates for detected objects (corners for stumps, legs, bat; center for ball). The Z-coordinate represents estimated distance in meters.
*   **Batsman Orientation:** Determines if the batsman has a Left ('L') or Right ('R') handed stance based on shoulder/hip positions ('U' if undetermined).
*   **Combined Output:** Merges data from all detectors into a single JSON file per video, structured by frame.
*   **Modular Design:** Individual trackers are implemented in separate Python files within the `ballAndObjectTrackingModule`.
*   **Importable Function:** Provides a main function `track_objects` for easy integration into other Python projects.

## Technology Used

*   Python 3.x
*   OpenCV (`opencv-python`)
*   MediaPipe
*   NumPy
*   MoviePy (for audio extraction)
*   OpenCV DNN Module (for YOLOv4 ball detection)

## Project Structure

```
SE-Project/
├── ballAndObjectTrackingModule/
│   ├── WicketDetection.py      # Wicket detector class
│   ├── detection.py            # Ball detector class (YOLO)
│   ├── legdetectionwork.py     # Leg/Pose detector class (MediaPipe)
│   ├── bat_tracking.py         # Bat detector class (MediaPipe + Heuristics)
│   ├── combined_tracker.py     # Main function to run all trackers & merge
│   ├── videosfortesting/       # Directory for sample videos (example)
│   └── ...                     # (Potentially YOLO model files downloaded here)
├── ballTrackingIntermediate/   # Temporary directory for intermediate JSONs (auto-cleaned)
├── final_tracking_output/      # Default directory for final merged JSON output
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Setup

1.  **Prerequisites:**
    *   Python 3.7+ recommended.
    *   `pip` (Python package installer).

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd SE-Project
    ```

3.  **Install Dependencies:**
    *   It's highly recommended to use a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Linux/macOS
        # venv\Scripts\activate  # On Windows
        ```
    *   Install the required packages:
        ```bash
        pip install -r requirements.txt
        ```
        *(Note: A `requirements.txt` file should be created with at least `opencv-python`, `mediapipe`, `numpy`, `moviepy`)*

4.  **Model Files:**
    *   The YOLOv4 weights (`yolov4.weights`) and config (`yolov4.cfg`) required for ball detection will be automatically downloaded by `detection.py` the first time it's run if they are not found in the project directory.

## Usage

The primary way to use this module is via the `track_objects` function in `combined_tracker.py`.

**1. From another Python Script:**

```python
import os
import sys

# Optional: Add the project root to sys.path if running from outside
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# Assuming your script is in the project root directory
from ballAndObjectTrackingModule.combined_tracker import track_objects

# --- Configuration ---
# Relative path from project root
video_file = os.path.join("ballAndObjectTrackingModule", "videosfortesting", "crick05.mp4") 
# Or provide an absolute path
# video_file = r"C:\path\to\your\video.mp4"

# Optional: Specify a different final output directory (relative to project root)
output_directory = "tracking_results"
# ---------------------

if not os.path.exists(video_file):
    print(f"Error: Video file not found: {video_file}")
else:
    print(f"Processing video: {video_file}")
    # Run the combined tracking
    final_json_path = track_objects(video_path=video_file, output_dir=output_directory)

    if final_json_path:
        print(f"\nTracking complete. Merged results saved to: {final_json_path}")
    else:
        print("\nTracking process failed.")

```

**2. From the Command Line:**

You can run the `combined_tracker.py` script directly. Make sure your terminal's working directory is the `ballAndObjectTrackingModule` directory.

```bash
# Navigate to the module directory
cd path/to/SE-Project/ballAndObjectTrackingModule

# Run with a video path relative to the module directory
python combined_tracker.py videosfortesting/crick05.mp4

# Specify a different output directory (relative to the project root)
python combined_tracker.py videosfortesting/crick05.mp4 --output ../my_combined_output

# Use an absolute video path
python combined_tracker.py /full/path/to/your/video.mp4
```

## Output Format

The `track_objects` function generates a single JSON file (e.g., `combined_tracking_crick05_YYYYMMDD_HHMMSS.json`) in the specified output directory (default: `final_tracking_output`).

The JSON file contains a top-level object with two keys: `audio_base64` and `frames`:

```json
{
  "audio_base64": "UklGRiQ...", // Base64 encoded string of the video's audio (MP3 format), or null if no audio/error.
  "frames": [
    {
      "frame_id": 0,
      "ball": {
        "center": [x, y, z_meters],
        "radius": radius_pixels
      },
      "bat": {
        "corners": [ [x, y, z], [x, y, z], [x, y, z], [x, y, z] ] // TL, TR, BR, BL
      },
      "leg": {
        "corners": [ [x, y, z], [x, y, z], [x, y, z], [x, y, z] ] // TL, TR, BR, BL of leg region bbox
      },
      "stumps": {
        "corners": [ [x, y, z], [x, y, z], [x, y, z], [x, y, z] ] // TL, TR, BR, BL of first detected stump
      },
      "batsman_orientation": "R" // "L", "R", or "U" (Unknown)
    },
    {
      "frame_id": 1,
      "ball": null, // Object not detected in this frame
      "bat": { ... },
      "leg": { ... },
      "stumps": null,
      "batsman_orientation": "R"
    }
    // ... more frames
  ]
}
```

*   The `frames` value is a list, where each element represents a single frame.
*   If an object (ball, bat, leg, stumps) is not detected in a specific frame, its value will be `null`.
*   Coordinates `[x, y]` are pixel values (origin top-left).
*   The `z` coordinate is the estimated distance in meters.

## Cleanup

The intermediate JSON files generated by individual trackers in the `ballTrackingIntermediate` directory are automatically deleted by the `track_objects` function upon successful completion and saving of the final merged file. 