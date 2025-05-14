# Cricket Ball Trajectory Predictor

This project provides functionality to predict cricket ball trajectories based on historical ball positions, including features like swing detection, bounce prediction, and impact analysis.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The trajectory predictor takes input from a `data.json` file and produces output in `trajectory_output.json`.

### Running the Predictor

```bash
python trajectory_predictor.py
```

### Input Format (`data.json`)

The input JSON file should have the following structure:

```json
{
    "frames": [
        {
            "frame_number": integer,
            "ball_position": [x, y, z],
            "timestamp": float
        },
        ...
    ],
    "collision": {  // Optional
        "spatial_detection": {
            "point": [x, y, z],
            "distance_from_bat_center": float
        },
        "audio_detection": {
            "detected": boolean
        }
    }
}
```

#### Required Fields:
- `frames`: Array of frame objects containing ball position data
  - `frame_number`: Sequential frame number
  - `ball_position`: 3D coordinates [x, y, z] of the ball
  - `timestamp`: Time of the frame in seconds

#### Optional Fields:
- `collision`: Object containing collision detection data
  - `spatial_detection`: Spatial collision information
  - `audio_detection`: Audio-based collision detection results

### Output Format (`trajectory_output.json`)

The output JSON file contains:

```json
{
    "previous_trajectory": [
        {
            "frame": integer,
            "position": [x, y, z],
            "timestamp": float
        },
        ...
    ],
    "predicted_trajectory": [
        {
            "frame": integer,
            "position": [x, y, z],
            "timestamp": float
        },
        ...
    ],
    "leg_impact_location": [x, y, z],  // If predicted to hit leg
    "swing_characteristics": {
        "swing_amount": float,
        "swing_direction": string,
        "lateral_deviation": float,
        "swing_type": string,
        "seam_position": string
    },
    "collision": {  // Only if collision data was provided in input
        "spatial_detection": {
            "point": [x, y, z],
            "distance_from_bat_center": float
        },
        "audio_detection": {
            "detected": boolean
        }
    }
}
```

#### Output Fields:
- `previous_trajectory`: Historical ball positions from input data
- `predicted_trajectory`: Predicted future ball positions
- `leg_impact_location`: Coordinates where ball is predicted to hit leg (if applicable)
- `swing_characteristics`: Analysis of ball swing behavior
  - `swing_amount`: Magnitude of swing
  - `swing_direction`: Direction of swing ("in" or "out")
  - `lateral_deviation`: Side-to-side movement
  - `swing_type`: Classification of swing behavior
  - `seam_position`: Position of the ball's seam
- `collision`: Collision data (included if provided in input)

## Features

- Ball trajectory prediction using physics-based modeling
- Swing analysis and characterization
- Bounce detection and prediction
- Impact prediction (stumps/legs)
- Collision detection integration
- Historical trajectory analysis

## Dependencies

The project requires the following Python packages:
- numpy
- scipy
- matplotlib

See `requirements.txt` for specific version requirements. 