# LBW Prediction System

A system for predicting Leg Before Wicket (LBW) decisions in cricket using physics-based trajectory prediction.

## Overview

This system analyzes the trajectory of a cricket ball after it hits the batsman's pad to determine if it would have hit the stumps. It uses physics-based calculations to predict the ball's path and includes features like bounce detection and swing analysis.

## Features

- **Trajectory Prediction**: Uses physics engine to predict ball path
- **Stump Collision Detection**: Determines if the ball would hit the stumps
- **Bounce Point Detection**: Identifies where the ball bounces
- **Swing Analysis**: Calculates swing angle, magnitude, and direction
- **Unit Conversion**: Handles conversion between inches and meters
- **Confidence Scoring**: Provides confidence level for predictions

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd lbw-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Tests

```bash
python test_lbw.py
```

This will run three test cases:

1. Direct Hit Test
2. Bounce Test
3. Edge Case Test

### Sample Output

Here's a sample output from a test case:

```json
{
  "result_id": "res1",
  "predicted_path": [
    {
      "pos_x": 18.796, // X position in meters
      "pos_y": 15.748, // Y position in meters
      "pos_z": 0.254, // Z position in meters
      "timestamp": 0.0 // Time in seconds
    }
    // ... more trajectory points
  ],
  "verdict": {
    "status": "Out", // "Out" or "Not Out"
    "will_hit_stumps": true, // Whether ball would hit stumps
    "impact_point": {
      "x": 19.05, // X coordinate of impact (meters)
      "y": 16.002, // Y coordinate of impact (meters)
      "z": 0.508, // Z coordinate of impact (meters)
      "relative_height": 0.62, // Height ratio relative to stumps (0-1)
      "relative_width": 0.0 // Width ratio relative to stumps (0-1)
    },
    "confidence": 0.9 // Confidence level (0-1)
  },
  "bounce_point": {
    // null if no bounce
    "pos_x": 19.304, // X position of bounce (meters)
    "pos_y": 0.0, // Y position of bounce (meters)
    "pos_z": 3.81, // Z position of bounce (meters)
    "timestamp": 1.0 // Time of bounce (seconds)
  },
  "swing_characteristics": {
    "swing_angle": 37.86, // Angle of swing in degrees
    "swing_magnitude": 3.91, // Magnitude of swing in meters
    "swing_direction": -1.0 // -1: Counter-clockwise, 1: Clockwise
  }
}
```

### Field Descriptions

#### Predicted Path

- `pos_x`, `pos_y`, `pos_z`: Ball position in meters
- `timestamp`: Time in seconds from start of trajectory

#### Verdict

- `status`: "Out" if ball would hit stumps, "Not Out" otherwise
- `will_hit_stumps`: Boolean indicating if ball would hit stumps
- `impact_point`: Location where ball would hit stumps
  - `x`, `y`, `z`: Coordinates in meters
  - `relative_height`: Height ratio (0 = bottom of stumps, 1 = top)
  - `relative_width`: Width ratio (0 = left edge, 1 = right edge)
- `confidence`: Confidence level in prediction (0-1)

#### Bounce Point

- `pos_x`, `pos_y`, `pos_z`: Position of bounce in meters
- `timestamp`: Time of bounce in seconds
- Note: This field is null if no bounce occurs

#### Swing Characteristics

- `swing_angle`: Deviation from straight line in degrees
- `swing_magnitude`: Lateral movement in meters
- `swing_direction`: -1 for counter-clockwise, 1 for clockwise

## Input Format

The system expects input in JSON format with the following structure:

```json
{
  "collision": {
    "collision": true,
    "distance": 8,
    "collision_point": [700, 620, 10],
    "bat_obb": {
      "center": [710, 625, 15],
      "basis": [
        [1, 0, 0],
        [0, 0.98, 0.19],
        [0, 0, 1]
      ],
      "half_size": [25, 50.99, 10]
    },
    "confidence": "high",
    "method": "spatial",
    "details": "Ball intersects bat by 3.00 units"
  },
  "trajectory": {
    "updated": true,
    "previous_velocity": [-2, 1, -1],
    "velocity": [15, -5, 20],
    "speed": 25.49,
    "direction": [0.588, -0.196, 0.784],
    "collision_point": [700, 620, 10],
    "normal": [0, 0, 1],
    "restitution_applied": 0.8,
    "friction_applied": 0.2,
    "spin_effect": [2, -1, 0],
    "details": "Trajectory updated based on collision physics"
  },
  "new_trajectory_steps": {
    "Step 0": [740, 620, 10],
    "Step 1": [742, 622, 12]
    // ... more steps
  },
  "stumps": {
    "corners": [
      [750, 580, 0],
      [770, 580, 0],
      [770, 660, 0],
      [750, 660, 0]
    ]
  }
}
```

## Output Files

Test results are saved in the `test_outputs` directory:

- `direct_hit_test_output.json`: Results for direct hit test
- `bounce_test_output.json`: Results for bounce test
- `edge_case_test_output.json`: Results for edge case test

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
