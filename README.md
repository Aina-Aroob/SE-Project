# Cricket LBW Prediction System

A sophisticated system for predicting Leg Before Wicket (LBW) decisions in cricket matches using physics-based modeling and trajectory analysis.

## Features

- Accurate ball trajectory prediction using physics-based modeling
- Surface condition consideration (dry, damp, green tops)
- Player movement analysis integration
- Real-time decision making with confidence levels
- Comprehensive physics simulation including:
  - Drag forces
  - Magnus effect (spin)
  - Bounce characteristics
  - Impact energy calculations

## System Components

1. **LBW Predictor** (`lbw_predictor.py`)

   - Main prediction engine
   - Handles input/output data processing
   - Makes final LBW decisions

2. **Physics Engine** (`physics.py`)

   - Detailed ball trajectory calculations
   - Force calculations (drag, magnus, gravity)
   - Impact and bounce modeling

3. **Configuration** (`config.py`)
   - System parameters and constants
   - Pitch and ball dimensions
   - Physics constants
   - Surface properties

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```python
from lbw_predictor import LBWPredictor

# Initialize predictor
predictor = LBWPredictor()

# Prepare input data
input_data = {
    "trajectory": [
        {"pos_x": 2.3, "pos_y": 1.1, "pos_z": 0.5, "timestamp": 0.0},
        {"pos_x": 2.1, "pos_y": 0.9, "pos_z": 0.4, "timestamp": 0.1}
    ],
    "velocity_vector": [1.1, -0.7, 0.4],
    "leg_contact_position": {"x": 2.0, "y": 0.7, "z": 0.35},
    "edge_detected": False,
    "decision_flag": [False, None]
}

# Get prediction
result = predictor.process_input(input_data)
print(result)
```

## Input Format

```json
{
    "trajectory": [
        {"pos_x": float, "pos_y": float, "pos_z": float, "timestamp": float},
        ...
    ],
    "velocity_vector": [float, float, float],
    "leg_contact_position": {"x": float, "y": float, "z": float},
    "edge_detected": bool,
    "decision_flag": [bool, null]
}
```

## Output Format

```json
{
    "result_id": "string",
    "predicted_path": [
        {"pos_x": float, "pos_y": float, "pos_z": float, "timestamp": float},
        ...
    ],
    "verdict": {
        "status": "string",
        "will_hit_stumps": bool,
        "impact_region": "string",
        "confidence": float
    }
}
```

## Physics Modeling

The system uses sophisticated physics modeling including:

- Runge-Kutta 4th order integration for trajectory prediction
- Drag force calculations based on ball velocity and air density
- Magnus effect for spin bowling
- Surface-specific bounce coefficients
- Energy conservation during impacts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
