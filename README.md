# LBW Prediction System

A physics-based system for predicting Leg Before Wicket (LBW) decisions in cricket using ball trajectory analysis.

## Features

- Accurate ball trajectory prediction using physics simulation
- Stump collision detection
- Impact region analysis
- Confidence scoring for predictions
- Simple command-line interface

## Requirements

- Python 3.8+
- Required Python packages:
  - numpy
  - pydantic

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the predictor with an input JSON file:

```bash
python main.py input.json
```

### With Output File

Save the prediction to a file:

```bash
python main.py input.json --output prediction.json
```

### Input Format

The input JSON file should follow this structure:

```json
{
  "trajectory": [
    { "pos_x": 2.3, "pos_y": 1.1, "pos_z": 0.5, "timestamp": 0.0 },
    { "pos_x": 2.1, "pos_y": 0.9, "pos_z": 0.4, "timestamp": 0.1 }
  ],
  "velocity_vector": [1.1, -0.7, 0.4],
  "leg_contact_position": { "x": 2.0, "y": 0.7, "z": 0.35 },
  "edge_detected": false,
  "decision_flag": [false, null]
}
```

### Output Format

The system returns a JSON response with:

- Initial and final ball positions
- LBW verdict (Out/Not Out)
- Impact region
- Confidence score

Example output:

```json
{
  "result_id": "res1",
  "predicted_path": [
    {
      "pos_x": 2.0,
      "pos_y": 0.7,
      "pos_z": 0.35,
      "timestamp": 0.0
    },
    {
      "pos_x": 2.7,
      "pos_y": -4.55,
      "pos_z": 0.6,
      "timestamp": 1.0
    }
  ],
  "verdict": {
    "status": "Not Out",
    "will_hit_stumps": false,
    "impact_region": "miss",
    "confidence": 0.0
  }
}
```
