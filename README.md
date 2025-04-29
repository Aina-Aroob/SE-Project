# LBW Prediction System

A physics-based system for predicting Leg Before Wicket (LBW) decisions in cricket using ball trajectory analysis. This system simulates ball physics including air resistance, Magnus effect, and stump collision detection to provide accurate LBW predictions.

## Features

- Accurate ball trajectory prediction using physics simulation
- Stump collision detection
- Impact region analysis
- Confidence scoring for predictions
- Simple command-line interface
- Comprehensive test suite for physics calculations

## Project Structure

```
.
├── main.py              # Main entry point
├── lbw_predictor.py     # Core prediction logic
├── physics.py          # Physics simulation module
├── config.py           # Configuration settings
├── tests/              # Test suite
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Requirements

- Python 3.8+
- Required Python packages:
  - numpy==2.2.5
  - pydantic==2.11.3
  - pytest==8.3.5 (for running tests)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:

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
    .
    .
    .
    .
    .
    .
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

## Running Tests

The project includes a comprehensive test suite that verifies the physics calculations and prediction logic. To run the tests:

```bash
pytest tests/
```

### Test Categories

The test suite covers various aspects of the system:

1. **Physics Calculations**

   - Drag force calculations
   - Magnus effect
   - Acceleration under gravity
   - Terminal velocity behavior

2. **Trajectory Prediction**

   - Projectile motion
   - Air resistance effects
   - Spin effects
   - Boundary conditions

3. **Edge Cases**
   - Zero velocity
   - High spin rates
   - Long duration trajectories
   - Various launch angles

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

## License

[Add your license information here]

## Contact

[Add your contact information here]

###test cases and their description
test_calculate_drag_force – Verifies the drag force magnitude and direction for a given velocity.

test_zero_velocity_edge_case – Ensures drag force is zero when velocity is zero.

test_calculate_acceleration – Checks acceleration under gravity and drag with no horizontal motion.

test_negative_position_trajectory – Confirms that trajectory progresses from negative positions.

test_predict_trajectory_no_air_resistance – Validates projectile motion when air resistance is disabled.

test_magnus_effect_direction – Ensures Magnus force acts orthogonally as expected with spin.

test_terminal_velocity_convergence – Checks that a ball's velocity stabilizes due to drag over time.

test_high_spin_magnus_effect – Verifies that high spin rates produce stronger Magnus forces.

test_low_angle_trajectory – Tests trajectory shape when launched at a low angle.

test_high_angle_trajectory – Tests trajectory shape when launched at a high angle.

test_drag_force_increases_with_velocity – Confirms that drag force increases with velocity magnitude.

test_drag_force_opposite_to_velocity – Ensures drag direction always opposes velocity.

test_acceleration_under_gravity_only – Validates pure gravitational acceleration when drag and Magnus are zero.

test_acceleration_with_drag_and_magnus – Tests acceleration under combined effects of drag and Magnus forces.

test_spin_affects_acceleration – Confirms that spin contributes to total acceleration via Magnus effect.

test_constant_spin_over_time – Checks that spin remains constant during trajectory prediction.

test_high_initial_velocity_behavior – Ensures system handles large initial velocities correctly.

test_long_duration_trajectory – Validates accuracy of long-duration trajectory predictions.

test_trajectory_lands_on_ground – Ensures that projectile motion eventually returns to ground level.

test_projectile_motion_with_air_resistance – Compares realistic projectile motion against analytical solutions with drag.

test_trajectory_timestamp_progression – Verifies that timestamps in the trajectory are increasing and consistent.

test_trajectory_continuity – Confirms that each step in trajectory smoothly follows the previous one.

test_predict_trajectory_start_equals_input_state – Ensures the first state in predicted trajectory matches the input state.

test_zero_velocity_results_in_stationary_trajectory – Checks that zero velocity produces a stationary trajectory.

test_trajectory_contains_correct_number_of_steps – Validates the number of trajectory points matches expectations based on time step.
