import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from physics import BallPhysics, BallState
from config import default_config


class TrajectoryPoint(BaseModel):
    pos_x: float
    pos_y: float
    pos_z: float
    timestamp: float


class VelocityVector(BaseModel):
    x: float
    y: float
    z: float


class ContactPosition(BaseModel):
    x: float
    y: float
    z: float


class InputData(BaseModel):
    trajectory: List[TrajectoryPoint]
    velocity_vector: List[float]
    leg_contact_position: ContactPosition
    edge_detected: bool
    decision_flag: List[Optional[bool]]


class Verdict(BaseModel):
    status: str
    will_hit_stumps: bool
    impact_region: str
    confidence: float


class OutputData(BaseModel):
    result_id: str
    predicted_path: List[TrajectoryPoint]
    verdict: Verdict


class LBWPredictor:
    def __init__(self):
        self.config = default_config
        self.physics = BallPhysics()

        # Update physics engine parameters from config
        self.physics.gravity = self.config.physics_constants.gravity
        self.physics.air_density = self.config.physics_constants.air_density
        self.physics.ball_radius = self.config.ball_properties.radius
        self.physics.ball_mass = self.config.ball_properties.mass
        self.physics.drag_coefficient = self.config.ball_properties.drag_coefficient
        self.physics.magnus_coefficient = self.config.ball_properties.magnus_coefficient

    def predict_path(self, input_data: InputData) -> List[TrajectoryPoint]:
        """Predict the ball's path after pad impact using physics engine."""
        # Convert input data to numpy arrays for calculations
        contact_pos = np.array(
            [
                input_data.leg_contact_position.x,
                input_data.leg_contact_position.y,
                input_data.leg_contact_position.z,
            ]
        )

        # Get initial velocity and use default spin (zero)
        initial_velocity = np.array(input_data.velocity_vector)
        spin = np.zeros(3)  # Default to no spin

        # Calculate post-impact velocity using physics engine with default surface type
        impact_velocity = self.physics.estimate_bounce_velocity(
            initial_velocity, "normal"  # Default surface type
        )

        # Create initial ball state
        initial_state = BallState(
            position=contact_pos, velocity=impact_velocity, spin=spin, timestamp=0.0
        )

        # Predict trajectory using physics engine
        predicted_states = self.physics.predict_trajectory(
            initial_state,
            time_step=self.config.physics_constants.time_step,
            duration=self.config.physics_constants.simulation_duration,
        )

        # Convert physics states to trajectory points
        predicted_path = []
        for state in predicted_states:
            predicted_path.append(
                TrajectoryPoint(
                    pos_x=state.position[0],
                    pos_y=state.position[1],
                    pos_z=state.position[2],
                    timestamp=state.timestamp,
                )
            )

        return predicted_path

    def check_stump_collision(
        self, predicted_path: List[TrajectoryPoint]
    ) -> Tuple[bool, str, float]:
        """Check if the predicted path would hit the stumps."""
        stump_region = {
            "x": [0, self.config.stump_dimensions.width],
            "y": [
                0,
                self.config.stump_dimensions.height
                + self.config.stump_dimensions.bail_height,
            ],
            "z": [
                self.config.pitch_dimensions.length / 2
                - self.config.stump_dimensions.depth / 2,
                self.config.pitch_dimensions.length / 2
                + self.config.stump_dimensions.depth / 2,
            ],
        }

        for point in predicted_path:
            if (
                stump_region["x"][0] <= point.pos_x <= stump_region["x"][1]
                and stump_region["y"][0] <= point.pos_y <= stump_region["y"][1]
                and stump_region["z"][0] <= point.pos_z <= stump_region["z"][1]
            ):

                # Determine impact region
                if point.pos_y < self.config.stump_dimensions.height / 3:
                    region = "low"
                elif point.pos_y < 2 * self.config.stump_dimensions.height / 3:
                    region = "middle"
                else:
                    region = "high"

                # Calculate confidence based on trajectory stability
                confidence = self._calculate_confidence(predicted_path)

                return True, region, confidence

        return False, "miss", 0.0

    def _calculate_confidence(self, trajectory: List[TrajectoryPoint]) -> float:
        """Calculate confidence based on trajectory stability."""
        # Calculate velocity changes between points
        velocities = []
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            dx = curr.pos_x - prev.pos_x
            dy = curr.pos_y - prev.pos_y
            dz = curr.pos_z - prev.pos_z
            dt = curr.timestamp - prev.timestamp
            if dt > 0:
                velocities.append(np.array([dx / dt, dy / dt, dz / dt]))

        if not velocities:
            return 0.0

        # Calculate velocity stability
        velocity_changes = [
            np.linalg.norm(v2 - v1) for v1, v2 in zip(velocities[:-1], velocities[1:])
        ]
        stability = 1.0 - np.mean(velocity_changes) / np.mean(
            [np.linalg.norm(v) for v in velocities]
        )

        # Map stability to confidence using thresholds
        if stability > self.config.confidence_thresholds["high"]:
            return 0.9
        elif stability > self.config.confidence_thresholds["medium"]:
            return 0.7
        elif stability > self.config.confidence_thresholds["low"]:
            return 0.5
        else:
            return 0.3

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return prediction results."""
        # Validate and parse input data
        validated_input = InputData(**input_data)

        # Predict path
        predicted_path = self.predict_path(validated_input)

        # Check stump collision
        will_hit, impact_region, confidence = self.check_stump_collision(predicted_path)

        # Prepare output
        verdict = Verdict(
            status="Out" if will_hit else "Not Out",
            will_hit_stumps=will_hit,
            impact_region=impact_region,
            confidence=confidence,
        )

        output = OutputData(
            result_id="res1", predicted_path=predicted_path, verdict=verdict
        )

        return output.model_dump()
