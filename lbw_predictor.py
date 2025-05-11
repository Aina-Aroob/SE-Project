import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, validator, Field
from physics import BallPhysics, BallState
from config import default_config
from utils import convert_position_inches_to_meters, convert_velocity_inches_to_meters


class TrajectoryPoint(BaseModel):
    pos_x: float = Field(..., description="X position in meters")
    pos_y: float = Field(..., description="Y position in meters")
    pos_z: float = Field(..., description="Z position in meters")
    timestamp: float = Field(..., description="Time in seconds", ge=0)

    @validator('pos_x', 'pos_y', 'pos_z')
    def validate_position(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Position must be a number")
        return float(v)


class VelocityVector(BaseModel):
    x: float = Field(..., description="X velocity component")
    y: float = Field(..., description="Y velocity component")
    z: float = Field(..., description="Z velocity component")


class ContactPosition(BaseModel):
    x: float = Field(..., description="X position of contact")
    y: float = Field(..., description="Y position of contact")
    z: float = Field(..., description="Z position of contact")


class CollisionInfo(BaseModel):
    collision: bool = Field(..., description="Whether collision occurred")
    distance: float = Field(..., description="Distance of collision", ge=0)
    collision_point: List[float] = Field(..., min_items=3, max_items=3)
    bat_obb: Dict[str, Any]
    confidence: str = Field(..., pattern="^(low|medium|high)$")
    method: str
    details: str

    @validator('collision_point')
    def validate_collision_point(cls, v):
        if len(v) != 3:
            raise ValueError("Collision point must have exactly 3 coordinates")
        return v


class TrajectoryInfo(BaseModel):
    updated: bool
    previous_velocity: List[float] = Field(..., min_items=3, max_items=3)
    velocity: List[float] = Field(..., min_items=3, max_items=3)
    speed: float = Field(..., ge=0)
    direction: List[float] = Field(..., min_items=3, max_items=3)
    collision_point: List[float] = Field(..., min_items=3, max_items=3)
    normal: List[float] = Field(..., min_items=3, max_items=3)
    restitution_applied: float = Field(..., ge=0, le=1)
    friction_applied: float = Field(..., ge=0, le=1)
    spin_effect: List[float] = Field(..., min_items=3, max_items=3)
    details: str


class StumpPosition(BaseModel):
    corners: List[List[float]] = Field(..., min_items=4, max_items=4)

    @validator('corners')
    def validate_corners(cls, v):
        if len(v) != 4:
            raise ValueError("Stumps must have exactly 4 corners")
        if not all(len(corner) == 3 for corner in v):
            raise ValueError("Each corner must have exactly 3 coordinates")
        return v


class InputData(BaseModel):
    collision: CollisionInfo
    trajectory: TrajectoryInfo
    new_trajectory_steps: Dict[str, List[float]]
    stumps: StumpPosition

    @validator('new_trajectory_steps')
    def validate_trajectory_steps(cls, v):
        for step, coords in v.items():
            if not step.startswith("Step "):
                raise ValueError(f"Invalid step format: {step}")
            if len(coords) != 3:
                raise ValueError(f"Invalid coordinates in step {step}")
        return v


class Verdict(BaseModel):
    status: str
    will_hit_stumps: bool
    impact_point: Dict[str, float]  # Changed from impact_region to impact_point
    confidence: float


class OutputData(BaseModel):
    result_id: str
    predicted_path: List[TrajectoryPoint]
    verdict: Verdict
    bounce_point: Optional[TrajectoryPoint] = None
    swing_characteristics: Dict[str, float] = {
        "swing_angle": 0.0,
        "swing_magnitude": 0.0,
        "swing_direction": 0.0
    }


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
        # Convert trajectory steps to TrajectoryPoint objects
        predicted_path = []
        for step_name, coords in input_data.new_trajectory_steps.items():
            # Convert coordinates from inches to meters for internal calculations
            coords_meters = convert_position_inches_to_meters(np.array(coords))
            step_num = int(step_name.split()[1])  # Extract step number from "Step X"
            predicted_path.append(
                TrajectoryPoint(
                    pos_x=coords_meters[0],  # Store in meters for physics calculations
                    pos_y=coords_meters[1],
                    pos_z=coords_meters[2],
                    timestamp=step_num * 0.1  # Assuming 0.1s time step
                )
            )
        
        # Print some trajectory points for debugging
        print("\nTrajectory points (meters):")
        for i, point in enumerate(predicted_path[:5]):
            print(f"Point {i}: ({point.pos_x:.3f}, {point.pos_y:.3f}, {point.pos_z:.3f})")
        
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
