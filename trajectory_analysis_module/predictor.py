import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, validator, Field

from .physics import BallPhysics, BallState
from .config import default_config
from .utils import convert_position_inches_to_meters, convert_velocity_inches_to_meters


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
    impact_point: Dict[str, float]
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
    """Main class for LBW prediction system."""
    
    def __init__(self, config=None):
        """Initialize the LBW predictor with optional custom configuration."""
        self.config = config or default_config
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
        predicted_path = []
        for step_name, coords in input_data.new_trajectory_steps.items():
            coords_meters = convert_position_inches_to_meters(np.array(coords))
            step_num = int(step_name.split()[1])
            predicted_path.append(
                TrajectoryPoint(
                    pos_x=coords_meters[0],
                    pos_y=coords_meters[1],
                    pos_z=coords_meters[2],
                    timestamp=step_num * 0.1
                )
            )
        return predicted_path
   
    def check_stump_collision(
        self, predicted_path: List[TrajectoryPoint], stump_corners: List[List[float]]
    ) -> Tuple[bool, Dict[str, float], float]:
        """Check if the predicted path would hit the stumps."""
        stump_corners_meters = [convert_position_inches_to_meters(np.array(corner)) for corner in stump_corners]
        
        stump_x_min = min(corner[0] for corner in stump_corners_meters)
        stump_x_max = max(corner[0] for corner in stump_corners_meters)
        stump_y_min = min(corner[1] for corner in stump_corners_meters)
        stump_y_max = max(corner[1] for corner in stump_corners_meters)
        stump_z_min = min(corner[2] for corner in stump_corners_meters)
        stump_z_max = max(corner[2] for corner in stump_corners_meters) + 1.0

        for point in predicted_path:
            if (
                stump_x_min <= point.pos_x <= stump_x_max
                and stump_y_min <= point.pos_y <= stump_y_max
                and stump_z_min <= point.pos_z <= stump_z_max
            ):
                impact_point = {
                    "x": point.pos_x,
                    "y": point.pos_y,
                    "z": point.pos_z,
                    "relative_height": (point.pos_y - stump_y_min) / (stump_y_max - stump_y_min),
                    "relative_width": (point.pos_x - stump_x_min) / (stump_x_max - stump_x_min)
                }

                confidence = self._calculate_confidence(predicted_path)
                return True, impact_point, confidence

        return False, {"x": 0, "y": 0, "z": 0, "relative_height": 0, "relative_width": 0}, 0.0

    def _calculate_confidence(self, trajectory: List[TrajectoryPoint]) -> float:
        """Calculate confidence based on trajectory stability."""
        velocities = []
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            dx = curr.pos_x - prev.pos_x
            dy = curr.pos_y - prev.pos_y
            dz = curr.pos_z - prev.pos_z
            dt = curr.timestamp - prev.timestamp
            if dt > 0:
                velocities.append([dx / dt, dy / dt, dz / dt])

        if not velocities:
            return 0.0

        velocity_changes = [
            sum((v2[i] - v1[i])**2 for i in range(3))**0.5 
            for v1, v2 in zip(velocities[:-1], velocities[1:])
        ]
        stability = 1.0 - sum(velocity_changes) / sum(
            sum(v[i]**2 for i in range(3))**0.5 for v in velocities
        )

        if stability > self.config.confidence_thresholds["high"]:
            return 0.9
        elif stability > self.config.confidence_thresholds["medium"]:
            return 0.7
        elif stability > self.config.confidence_thresholds["low"]:
            return 0.5
        else:
            return 0.3

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return prediction results.
        
        Args:
            input_data: Dictionary containing the input data in the format specified in README.md
            
        Returns:
            Dictionary containing the prediction results including trajectory, verdict, and characteristics
        """
        validated_input = InputData(**input_data)
        
        predicted_path = self.predict_path(validated_input)
        
        hit, impact_point, confidence = self.check_stump_collision(
            predicted_path, 
            validated_input.stumps.corners
        )

        bounce_point = None
        for i in range(len(predicted_path) - 1):
            current = predicted_path[i]
            next_point = predicted_path[i + 1]
            if current.pos_y > 0 and next_point.pos_y <= 0:
                t = current.pos_y / (current.pos_y - next_point.pos_y)
                bounce_x = current.pos_x + t * (next_point.pos_x - current.pos_x)
                bounce_z = current.pos_z + t * (next_point.pos_z - current.pos_z)
                bounce_point = TrajectoryPoint(
                    pos_x=bounce_x,
                    pos_y=0.0,
                    pos_z=bounce_z,
                    timestamp=current.timestamp + t * (next_point.timestamp - current.timestamp)
                )
                break

        swing_characteristics = self._calculate_swing_characteristics(predicted_path)
        
        output = OutputData(
            result_id="res1",
            predicted_path=predicted_path,
            verdict=Verdict(
                status="Out" if hit else "Not Out",
                will_hit_stumps=hit,
                impact_point=impact_point,
                confidence=confidence
            ),
            bounce_point=bounce_point,
            swing_characteristics=swing_characteristics
        )
        
        return output.dict()

    def _calculate_swing_characteristics(self, predicted_path: List[TrajectoryPoint]) -> Dict[str, float]:
        """Calculate swing characteristics from the predicted path."""
        if len(predicted_path) < 2:
            return {
                "swing_angle": 0.0,
                "swing_magnitude": 0.0,
                "swing_direction": 0.0
            }

        initial_pos = np.array([predicted_path[0].pos_x, predicted_path[0].pos_y, predicted_path[0].pos_z])
        final_pos = np.array([predicted_path[-1].pos_x, predicted_path[-1].pos_y, predicted_path[-1].pos_z])
        
        initial_direction = np.array([
            predicted_path[1].pos_x - predicted_path[0].pos_x,
            predicted_path[1].pos_y - predicted_path[0].pos_y,
            predicted_path[1].pos_z - predicted_path[0].pos_z
        ])
        initial_direction = initial_direction / np.linalg.norm(initial_direction)
        
        final_direction = np.array([
            predicted_path[-1].pos_x - predicted_path[-2].pos_x,
            predicted_path[-1].pos_y - predicted_path[-2].pos_y,
            predicted_path[-1].pos_z - predicted_path[-2].pos_z
        ])
        final_direction = final_direction / np.linalg.norm(final_direction)
        
        swing_angle = np.arccos(np.clip(np.dot(initial_direction, final_direction), -1.0, 1.0))
        swing_angle_degrees = np.degrees(swing_angle)
        
        lateral_movement = np.linalg.norm(np.cross(final_pos - initial_pos, initial_direction))
        
        cross_product = np.cross(initial_direction, final_direction)
        swing_direction = np.sign(cross_product[1])
        
        return {
            "swing_angle": float(swing_angle_degrees),
            "swing_magnitude": float(lateral_movement),
            "swing_direction": float(swing_direction)
        } 