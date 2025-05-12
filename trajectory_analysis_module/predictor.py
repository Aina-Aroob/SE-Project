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
        """Predict the ball's path after pad impact using enhanced physics engine.
        
        The prediction includes:
        - Initial velocity and spin from collision
        - Air resistance effects
        - Magnus effect (swing)
        - Bounce effects
        - Wind effects (if configured)
        - High-resolution trajectory (0.01s time steps)
        """
        predicted_path = []
        
        # Extract initial conditions from collision
        initial_velocity = np.array(input_data.trajectory.velocity)
        initial_spin = np.array(input_data.trajectory.spin_effect)
        initial_pos = np.array(input_data.trajectory.collision_point)
        
        # Convert to meters
        initial_velocity = convert_velocity_inches_to_meters(initial_velocity)
        initial_pos = convert_position_inches_to_meters(initial_pos)
        
        # Create initial ball state
        ball_state = BallState(
            position=initial_pos,
            velocity=initial_velocity,
            spin=initial_spin,
            timestamp=0.0
        )
        
        # Use finer time step for more detailed trajectory
        time_step = 0.01  # 10ms steps
        duration = self.config.physics_constants.simulation_duration
        
        # Predict trajectory using physics engine with fine time steps
        trajectory_states = self.physics.predict_trajectory_with_bounce(
            ball_state,
            time_step=time_step,
            duration=duration,
            max_bounces=1
        )
        
        # Convert states to trajectory points, avoiding duplicates
        last_timestamp = -1
        for state in trajectory_states:
            if state.timestamp > last_timestamp:
                predicted_path.append(
                    TrajectoryPoint(
                        pos_x=state.position[0],
                        pos_y=state.position[1],
                        pos_z=state.position[2],
                        timestamp=state.timestamp
                    )
                )
                last_timestamp = state.timestamp
        
        # If we have trajectory steps from input, use them to validate and adjust prediction
        if input_data.new_trajectory_steps:
            # Calculate average error between predicted and actual trajectory
            errors = []
            for step_name, coords in input_data.new_trajectory_steps.items():
                coords_meters = convert_position_inches_to_meters(np.array(coords))
                step_num = int(step_name.split()[1])
                timestamp = step_num * 0.1  # Original input uses 0.1s steps
                
                # Find closest predicted point
                closest_point = min(predicted_path, key=lambda p: abs(p.timestamp - timestamp))
                error = np.linalg.norm(np.array([
                    coords_meters[0] - closest_point.pos_x,
                    coords_meters[1] - closest_point.pos_y,
                    coords_meters[2] - closest_point.pos_z
                ]))
                errors.append(error)
            
            # If average error is too high, adjust prediction
            avg_error = sum(errors) / len(errors)
            if avg_error > 0.1:  # 10cm threshold
                # Create high-resolution adjusted path
                adjusted_path = []
                # Get the original points
                original_points = []
                for step_name, coords in input_data.new_trajectory_steps.items():
                    coords_meters = convert_position_inches_to_meters(np.array(coords))
                    step_num = int(step_name.split()[1])
                    original_points.append({
                        'pos': coords_meters,
                        'timestamp': step_num * 0.1
                    })
                
                # Sort points by timestamp
                original_points.sort(key=lambda x: x['timestamp'])
                
                # Interpolate between original points to get 0.01s resolution
                for i in range(len(original_points) - 1):
                    p1 = original_points[i]
                    p2 = original_points[i + 1]
                    
                    # Calculate number of points needed between these two points
                    time_diff = p2['timestamp'] - p1['timestamp']
                    num_points = int(time_diff / time_step)
                    
                    for j in range(num_points + 1):
                        t = j / num_points
                        pos = p1['pos'] * (1 - t) + p2['pos'] * t
                        timestamp = p1['timestamp'] + t * time_diff
                        
                        # Only add point if it's not a duplicate
                        if not adjusted_path or timestamp > adjusted_path[-1].timestamp:
                            adjusted_path.append(
                                TrajectoryPoint(
                                    pos_x=float(pos[0]),
                                    pos_y=float(pos[1]),
                                    pos_z=float(pos[2]),
                                    timestamp=timestamp
                                )
                            )
                
                return adjusted_path
        
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

                confidence = self._calculate_confidence(predicted_path, impact_point)
                return True, impact_point, confidence

        return False, {"x": 0, "y": 0, "z": 0, "relative_height": 0, "relative_width": 0}, 0.0

    def _calculate_confidence(self, trajectory: List[TrajectoryPoint], impact_point: Dict[str, float] = None) -> float:
        """Calculate confidence based on multiple factors:
        1. Distance to stumps
        2. Trajectory stability
        3. Swing characteristics
        4. Impact point relative position
        """
        # Initialize confidence components
        stability_confidence = 0.0
        impact_confidence = 0.0
        swing_confidence = 0.0

        # 1. Calculate trajectory stability confidence
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

        if velocities:
            velocity_changes = [
                sum((v2[i] - v1[i])**2 for i in range(3))**0.5 
                for v1, v2 in zip(velocities[:-1], velocities[1:])
            ]
            stability = 1.0 - sum(velocity_changes) / sum(
                sum(v[i]**2 for i in range(3))**0.5 for v in velocities
            )
            stability_confidence = min(1.0, max(0.0, stability))

        # 2. Calculate impact point confidence
        if impact_point:
            # Higher confidence for impacts closer to middle of stumps
            height_confidence = 1.0 - abs(impact_point.get("relative_height", 0.5) - 0.5) * 2
            width_confidence = 1.0 - abs(impact_point.get("relative_width", 0.5) - 0.5) * 2
            impact_confidence = (height_confidence + width_confidence) / 2

        # 3. Calculate swing confidence
        swing_chars = self._calculate_swing_characteristics(trajectory)
        # Lower confidence for extreme swing angles
        swing_angle = abs(swing_chars["swing_angle"])
        swing_confidence = 1.0 - min(1.0, swing_angle / 45.0)  # 45 degrees as threshold

        # Combine all confidence factors with weights
        weights = {
            "stability": 0.4,
            "impact": 0.4,
            "swing": 0.2
        }

        final_confidence = (
            stability_confidence * weights["stability"] +
            impact_confidence * weights["impact"] +
            swing_confidence * weights["swing"]
        )

        # Map to confidence levels
        if final_confidence > self.config.confidence_thresholds["high"]:
            return 0.9
        elif final_confidence > self.config.confidence_thresholds["medium"]:
            return 0.7
        elif final_confidence > self.config.confidence_thresholds["low"]:
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

    def _calculate_swing_characteristics(self, trajectory_points: List[TrajectoryPoint]) -> Dict[str, float]:
        """Calculate swing characteristics from trajectory points.
        
        Returns:
            Dict containing:
            - swing_angle: Angle of swing in degrees
            - swing_magnitude: Magnitude of swing in meters
            - swing_direction: Direction of swing (-1 for inswing, 1 for outswing)
            - swing_rate: Rate of swing in degrees per second
            - swing_consistency: How consistent the swing is (0-1)
            - late_swing: Amount of late swing in degrees
        """
        if len(trajectory_points) < 2:
            return {
                'swing_angle': 0.0,
                'swing_magnitude': 0.0,
                'swing_direction': 0.0,
                'swing_rate': 0.0,
                'swing_consistency': 1.0,
                'late_swing': 0.0
            }
        
        # Calculate initial and final positions
        initial_pos = np.array([
            trajectory_points[0].pos_x,
            trajectory_points[0].pos_y,
            trajectory_points[0].pos_z
        ])
        final_pos = np.array([
            trajectory_points[-1].pos_x,
            trajectory_points[-1].pos_y,
            trajectory_points[-1].pos_z
        ])
        
        # Calculate the main trajectory vector
        trajectory_vector = final_pos - initial_pos
        trajectory_length = np.linalg.norm(trajectory_vector)
        
        if trajectory_length == 0:
            return {
                'swing_angle': 0.0,
                'swing_magnitude': 0.0,
                'swing_direction': 0.0,
                'swing_rate': 0.0,
                'swing_consistency': 1.0,
                'late_swing': 0.0
            }
        
        # Calculate swing angle using the maximum lateral deviation
        lateral_positions = [p.pos_z for p in trajectory_points]
        max_lateral_deviation = max(abs(max(lateral_positions)), abs(min(lateral_positions)))
        
        # Calculate swing angle using arcsin of the ratio of lateral deviation to trajectory length
        swing_angle = np.degrees(np.arcsin(max_lateral_deviation / trajectory_length))
        
        # Determine swing direction based on the sign of the final lateral position
        swing_direction = 1.0 if final_pos[2] > initial_pos[2] else -1.0
        
        # Calculate swing magnitude as the maximum lateral deviation
        swing_magnitude = max_lateral_deviation
        
        # Calculate swing rate (degrees per second)
        time_diff = trajectory_points[-1].timestamp - trajectory_points[0].timestamp
        swing_rate = swing_angle / time_diff if time_diff > 0 else 0.0
        
        # Calculate swing consistency using a more sophisticated approach
        direction_changes = []
        lateral_velocities = []
        
        for i in range(len(trajectory_points) - 1):
            p1 = trajectory_points[i]
            p2 = trajectory_points[i + 1]
            dt = p2.timestamp - p1.timestamp
            
            if dt > 0:
                # Calculate lateral velocity
                lateral_velocity = (p2.pos_z - p1.pos_z) / dt
                lateral_velocities.append(lateral_velocity)
                
                # Calculate direction change
                direction = np.array([
                    p2.pos_x - p1.pos_x,
                    p2.pos_y - p1.pos_y,
                    p2.pos_z - p1.pos_z
                ])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    if i > 0:
                        prev_direction = np.array([
                            p1.pos_x - trajectory_points[i-1].pos_x,
                            p1.pos_y - trajectory_points[i-1].pos_y,
                            p1.pos_z - trajectory_points[i-1].pos_z
                        ])
                        if np.linalg.norm(prev_direction) > 0:
                            prev_direction = prev_direction / np.linalg.norm(prev_direction)
                            dot = np.clip(np.dot(direction, prev_direction), -1.0, 1.0)
                            angle = np.degrees(np.arccos(dot))
                            direction_changes.append(angle)
        
        # Calculate swing consistency based on both direction changes and lateral velocity consistency
        if direction_changes and lateral_velocities:
            # Calculate direction change consistency
            avg_direction_change = sum(direction_changes) / len(direction_changes)
            direction_consistency = 1.0 - min(avg_direction_change / 45.0, 1.0)
            
            # Calculate lateral velocity consistency
            lateral_velocity_std = np.std(lateral_velocities)
            lateral_velocity_mean = np.mean(lateral_velocities)
            velocity_consistency = 1.0 - min(lateral_velocity_std / abs(lateral_velocity_mean), 1.0)
            
            # Combine both consistency measures
            swing_consistency = (direction_consistency + velocity_consistency) / 2
        else:
            swing_consistency = 1.0
        
        # Calculate late swing (swing in the last 0.2 seconds)
        late_start_idx = max(0, len(trajectory_points) - int(0.2 / 0.01))  # 0.01s time step
        if late_start_idx < len(trajectory_points) - 1:
            late_initial = np.array([
                trajectory_points[late_start_idx].pos_x,
                trajectory_points[late_start_idx].pos_y,
                trajectory_points[late_start_idx].pos_z
            ])
            late_final = np.array([
                trajectory_points[-1].pos_x,
                trajectory_points[-1].pos_y,
                trajectory_points[-1].pos_z
            ])
            late_vector = late_final - late_initial
            late_length = np.linalg.norm(late_vector)
            if late_length > 0:
                late_lateral_deviation = abs(late_final[2] - late_initial[2])
                late_swing = np.degrees(np.arcsin(late_lateral_deviation / late_length))
                
                # Adjust late swing based on the overall swing characteristics
                late_swing = min(late_swing, swing_angle * 0.8)  # Late swing shouldn't exceed 80% of total swing
            else:
                late_swing = 0.0
        else:
            late_swing = 0.0
        
        return {
            'swing_angle': float(swing_angle),
            'swing_magnitude': float(swing_magnitude),
            'swing_direction': float(swing_direction),
            'swing_rate': float(swing_rate),
            'swing_consistency': float(swing_consistency),
            'late_swing': float(late_swing)
        } 