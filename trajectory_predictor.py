import numpy as np
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any


@dataclass
class BallTrajectory:
    """Data class to store the predicted trajectory and related information."""
    trajectory_points: List[Tuple[float, float, float]]
    historical_points: List[Tuple[float, float, float]]
    bounce_point: Optional[Tuple[float, float, float]] = None
    impact_location: Optional[Tuple[float, float, float]] = None
    leg_impact_location: Optional[Tuple[float, float, float]] = None
    will_hit_stumps: bool = False
    will_hit_leg: bool = False
    swing_characteristics: Dict[str, float] = None


class TrajectoryPredictor:
    def __init__(self, data_file: str):
        """Initialize the trajectory predictor with the data file.
        
        Args:
            data_file: Path to the JSON file containing frame data
        """
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.frames = self.data.get('frames', [])
        self.ball_positions = []
        self.times = []
        self.stump_position = None
        self.last_frame = None
        self.last_frame_velocity = None
        self.leg_position = None
        self._extract_data()
        
    def _extract_data(self):
        """Extract ball positions, timestamps, and the last frame data."""
        # First pass: collect all ball positions for velocity calculation
        for frame in self.frames:
            if 'ball' in frame and 'center' in frame['ball']:
                x, y, z = frame['ball']['center']
                self.ball_positions.append((x, y, z))
                self.times.append(frame['frame_id'])
            
            # Extract stump position from any frame (assuming it doesn't move significantly)
            if 'stumps' in frame:
                # Calculate center of stumps as average of corners
                corners = frame['stumps']['corners']
                x_sum = sum(corner[0] for corner in corners)
                y_sum = sum(corner[1] for corner in corners)
                z_sum = sum(corner[2] for corner in corners)
                self.stump_position = (x_sum/4, y_sum/4, z_sum/4)
        
        # Find the last frame (highest frame_id)
        if self.frames:
            self.last_frame = max(self.frames, key=lambda x: x['frame_id'])
            
            # Get leg position from the last frame
            if 'leg' in self.last_frame:
                self.leg_position = self.last_frame['leg']['corners']
            
            # Calculate velocity from the last few frames
            if len(self.ball_positions) >= 3:
                # Use the last 3 frames to calculate velocity
                pos_t0 = self.ball_positions[-3]
                pos_t1 = self.ball_positions[-2]
                pos_t2 = self.ball_positions[-1]
                
                # Calculate velocities using finite differences
                vx1 = pos_t1[0] - pos_t0[0]
                vy1 = pos_t1[1] - pos_t0[1]
                vz1 = pos_t1[2] - pos_t0[2]
                
                vx2 = pos_t2[0] - pos_t1[0]
                vy2 = pos_t1[1] - pos_t1[1]
                vz2 = pos_t2[2] - pos_t1[2]
                
                # Average the velocities for more stability
                vx = (vx1 + vx2) / 2
                vy = (vy1 + vy2) / 2
                vz = (vz1 + vz2) / 2
                
                self.last_frame_velocity = (vx, vy, vz)
            else:
                # If we don't have enough frames, use a simple approximation
                self.last_frame_velocity = (0, 0, 0)
    
    def _model_trajectory(self, t, x0, y0, z0, vx, vy, vz, ax, ay, az):
        """Physical model for ball trajectory including gravity, drag and other forces.
        
        Args:
            t: Time parameter
            x0, y0, z0: Initial positions
            vx, vy, vz: Initial velocities
            ax, ay, az: Accelerations (including gravity, drag, etc.)
        
        Returns:
            Tuple of x, y, z positions at time t
        """
        # Add drag effect (proportional to velocity squared)
        # Using a simplified model for cricket ball drag
        drag_coef = 0.05  # Simplified drag coefficient
        
        # Apply drag to velocities
        vx_drag = vx * (1 - drag_coef * abs(vx) * t)
        vy_drag = vy * (1 - drag_coef * abs(vy) * t)
        vz_drag = vz * (1 - drag_coef * abs(vz) * t)
        
        # Basic physics model with drag-adjusted velocities
        x = x0 + vx_drag * t + 0.5 * ax * t**2
        y = y0 + vy_drag * t + 0.5 * ay * t**2
        z = z0 + vz_drag * t + 0.5 * az * t**2
        
        # Apply spin effects (lateral movement)
        spin_factor = 0.2  # Simplified spin effect
        # More pronounced effect as the ball travels farther
        lateral_movement = spin_factor * t * np.sin(t/5)
        
        # Add spin effect to x position (lateral movement)
        x += lateral_movement
        
        return x, y, z
    
    def _detect_bounce(self, trajectory: List[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
        """Detect bounce point by finding where the ball hits the ground (z close to 0).
        
        Args:
            trajectory: List of (x, y, z) points representing ball trajectory
        
        Returns:
            Bounce point coordinates or None if no bounce detected
        """
        # Extract z-coordinates
        z_coords = [point[2] for point in trajectory]
        
        # Find points where z is close to ground and velocity changes
        for i in range(1, len(z_coords) - 1):
            # Check if this point is close to the ground
            if z_coords[i] < 8:  # Assuming 8 inches is close to ground
                # Check if z direction changes (ball starts moving up after bounce)
                if z_coords[i+1] > z_coords[i]:
                    return trajectory[i]
        
        # Find local minima in z-coordinates as a fallback
        for i in range(1, len(z_coords) - 1):
            if z_coords[i] < z_coords[i-1] and z_coords[i] < z_coords[i+1]:
                # Check if this is a valid bounce (z near ground level)
                if z_coords[i] < 20:  # Assuming ground level is around 0-20 inches
                    return trajectory[i]
        
        return None
    
    def _calculate_swing(self, trajectory: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """Calculate swing characteristics from the trajectory.
        
        Args:
            trajectory: List of (x, y, z) points representing ball trajectory
        
        Returns:
            Dictionary with swing characteristics
        """
        if len(trajectory) < 5:
            return {
                "swing_amount": 0,
                "swing_direction": 0,
                "lateral_deviation": 0,
                "seam_position": "unknown"
            }
            
        # Calculate lateral deviation (swing)
        start_x = trajectory[0][0]
        mid_point = len(trajectory) // 2
        end_x = trajectory[-1][0]
        
        # Linear path would be a straight line from start to end
        expected_mid_x = (start_x + end_x) / 2
        actual_mid_x = trajectory[mid_point][0]
        
        # Deviation from linear path indicates swing
        swing_amount = abs(actual_mid_x - expected_mid_x)
        
        # Direction of swing (positive = right, negative = left)
        swing_direction = 1 if actual_mid_x > expected_mid_x else -1
        
        # Check for late swing (more pronounced in the latter part of trajectory)
        first_half = trajectory[:mid_point]
        second_half = trajectory[mid_point:]
        
        first_half_deviation = abs(first_half[len(first_half)//2][0] - (first_half[0][0] + first_half[-1][0])/2)
        second_half_deviation = abs(second_half[len(second_half)//2][0] - (second_half[0][0] + second_half[-1][0])/2)
        
        # Determine swing type based on where it's more pronounced
        swing_type = "late" if second_half_deviation > first_half_deviation else "early"
        
        # Estimate seam position based on swing characteristics
        seam_position = "across seam" if swing_amount > 30 else "seam up"
        
        return {
            "swing_amount": float(swing_amount),
            "swing_direction": int(swing_direction),
            "lateral_deviation": float(actual_mid_x - expected_mid_x),
            "swing_type": swing_type,
            "seam_position": seam_position
        }
    
    def _check_stump_impact(self, trajectory: List[Tuple[float, float, float]]) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
        """Check if the trajectory will hit the stumps.
        
        Args:
            trajectory: List of (x, y, z) points representing ball trajectory
        
        Returns:
            Tuple of (will_hit_stumps, impact_location)
        """
        if not self.stump_position:
            return False, None
        
        # Define stump dimensions (approximation)
        stump_x, stump_y, stump_z = self.stump_position
        stump_width = 30  # inches (wider than real stumps to account for uncertainty)
        stump_height = 28  # inches
        
        # Check if any point in the trajectory hits the stumps
        for i in range(1, len(trajectory)):
            prev_point = trajectory[i-1]
            curr_point = trajectory[i]
            
            # Check if trajectory crosses the stumps plane
            if (prev_point[1] < stump_y and curr_point[1] >= stump_y) or \
               (prev_point[1] > stump_y and curr_point[1] <= stump_y):
                
                # Interpolate to find exact crossing point
                t = (stump_y - prev_point[1]) / (curr_point[1] - prev_point[1])
                impact_x = prev_point[0] + t * (curr_point[0] - prev_point[0])
                impact_z = prev_point[2] + t * (curr_point[2] - prev_point[2])
                impact_point = (impact_x, stump_y, impact_z)
                
                # Check if impact is within stump dimensions
                if (abs(impact_x - stump_x) < stump_width/2 and 
                    impact_z > 0 and impact_z < stump_height):
                    return True, impact_point
        
        return False, None
    
    def _check_leg_impact(self, trajectory: List[Tuple[float, float, float]]) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
        """Check if the trajectory will hit the batsman's leg.
        
        Args:
            trajectory: List of (x, y, z) points representing ball trajectory
        
        Returns:
            Tuple of (will_hit_leg, impact_location)
        """
        if not self.leg_position:
            return False, None
        
        # Convert leg corners to numpy array for easier manipulation
        leg_corners = np.array(self.leg_position)
        
        # Check if any point in the trajectory hits the leg
        for i in range(1, len(trajectory)):
            prev_point = np.array(trajectory[i-1])
            curr_point = np.array(trajectory[i])
            
            # Calculate direction vector of ball movement
            direction = curr_point - prev_point
            
            # Check for intersection with the leg
            # We'll simplify by checking if the trajectory passes through the leg's bounding box
            
            # Find the bounding box of the leg
            min_x = min(corner[0] for corner in leg_corners)
            max_x = max(corner[0] for corner in leg_corners)
            min_y = min(corner[1] for corner in leg_corners)
            max_y = max(corner[1] for corner in leg_corners)
            min_z = min(corner[2] for corner in leg_corners)
            max_z = max(corner[2] for corner in leg_corners)
            
            # Add some buffer to account for ball radius (approximately 2 inches)
            ball_radius = 2
            min_x -= ball_radius
            max_x += ball_radius
            min_y -= ball_radius
            max_y += ball_radius
            min_z -= ball_radius
            max_z += ball_radius
            
            # Check if the ball trajectory segment intersects the bounding box
            # We'll use a simplified approach with ray-box intersection
            
            # For each axis, calculate time of intersection with the bounding box planes
            t_min_x = (min_x - prev_point[0]) / direction[0] if direction[0] != 0 else float('-inf' if prev_point[0] < min_x else 'inf')
            t_max_x = (max_x - prev_point[0]) / direction[0] if direction[0] != 0 else float('-inf' if prev_point[0] > max_x else 'inf')
            t_min_y = (min_y - prev_point[1]) / direction[1] if direction[1] != 0 else float('-inf' if prev_point[1] < min_y else 'inf')
            t_max_y = (max_y - prev_point[1]) / direction[1] if direction[1] != 0 else float('-inf' if prev_point[1] > max_y else 'inf')
            t_min_z = (min_z - prev_point[2]) / direction[2] if direction[2] != 0 else float('-inf' if prev_point[2] < min_z else 'inf')
            t_max_z = (max_z - prev_point[2]) / direction[2] if direction[2] != 0 else float('-inf' if prev_point[2] > max_z else 'inf')
            
            # Find the maximum entry time and minimum exit time
            t_enter = max(min(t_min_x, t_max_x), min(t_min_y, t_max_y), min(t_min_z, t_max_z))
            t_exit = min(max(t_min_x, t_max_x), max(t_min_y, t_max_y), max(t_min_z, t_max_z))
            
            # Check if there's an intersection (entry happens before exit and within segment)
            if t_enter <= t_exit and 0 <= t_enter <= 1:
                # Calculate the impact point
                impact_point = tuple(prev_point + t_enter * direction)
                return True, impact_point
        
        return False, None
    
    def _extrapolate_stump_impact(self, trajectory: List[Tuple[float, float, float]], leg_impact_idx: int = None) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
        """Extrapolate trajectory to find where it would hit the stumps, even after leg impact.
        
        Args:
            trajectory: List of (x, y, z) points representing ball trajectory
            leg_impact_idx: Index of leg impact point in trajectory, if known
        
        Returns:
            Tuple of (would_hit_stumps, impact_location)
        """
        if not self.stump_position:
            return False, None
            
        # Define stump dimensions (approximation)
        stump_x, stump_y, stump_z = self.stump_position
        stump_width = 30  # inches (wider than real stumps to account for uncertainty)
        stump_height = 28  # inches
        
        # If we know the leg impact point, we'll use the trajectory direction at that point
        # Otherwise, we'll use the last few points of the trajectory
        if leg_impact_idx is not None and leg_impact_idx > 0 and leg_impact_idx < len(trajectory):
            # Use points just before leg impact to determine direction
            pre_impact_idx = max(0, leg_impact_idx - 1)
            direction_pt1 = trajectory[pre_impact_idx]
            direction_pt2 = trajectory[leg_impact_idx]
        elif len(trajectory) >= 2:
            # Use the last two points to determine direction
            direction_pt1 = trajectory[-2]
            direction_pt2 = trajectory[-1]
        else:
            return False, None
            
        # Calculate direction vector
        dx = direction_pt2[0] - direction_pt1[0]
        dy = direction_pt2[1] - direction_pt1[1]
        dz = direction_pt2[2] - direction_pt1[2]
        
        # If not moving in y direction, can't hit stumps
        if abs(dy) < 1e-6:
            return False, None
            
        # Calculate where the extrapolated line intersects the stumps plane
        starting_point = direction_pt2  # Use the last point of our analysis
        t = (stump_y - starting_point[1]) / dy
        
        # If t is negative, we're moving away from stumps
        if t < 0:
            return False, None
            
        # Calculate the impact coordinates
        impact_x = starting_point[0] + t * dx
        impact_z = starting_point[2] + t * dz
        impact_point = (impact_x, stump_y, impact_z)
        
        # Check if impact is within stump dimensions
        if (abs(impact_x - stump_x) < stump_width/2 and 
            impact_z > 0 and impact_z < stump_height):
            return True, impact_point
            
        return False, None
    
    def predict_trajectory(self, future_frames: int = 10) -> BallTrajectory:
        """Predict the future trajectory of the ball from the last frame.
        
        Args:
            future_frames: Number of future frames to predict
        
        Returns:
            BallTrajectory object with prediction results
        """
        if not self.last_frame or 'ball' not in self.last_frame:
            raise ValueError("No valid last frame data found")
            
        # Get initial position from last frame
        x0, y0, z0 = self.last_frame['ball']['center']
        
        # Use calculated velocity or set default if not available
        if self.last_frame_velocity:
            vx, vy, vz = self.last_frame_velocity
        else:
            # Estimate velocity based on the context of cricket
            # Ball moving toward the batsman (decreasing y)
            vx = 0      # Lateral movement
            vy = -30    # Forward movement (toward batsman)
            vz = 0      # Vertical movement
        
        # Physics parameters
        gravity = -9.8 * 0.0254 * 4  # Gravity in inches/frame^2 (scaled for cricket)
        ax = 0          # No acceleration in x
        ay = 0          # No acceleration in y
        az = gravity    # Gravity in z
        
        # Generate times for prediction
        times = np.arange(0, future_frames + 1)
        
        # Generate trajectory points
        trajectory = [
            self._model_trajectory(t, x0, y0, z0, vx, vy, vz, ax, ay, az) 
            for t in times
        ]
        
        # Detect bounce point
        bounce_point = self._detect_bounce(trajectory)
        
        # Calculate swing characteristics
        swing_characteristics = self._calculate_swing(trajectory)
        
        # Check if ball will hit leg
        will_hit_leg, leg_impact_location = self._check_leg_impact(trajectory)
        leg_impact_idx = None
        
        # If leg impact found, find its position in the trajectory list
        if will_hit_leg and leg_impact_location:
            for i, point in enumerate(trajectory):
                distance = np.sqrt(sum((a - b)**2 for a, b in zip(point, leg_impact_location)))
                if distance < 5:  # Use small threshold to find closest point
                    leg_impact_idx = i
                    break
        
        # Check if ball will hit stumps directly
        will_hit_stumps, impact_location = self._check_stump_impact(trajectory)
        
        # If not hitting stumps directly but hitting leg, check if it would hit stumps later
        would_hit_stumps = False
        extrapolated_impact = None
        
        if not will_hit_stumps and will_hit_leg:
            would_hit_stumps, extrapolated_impact = self._extrapolate_stump_impact(trajectory, leg_impact_idx)
            
            # Update stump impact prediction
            if would_hit_stumps and extrapolated_impact:
                will_hit_stumps = True
                impact_location = extrapolated_impact
        
        return BallTrajectory(
            trajectory_points=trajectory,
            historical_points=self.ball_positions,
            bounce_point=bounce_point,
            impact_location=impact_location,
            leg_impact_location=leg_impact_location,
            will_hit_stumps=will_hit_stumps, 
            will_hit_leg=will_hit_leg,
            swing_characteristics=swing_characteristics
        )
    

def main():
    """Main function to demonstrate trajectory prediction."""
    # Initialize predictor with data file
    predictor = TrajectoryPredictor('sample_input.json')
    
    # Predict trajectory
    trajectory = predictor.predict_trajectory(future_frames=10)
    
    # Print results
    print(f"Predicted trajectory from last frame (frame {predictor.last_frame['frame_id']})")
    print(f"Historical ball positions: {len(trajectory.historical_points)} frames")
    print(f"Predicted ball positions: {len(trajectory.trajectory_points)} frames")
    
    # Detail the impact predictions
    impact_results = []
    if trajectory.will_hit_leg:
        impact_results.append(f"WILL HIT LEG at ({trajectory.leg_impact_location[0]:.2f}, {trajectory.leg_impact_location[1]:.2f}, {trajectory.leg_impact_location[2]:.2f})")
    
    if trajectory.will_hit_stumps and trajectory.will_hit_leg:
        impact_results.append(f"WOULD HIT STUMPS at ({trajectory.impact_location[0]:.2f}, {trajectory.impact_location[1]:.2f}, {trajectory.impact_location[2]:.2f}) - LBW CANDIDATE")
    elif trajectory.will_hit_stumps:
        impact_results.append(f"WILL HIT STUMPS at ({trajectory.impact_location[0]:.2f}, {trajectory.impact_location[1]:.2f}, {trajectory.impact_location[2]:.2f})")
    
    if not impact_results:
        print("Prediction: WILL MISS ALL TARGETS")
    else:
        print("Prediction: " + " & ".join(impact_results))
    
    if trajectory.bounce_point:
        print(f"Bounce point: ({trajectory.bounce_point[0]:.2f}, {trajectory.bounce_point[1]:.2f}, {trajectory.bounce_point[2]:.2f})")
    
    print(f"Swing characteristics:")
    for key, value in trajectory.swing_characteristics.items():
        if isinstance(value, (int, float)):
            print(f"  - {key}: {value:.2f}")
        else:
            print(f"  - {key}: {value}")
    
    
    # Prepare output for next module
    output = {
        "previous_trajectory": trajectory.historical_points,
        "predicted_trajectory": trajectory.trajectory_points,
        "bounce_point": trajectory.bounce_point,
        "stump_impact_location": trajectory.impact_location,
        "leg_impact_location": trajectory.leg_impact_location,
        "will_hit_stumps": trajectory.will_hit_stumps,
        "will_hit_leg": trajectory.will_hit_leg,
        "swing_characteristics": trajectory.swing_characteristics
    }
    
    # Add collision information if it exists in the original data
    if 'collision' in predictor.data:
        output["collision"] = predictor.data["collision"]
    
    # Save output to JSON file for next module
    with open('sample_output.json', 'w') as f:
        # Convert tuples to lists for JSON serialization
        json_output = {
            "previous_trajectory": [[float(x), float(y), float(z)] for x, y, z in output["previous_trajectory"]],
            "predicted_trajectory": [[float(x), float(y), float(z)] for x, y, z in output["predicted_trajectory"]],
            "bounce_point": [float(x) for x in output["bounce_point"]] if output["bounce_point"] else None,
            "stump_impact_location": [float(x) for x in output["stump_impact_location"]] if output["stump_impact_location"] else None,
            "leg_impact_location": [float(x) for x in output["leg_impact_location"]] if output["leg_impact_location"] else None,
            "will_hit_stumps": output["will_hit_stumps"],
            "will_hit_leg": output["will_hit_leg"],
            "swing_characteristics": {k: float(v) if isinstance(v, (int, float)) else v 
                                     for k, v in output["swing_characteristics"].items()}
        }

        # Add collision from output if it exists
        if "collision" in output:
            json_output["collision"] = output["collision"]
            
        json.dump(json_output, f, indent=2)


if __name__ == "__main__":
    main() 