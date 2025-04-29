import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class BallState:
    position: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray
    timestamp: float

class BallPhysics:
    def __init__(self):
        self.gravity = 9.81  # m/s²
        self.air_density = 1.225  # kg/m³
        self.ball_radius = 0.036  # meters
        self.ball_mass = 0.156  # kg
        self.drag_coefficient = 0.4
        self.magnus_coefficient = 0.1
        
    def calculate_drag_force(self, velocity: np.ndarray) -> np.ndarray:
        """Calculate drag force on the ball."""
        speed = np.linalg.norm(velocity)
        if speed == 0:
            return np.zeros(3)
            
        drag_force = -0.5 * self.air_density * self.drag_coefficient * \
                    np.pi * self.ball_radius**2 * speed * velocity
        return drag_force
        
    def calculate_magnus_force(self, velocity: np.ndarray, spin: np.ndarray) -> np.ndarray:
        """Calculate Magnus force due to ball spin."""
        if np.linalg.norm(velocity) == 0 or np.linalg.norm(spin) == 0:
            return np.zeros(3)
            
        magnus_force = self.magnus_coefficient * np.cross(spin, velocity)
        return magnus_force
        
    def calculate_acceleration(self, velocity: np.ndarray, spin: np.ndarray) -> np.ndarray:
        """Calculate total acceleration on the ball."""
        drag_force = self.calculate_drag_force(velocity)
        magnus_force = self.calculate_magnus_force(velocity, spin)
        
        total_force = drag_force + magnus_force
        total_force[1] -= self.gravity * self.ball_mass
        
        return total_force / self.ball_mass
        
    def predict_trajectory(self, 
                         initial_state: BallState,
                         time_step: float = 0.01,
                         duration: float = 1.0) -> List[BallState]:
        """Predict ball trajectory using physics calculations."""
        states = [initial_state]
        current_state = initial_state
        
        for t in np.arange(0, duration, time_step):
            # Calculate new state using Runge-Kutta 4th order method
            k1_v = self.calculate_acceleration(current_state.velocity, current_state.spin)
            k1_p = current_state.velocity
            
            k2_v = self.calculate_acceleration(current_state.velocity + 0.5 * time_step * k1_v, current_state.spin)
            k2_p = current_state.velocity + 0.5 * time_step * k1_v
            
            k3_v = self.calculate_acceleration(current_state.velocity + 0.5 * time_step * k2_v, current_state.spin)
            k3_p = current_state.velocity + 0.5 * time_step * k2_v
            
            k4_v = self.calculate_acceleration(current_state.velocity + time_step * k3_v, current_state.spin)
            k4_p = current_state.velocity + time_step * k3_v
            
            # Update velocity and position
            new_velocity = current_state.velocity + (time_step / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            new_position = current_state.position + (time_step / 6) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
            
            # Create new state
            new_state = BallState(
                position=new_position,
                velocity=new_velocity,
                spin=current_state.spin,  # Assuming spin remains constant
                timestamp=current_state.timestamp + time_step
            )
            
            states.append(new_state)
            current_state = new_state
            
        return states
        
    def calculate_impact_energy(self, velocity: np.ndarray) -> float:
        """Calculate kinetic energy at impact."""
        return 0.5 * self.ball_mass * np.linalg.norm(velocity)**2
        
    def estimate_bounce_velocity(self, 
                               impact_velocity: np.ndarray,
                               surface_type: str = "normal") -> np.ndarray:
        """Estimate velocity after bounce based on surface type."""
        # Coefficients for different surface types
        bounce_coefficients = {
            "dry": 0.7,
            "damp": 0.6,
            "green": 0.5,
            "normal": 0.65
        }
        
        coefficient = bounce_coefficients.get(surface_type, 0.65)
        return impact_velocity * coefficient 