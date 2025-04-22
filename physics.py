import numpy as np
from typing import List
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

    def calculate_drag_force(self, velocity: np.ndarray) -> np.ndarray:
        """Calculate drag force on the ball."""
        speed = np.linalg.norm(velocity)
        if speed == 0:
            return np.zeros(3)
            
        drag_force = -0.5 * self.air_density * self.drag_coefficient * \
                    np.pi * self.ball_radius**2 * speed * velocity
        return drag_force

    def calculate_acceleration(self, velocity: np.ndarray, spin: np.ndarray) -> np.ndarray:
        """Calculate total acceleration on the ball."""
        drag_force = self.calculate_drag_force(velocity)
        
        total_force = drag_force
        total_force[1] -= self.gravity * self.ball_mass  
        
        return total_force / self.ball_mass

    def predict_trajectory(self, 
                         initial_state: BallState,
                         time_step: float = 0.01,
                         duration: float = 1.0) -> List[BallState]:
        """Predict ball trajectory using simple Euler integration."""
        states = [initial_state]
        current_state = initial_state
        
        for t in np.arange(0, duration, time_step):
            acceleration = self.calculate_acceleration(current_state.velocity, current_state.spin)
            
            new_velocity = current_state.velocity + acceleration * time_step
            new_position = current_state.position + current_state.velocity * time_step
            
            new_state = BallState(
                position=new_position,
                velocity=new_velocity,
                spin=current_state.spin,
                timestamp=current_state.timestamp + time_step
            )
            
            states.append(new_state)
            current_state = new_state
            
        return states

    def calculate_impact_energy(self, velocity: np.ndarray) -> float:
        """Calculate kinetic energy at impact."""
        return 0.5 * self.ball_mass * np.linalg.norm(velocity)**2