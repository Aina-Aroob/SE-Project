from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel

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
        self.stump_width = 0.22  # meters
        self.stump_height = 0.71  # meters
        self.bail_height = 0.13  # meters
        self.pitch_length = 20.12  # meters
        
    def predict_path(self, input_data: InputData) -> List[TrajectoryPoint]:
        """Predict the ball's path after pad impact."""
        # Convert input data to numpy arrays for calculations
        trajectory = np.array([[p.pos_x, p.pos_y, p.pos_z] for p in input_data.trajectory])
        velocity = np.array(input_data.velocity_vector)
        contact_pos = np.array([
            input_data.leg_contact_position.x,
            input_data.leg_contact_position.y,
            input_data.leg_contact_position.z
        ])
        
        # Calculate post-impact trajectory
        # This is a simplified version - actual implementation would include
        # more complex physics calculations
        predicted_path = []
        time_step = 0.1
        current_pos = contact_pos
        current_vel = velocity * 0.7  # Simplified energy loss on impact
        
        for t in range(10):  # Predict next 10 time steps
            current_pos = current_pos + current_vel * time_step
            current_vel[1] -= 9.81 * time_step  # Gravity effect
            predicted_path.append(TrajectoryPoint(
                pos_x=current_pos[0],
                pos_y=current_pos[1],
                pos_z=current_pos[2],
                timestamp=t * time_step
            ))
            
        return predicted_path
    
    def check_stump_collision(self, predicted_path: List[TrajectoryPoint]) -> Tuple[bool, str, float]:
        """Check if the predicted path would hit the stumps."""
        stump_region = {
            'x': [0, self.stump_width],
            'y': [0, self.stump_height + self.bail_height],
            'z': [self.pitch_length/2 - 0.1, self.pitch_length/2 + 0.1]
        }
        
        for point in predicted_path:
            if (stump_region['x'][0] <= point.pos_x <= stump_region['x'][1] and
                stump_region['y'][0] <= point.pos_y <= stump_region['y'][1] and
                stump_region['z'][0] <= point.pos_z <= stump_region['z'][1]):
                
                # Determine impact region
                if point.pos_y < self.stump_height/3:
                    region = "low"
                elif point.pos_y < 2*self.stump_height/3:
                    region = "middle"
                else:
                    region = "high"
                    
                # Calculate confidence based on trajectory stability
                confidence = 0.85  # This would be calculated based on trajectory analysis
                
                return True, region, confidence
                
        return False, "miss", 0.0
    
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
            confidence=confidence
        )
        
        output = OutputData(
            result_id="res1",
            predicted_path=predicted_path,
            verdict=verdict
        )
        
        return output.dict()

# Example usage
if __name__ == "__main__":
    predictor = LBWPredictor()
    
    # Example input data
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
    
    result = predictor.process_input(input_data)
    print(result) 