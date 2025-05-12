import pytest
import json
from bat_detection import (
    detect_collision,
    detect_ball_bat_collision,
    calculate_distance,
    process_input
)

# Sample test data
sample_ball = {
    "center": [682, 619, 15.0],
    "radius": 14.0
}

sample_bat = {
    "corners": [
        [700, 600, 5],
        [720, 600, 5],
        [720, 640, 5],
        [700, 640, 5]
    ]
}

sample_input = {
    "ball": sample_ball,
    "bat": sample_bat
}

# Test cases
def test_ball_bat_collision_detection():
    # Test case 1: Ball close to bat (should detect collision)
    close_ball = {"center": [710, 620, 10], "radius": 12.0}
    close_input = {"ball": close_ball, "bat": sample_bat}
    result = detect_collision(close_input)
    assert result["collision"] == True
    assert result["confidence"] != "none"
    
    # Test case 2: Ball far from bat (should not detect collision)
    far_ball = {"center": [800, 700, 50], "radius": 12.0}
    far_input = {"ball": far_ball, "bat": sample_bat}
    result = detect_collision(far_input)
    assert result["collision"] == False
    assert result["confidence"] == "none"

def test_distance_calculation():
    # Test case for 3D distance calculation
    point_a = [1, 2, 3]
    point_b = [4, 5, 6]
    expected_distance = 5.196  # sqrt(27)
    
    actual_distance = calculate_distance(point_a, point_b)
    assert abs(actual_distance - expected_distance) < 0.001  # Allow for floating point precision

def test_process_input():
    # Test the full processing with a sample input
    test_input = {
        "frame_id": 71,
        "detection": {
            "center": [470, 890, 22],
            "radius": 11.0,
            "velocity": [-5, 10, -2]
        },
        "bat": {
            "box_vectors": [
                [450, 880, 20],
                [500, 880, 20],
                [500, 980, 40],
                [450, 980, 40]
            ],
            "swing_velocity": [2, 15, 5]
        },
        "physics": {
            "restitution": 0.8,
            "friction": 0.2
        }
    }
    
    result = process_input(test_input)
    
    # Check that the output has the expected structure
    assert "collision" in result
    assert "trajectory" in result
    assert "collision" in result["collision"]
    assert "updated" in result["trajectory"]