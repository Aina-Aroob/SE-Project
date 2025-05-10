import numpy as np

# Conversion constants
INCHES_TO_METERS = 0.0254
METERS_TO_INCHES = 39.3701

def inches_to_meters(value: float) -> float:
    """Convert inches to meters."""
    return value * INCHES_TO_METERS

def meters_to_inches(value: float) -> float:
    """Convert meters to inches."""
    return value * METERS_TO_INCHES

def convert_position_inches_to_meters(position: np.ndarray) -> np.ndarray:
    """Convert a position array from inches to meters."""
    return position * INCHES_TO_METERS

def convert_position_meters_to_inches(position: np.ndarray) -> np.ndarray:
    """Convert a position array from meters to inches."""
    return position * METERS_TO_INCHES

def convert_velocity_inches_to_meters(velocity: np.ndarray) -> np.ndarray:
    """Convert a velocity array from inches/second to meters/second."""
    return velocity * INCHES_TO_METERS

def convert_velocity_meters_to_inches(velocity: np.ndarray) -> np.ndarray:
    """Convert a velocity array from meters/second to inches/second."""
    return velocity * METERS_TO_INCHES 