from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class StumpDimensions:
    width: float = 0.22  # meters
    height: float = 0.71  # meters
    bail_height: float = 0.13  # meters
    depth: float = 0.1  # meters

@dataclass
class PitchDimensions:
    length: float = 20.12  # meters
    width: float = 3.05  # meters
    crease_length: float = 1.22  # meters

@dataclass
class BallProperties:
    radius: float = 0.036  # meters
    mass: float = 0.156  # kg
    drag_coefficient: float = 0.4
    magnus_coefficient: float = 0.1

@dataclass
class PhysicsConstants:
    gravity: float = 9.81  # m/s²
    air_density: float = 1.225  # kg/m³
    time_step: float = 0.01  # seconds
    simulation_duration: float = 1.0  # seconds

@dataclass
class SurfaceProperties:
    bounce_coefficients: Dict[str, float] = field(default_factory=lambda: {
        "dry": 0.7,
        "damp": 0.6,
        "green": 0.5,
        "normal": 0.65
    })

@dataclass
class SystemConfig:
    stump_dimensions: StumpDimensions = field(default_factory=StumpDimensions)
    pitch_dimensions: PitchDimensions = field(default_factory=PitchDimensions)
    ball_properties: BallProperties = field(default_factory=BallProperties)
    physics_constants: PhysicsConstants = field(default_factory=PhysicsConstants)
    surface_properties: SurfaceProperties = field(default_factory=SurfaceProperties)
    
    # Confidence thresholds
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    })

# Create default configuration
default_config = SystemConfig() 