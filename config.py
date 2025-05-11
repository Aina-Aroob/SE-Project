from dataclasses import dataclass, field
from typing import Dict, List
from utils import inches_to_meters

@dataclass
class StumpDimensions:
    width: float = 0.22  # meters
    height: float = 0.71  # meters
    bail_height: float = 0.13  # meters
    depth: float = 0.1  # meters

    @classmethod
    def from_inches(cls, width: float, height: float, bail_height: float, depth: float):
        """Create StumpDimensions from inch measurements."""
        return cls(
            width=inches_to_meters(width),
            height=inches_to_meters(height),
            bail_height=inches_to_meters(bail_height),
            depth=inches_to_meters(depth)
        )


@dataclass
class PitchDimensions:
    length: float = 20.12  # meters (792 inches)
    width: float = 3.05  # meters (120 inches)
    crease_length: float = 1.22  # meters (48 inches)

    @classmethod
    def from_inches(cls, length: float, width: float, crease_length: float):
        """Create PitchDimensions from inch measurements."""
        return cls(
            length=inches_to_meters(length),
            width=inches_to_meters(width),
            crease_length=inches_to_meters(crease_length)
        )

@dataclass
class BallProperties:
    radius: float = 0.036  # meters (1.42 inches)
    mass: float = 0.156  # kg
    drag_coefficient: float = 0.4
    magnus_coefficient: float = 0.1

    @classmethod
    def from_inches(cls, radius: float, mass: float, drag_coefficient: float = 0.4, magnus_coefficient: float = 0.1):
        """Create BallProperties from inch measurements."""
        return cls(
            radius=inches_to_meters(radius),
            mass=mass,
            drag_coefficient=drag_coefficient,
            magnus_coefficient=magnus_coefficient
        )

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