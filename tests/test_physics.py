import numpy as np
from physics import BallPhysics, BallState

def test_calculate_drag_force():
    physics = BallPhysics()
    # Test magnitude and direction
    velocity = np.array([10, 0, 0])
    drag = physics.calculate_drag_force(velocity)
    expected_magnitude = 0.5 * physics.air_density * physics.drag_coefficient * np.pi * (physics.ball_radius**2) * 100
    assert np.isclose(np.linalg.norm(drag), expected_magnitude, rtol=0.01)
    assert np.allclose(drag / np.linalg.norm(drag), [-1, 0, 0])  # Direction check

def test_zero_velocity_edge_case():
    physics = BallPhysics()
    assert np.all(physics.calculate_drag_force(np.zeros(3)) == 0)

def test_calculate_acceleration():
    physics = BallPhysics()
    # Test gravity + drag
    accel = physics.calculate_acceleration(velocity=np.array([0, 10, 0]), spin=np.zeros(3))
    assert accel[1] < -physics.gravity  # Drag adds to downward acceleration
    assert accel[0] == accel[2] == 0  # No horizontal forces

def test_negative_position_trajectory():
    physics = BallPhysics()
    state = BallState(
        position=np.array([-1, -1, -1]),
        velocity=np.array([-5, -5, 0]),  # Moving toward origin
        spin=np.zeros(3),
        timestamp=0.0
    )
    trajectory = physics.predict_trajectory(state, duration=0.1)
    assert trajectory[-1].position[0] < -1  # Verify negative x movement
    
def test_predict_trajectory_no_air_resistance():
    physics = BallPhysics()
    physics.air_density = 0  # Disable drag
    state = BallState(
        position=np.zeros(3),
        velocity=np.array([10, 10, 0]),  # 45° angle
        spin=np.zeros(3),
        timestamp=0.0
    )
    trajectory = physics.predict_trajectory(state, duration=1.0)
    # Analytical projectile motion check
    t = trajectory[-1].timestamp
    expected_x = 10 * t
    expected_y = 10 * t - 0.5 * physics.gravity * t**2
    assert np.isclose(trajectory[-1].position[0], expected_x, rtol=0.01)
    assert np.isclose(trajectory[-1].position[1], expected_y, rtol=0.01)

def test_impact_energy():
    physics = BallPhysics()
    energy = physics.calculate_impact_energy(np.array([20, 0, 0]))
    assert np.isclose(energy, 0.5 * physics.ball_mass * 400)  # 0.5*m*v^2


def test_energy_conservation():
    physics = BallPhysics()
    physics.air_density = 0  # Disable drag for conservation test
    state = BallState(
        position=np.zeros(3),
        velocity=np.array([10,10,0]),
        spin=np.zeros(3),
        timestamp=0
    )
    trajectory = physics.predict_trajectory(state, duration=1.0)
    
    initial_energy = physics.calculate_impact_energy(state.velocity)
    final_energy = (physics.calculate_impact_energy(trajectory[-1].velocity) +
                   physics.ball_mass * physics.gravity * trajectory[-1].position[1])
    assert np.isclose(initial_energy, final_energy, rtol=0.01)  

def test_energy_loss_with_drag():
    physics = BallPhysics()  # Drag enabled
    state = BallState(position=np.zeros(3), velocity=np.array([10,0,0]), spin=np.zeros(3), timestamp=0)
    trajectory = physics.predict_trajectory(state, duration=1.0)
    
    initial_energy = physics.calculate_impact_energy(state.velocity)
    final_energy = (physics.calculate_impact_energy(trajectory[-1].velocity) +
                   physics.ball_mass * physics.gravity * trajectory[-1].position[1])
    
    # Energy should decrease with drag
    assert final_energy < initial_energy
    # Verify reasonable energy loss (e.g., not 100%)
    assert final_energy > 0.5 * initial_energy  