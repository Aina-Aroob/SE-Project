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

def test_magnus_force_orthogonal_spin():
    physics = BallPhysics()
    # Backspin (spin in +z direction with velocity in +x)
    force = physics.calculate_magnus_force(
        velocity=np.array([10, 0, 0]),
        spin=np.array([0, 0, 5])  # Positive z = backspin
    )
    # Magnus force should be upward for backspin (F = k * ω × v)
    expected_force = np.array([0, physics.magnus_coefficient * 10 * 5, 0])
    assert np.allclose(force, expected_force, rtol=0.01)

def test_magnus_force_zero_spin():
    physics = BallPhysics()
    assert np.all(physics.calculate_magnus_force(
        np.array([10, 0, 0]),
        np.zeros(3)
    ) == 0)  # No spin = no Magnus force

def test_magnus_force_topspin():
    physics = BallPhysics()
    force = physics.calculate_magnus_force(
        velocity=np.array([10, 0, 0]),
        spin=np.array([0, 0, -5])  # Topspin
    )
    expected_force = np.array([0, -physics.magnus_coefficient * 10 * 5, 0])  # Downward
    assert np.allclose(force, expected_force, rtol=0.01)

def test_zero_velocity_magnus_force(): # Edge Case
    physics = BallPhysics()
    force = physics.calculate_magnus_force(np.array([0, 0, 0]), np.array([0, 0, 10]))
    assert np.allclose(force, np.zeros(3), atol=1e-10)

def test_magnus_force_high_spin(): # Edge Case
    physics = BallPhysics()
    force = physics.calculate_magnus_force(np.array([20, 0, 0]), np.array([0, 0, 50]))
    expected_force = np.array([0, physics.magnus_coefficient * 20 * 50, 0])
    assert np.allclose(force, expected_force, rtol=0.01)

def test_bounce_velocity_surfaces():
    physics = BallPhysics()
    # Test surface types
    for surface, coeff in [("dry", 0.7), ("damp", 0.6), ("green", 0.5)]:
        bounced_vel = physics.estimate_bounce_velocity(
            impact_velocity=np.array([10, -5, 0]),
            surface_type=surface
        )
        assert np.isclose(
            np.linalg.norm(bounced_vel),
            coeff * np.linalg.norm([10, 5, 0]),  # Expected velocity reduction
            rtol=0.1
        )

def test_runge_kutta_high_spin():
    physics = BallPhysics()
    physics.air_density = 0  # Disable drag for clearer Magnus effect observation
    physics.magnus_coefficient = 0.2
    
    # Test cases: (spin_z, comparison)
    test_cases = [
        (10, "greater"),   # Backspin should increase height
        (-10, "less")      # Topspin should reduce height
    ]
    
    for spin_z, comparison in test_cases:
        state = BallState(
            position=np.zeros(3),
            velocity=np.array([15, 5, 0]),  # Forward and slightly upward
            spin=np.array([0, 0, spin_z]),
            timestamp=0.0
        )
        
        duration = 0.2  # Long enough to see height difference
        time_step = 0.001
        
        # With spin
        physics.magnus_coefficient = 0.2
        trajectory_with_spin = physics.predict_trajectory(state, time_step=time_step, duration=duration)
        final_y_with_spin = trajectory_with_spin[-1].position[1]
        
        # Without spin
        physics.magnus_coefficient = 0
        trajectory_no_spin = physics.predict_trajectory(state, time_step=time_step, duration=duration)
        final_y_no_spin = trajectory_no_spin[-1].position[1]

        percent_change = abs(final_y_with_spin - final_y_no_spin) / abs(final_y_no_spin)
        assert percent_change > 0.05, (
            f"Height difference due to spin z={spin_z} is not >5% (got {percent_change:.2%})"
        )

        if comparison == "greater":
            assert final_y_with_spin > final_y_no_spin, (
                f"Backspin z={spin_z} should increase lift (got {final_y_with_spin} vs {final_y_no_spin})"
            )
        else:
            assert final_y_with_spin < final_y_no_spin, (
                f"Topspin z={spin_z} should decrease lift (got {final_y_with_spin} vs {final_y_no_spin})"
            )