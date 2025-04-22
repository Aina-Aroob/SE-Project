from bat_detection import detect_bat_edge_contact, update_trajectory
import numpy as np

def test_no_contact():
    ball_trajectory = [[0, 0, 0]]
    bat_edges = [[10, 10, 10]]
    detected, _ = detect_bat_edge_contact(ball_trajectory, bat_edges)
    assert not detected

def test_reflection():
    ball_velocity = np.array([1, -1, 0])
    bat_normal = np.array([0, 1, 0])
    reflection = update_trajectory(ball_velocity, bat_normal)
    expected = np.array([1, 1, 0])
    assert np.allclose(reflection, expected)