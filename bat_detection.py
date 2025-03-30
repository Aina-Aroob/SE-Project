import numpy as np

ball_radius = 2.80 #assumint inches

def detect_bat_edge_contact(ball_trajectory, bat_edges):
    for ball_pos in ball_trajectory:
        for edge in bat_edges:
            distance = np.linalg.norm(np.array(ball_pos) - np.array(edge))
            if distance <= ball_radius:  # Assuming ball_radius is predefined
                return True, ball_pos  # Contact detected
    return False, None  # No contact detected

def update_trajectory(ball_velocity, bat_normal):
    # Reflect ball velocity using bat normal
    reflection = ball_velocity - 2 * np.dot(ball_velocity, bat_normal) * bat_normal
    return reflection  # Updated velocity after impact

if __name__ == "__main__":
    pass