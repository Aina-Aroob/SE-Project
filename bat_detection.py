import numpy as np
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ball_radius = 2.80  # assuming inches

def detect_bat_edge_contact(ball_trajectory, bat_edges):
    start_time = time.perf_counter()
    
    for ball_pos in ball_trajectory:
        for edge in bat_edges:
            distance = np.linalg.norm(np.array(ball_pos) - np.array(edge))
            if distance <= ball_radius:
                duration = time.perf_counter() - start_time
                logging.info(f"Contact detected at {ball_pos}, distance={distance:.2f}, time={duration:.6f}s")
                return True, ball_pos
    
    duration = time.perf_counter() - start_time
    logging.info(f"No contact detected, time={duration:.6f}s")
    return False, None

def update_trajectory(ball_velocity, bat_normal):
    start_time = time.perf_counter()

    reflection = ball_velocity - 2 * np.dot(ball_velocity, bat_normal) * bat_normal

    duration = time.perf_counter() - start_time
    logging.info(f"Velocity updated. New vector: {reflection.tolist()}, time={duration:.6f}s")

    return reflection

if __name__ == "__main__":
    pass