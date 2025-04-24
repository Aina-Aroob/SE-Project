import numpy as np
import logging
import time
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ball_radius = 2.80 # assuming inches

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
    # Test case 1: Ball trajectory that will hit the bat
    print("TEST CASE 1: Ball trajectory with contact")
    ball_trajectory_1 = [
        [10.0, 5.0, 0.0],  # Starting position
        [9.0, 4.5, 0.0],   # Moving toward bat
        [8.0, 4.0, 0.0],   # Getting closer
        [7.0, 3.5, 0.0],   # Getting closer
        [6.0, 3.0, 0.0],   # Getting closer
        [5.0, 2.5, 0.0],   # Very close to bat
        [4.0, 2.0, 0.0]    # Should hit bat edge
    ]
    
    # Define bat edges (simplified as points along the bat)
    bat_edges = [
        [4.0, 0.0, 0.0],  # Bottom edge
        [4.0, 2.0, 0.0],  # Middle point (contact point)
        [4.0, 4.0, 0.0]   # Top edge
    ]
    
    # Test detection
    contact, contact_point = detect_bat_edge_contact(ball_trajectory_1, bat_edges)
    print(f"Contact detected: {contact}")
    if contact:
        print(f"Contact point: {contact_point}")
        
        # Test trajectory update after contact
        # Initial velocity vector (direction the ball was moving)
        ball_velocity = np.array([-1.0, -0.5, 0.0])  # Moving toward the bat
        
        # Bat normal vector (perpendicular to bat surface)
        bat_normal = np.array([1.0, 0.0, 0.0])  # Assuming bat is vertical
        
        # Update trajectory
        new_velocity = update_trajectory(ball_velocity, bat_normal)
        print(f"Original velocity: {ball_velocity}")
        print(f"New velocity after hit: {new_velocity}")
    
    # Test case 2: Ball trajectory that misses the bat
    print("\nTEST CASE 2: Ball trajectory with no contact")
    ball_trajectory_2 = [
        [10.0, 10.0, 0.0],
        [9.0, 9.5, 0.0],
        [8.0, 9.0, 0.0],
        [7.0, 8.5, 0.0],
        [6.0, 8.0, 0.0],
        [5.0, 7.5, 0.0],
        [4.0, 7.0, 0.0]  # Will miss the bat (too high)
    ]
    
    # Test detection
    contact, contact_point = detect_bat_edge_contact(ball_trajectory_2, bat_edges)
    print(f"Contact detected: {contact}")
    if contact:
        print(f"Contact point: {contact_point}")
    
    # Test case 3: Edge case - ball just grazes the bat
    print("\nTEST CASE 3: Ball grazing the bat edge")
    # Calculate a position that's exactly ball_radius away from a bat edge
    grazing_point = [bat_edges[2][0] + ball_radius, bat_edges[2][1], bat_edges[2][2]]
    
    ball_trajectory_3 = [
        [10.0, 4.5, 0.0],
        [9.0, 4.4, 0.0],
        [8.0, 4.3, 0.0],
        [7.0, 4.2, 0.0],
        [6.0, 4.1, 0.0],
        grazing_point    # Exactly ball_radius away from top bat edge
    ]
    
    # Test detection
    contact, contact_point = detect_bat_edge_contact(ball_trajectory_3, bat_edges)
    print(f"Contact detected: {contact}")
    if contact:
        print(f"Contact point: {contact_point}")
        
        # Test trajectory update with different angle
        ball_velocity = np.array([-1.0, -0.1, 0.0])  # Moving almost horizontally
        bat_normal = np.array([0.9, 0.1, 0.0])       # Slightly angled bat
        bat_normal = bat_normal / np.linalg.norm(bat_normal)  # Normalize
        
        # Update trajectory
        new_velocity = update_trajectory(ball_velocity, bat_normal)
        print(f"Original velocity: {ball_velocity}")
        print(f"New velocity after grazing hit: {new_velocity}")
    
    # Bonus: Test with a 3D scenario
    print("\nTEST CASE 4: 3D ball trajectory")
    ball_trajectory_4 = [
        [10.0, 5.0, 2.0],
        [9.0, 4.5, 1.5],
        [8.0, 4.0, 1.0],
        [7.0, 3.5, 0.5],
        [6.0, 3.0, 0.0],
        [5.0, 2.5, -0.5],
        [4.0, 2.0, -1.0]
    ]
    
    # 3D bat edges
    bat_edges_3d = [
        [4.0, 0.0, -2.0],
        [4.0, 1.0, -1.5],
        [4.0, 2.0, -1.0],  # Should be contact point
        [4.0, 3.0, -0.5],
        [4.0, 4.0, 0.0]
    ]
    
    # Test detection
    contact, contact_point = detect_bat_edge_contact(ball_trajectory_4, bat_edges_3d)
    print(f"Contact detected: {contact}")
    if contact:
        print(f"Contact point: {contact_point}")
        
        # Test 3D trajectory update
        ball_velocity = np.array([-1.0, -0.5, -0.5])  # Moving in 3D
        
        # Calculate bat normal based on adjacent points (more realistic)
        p1 = np.array(bat_edges_3d[1])
        p2 = np.array(bat_edges_3d[2])
        p3 = np.array(bat_edges_3d[3])
        
        # Calculate vectors along the bat
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Cross product gives normal vector to the bat surface
        bat_normal = np.cross(v1, v2)
        bat_normal = bat_normal / np.linalg.norm(bat_normal)  # Normalize
        
        # Update trajectory
        new_velocity = update_trajectory(ball_velocity, bat_normal)
        print(f"Original 3D velocity: {ball_velocity}")
        print(f"New 3D velocity after hit: {new_velocity}")