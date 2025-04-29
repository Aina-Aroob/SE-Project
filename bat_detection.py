"""
Cricket Ball-Bat Collision and Trajectory Update System

This module detects collisions between a cricket ball and bat
based on JSON input containing ball position and bat box vectors,
and updates the ball trajectory after collisions.
"""
import json
import math
import numpy as np

# Sample input structure (for reference)
sample_input = {
    "frame_id": 70,
    "detection": {
        "center": [472, 893, 25],  # x, y, z coordinates of the ball
        "radius": 11.0,  # radius of the ball in pixels/units
        "velocity": [-5, 10, -2]  # incoming velocity vector [vx, vy, vz]
    },
    "bat": {
        "box_vectors": [
            [450, 880, 20],  # bottom-left corner
            [500, 880, 20],  # bottom-right corner
            [500, 980, 40],  # top-right corner  
            [450, 980, 40]   # top-left corner
        ],
        "swing_velocity": [2, 15, 5]  # bat swing velocity [vx, vy, vz]
    },
    "audio": "base64EncodedAudioString",  # will contain sound data
    "physics": {
        "restitution": 0.8,  # coefficient of restitution (ball bounce)
        "friction": 0.2      # friction coefficient
    }
}


def detect_collision(input_data):
    """
    Determine if the ball is colliding with the bat
    
    Args:
        input_data (dict): The JSON input with ball and bat data
        
    Returns:
        dict: Collision result with details
    """
    # Extract ball data
    ball_center = input_data["detection"]["center"]
    ball_radius = input_data["detection"]["radius"]
    
    # Extract bat data (assuming bat data is available)
    if "bat" not in input_data or "box_vectors" not in input_data["bat"]:
        return {
            "collision": False,
            "error": "Missing bat data"
        }
    
    bat_box = input_data["bat"]["box_vectors"]
    
    # Check if we have audio data (can be used for sound-based collision detection)
    has_audio = "audio" in input_data and input_data["audio"]
    
    # Perform collision detection algorithm
    collision_result = detect_ball_bat_collision(ball_center, ball_radius, bat_box)
    
    # If audio is available, we could enhance detection with sound analysis
    if has_audio and not collision_result["collision"]:
        audio_result = analyze_audio_for_collision(input_data["audio"])
        if audio_result["collision"]:
            # If audio suggests collision but spatial data doesn't,
            # we might have a grazing collision or a detection error
            return {
                "collision": True,
                "confidence": "medium",
                "method": "audio",
                "details": "Collision detected via sound analysis"
            }
    
    return collision_result


def detect_ball_bat_collision(ball_center, ball_radius, bat_box):
    """
    Detect collision between ball and bat based on spatial data
    
    Args:
        ball_center (list): [x, y, z] coordinates of ball center
        ball_radius (float): Radius of the ball
        bat_box (list): Array of 4 [x, y, z] coordinates representing bat corners
        
    Returns:
        dict: Collision result
    """
    # For 3D collision detection:
    # 1. Convert bat box vectors to a 3D oriented bounding box (OBB)
    # 2. Calculate the closest point on the OBB to the ball center
    # 3. Check if the distance from ball center to the closest point is less than ball radius
    
    # Create bat oriented bounding box
    bat_obb = create_oriented_bounding_box(bat_box)
    
    # Find closest point on bat to ball center
    closest_point = find_closest_point_on_obb(ball_center, bat_obb)
    
    # Calculate distance between ball center and closest point
    distance = calculate_distance(ball_center, closest_point)
    
    # Collision occurs if distance is less than ball radius
    collision = distance <= ball_radius
    
    return {
        "collision": collision,
        "distance": distance,
        "collision_point": closest_point if collision else None,
        "bat_obb": bat_obb,  # Include OBB for trajectory calculation
        "confidence": "high" if collision else "none",
        "method": "spatial",
        "details": f"Ball intersects bat by {ball_radius - distance:.2f} units" if collision 
                  else f"Ball is {distance - ball_radius:.2f} units away from bat"
    }


def create_oriented_bounding_box(corners):
    """
    Create an oriented bounding box from 4 corner points
    
    Args:
        corners (list): Array of 4 [x, y, z] coordinates
        
    Returns:
        dict: OBB representation
    """
    # Calculate center of the box
    center = calculate_box_center(corners)
    
    # Calculate the three basis vectors of the box
    vec_x = subtract_vectors(corners[1], corners[0])  # Bottom edge
    vec_y = subtract_vectors(corners[3], corners[0])  # Left edge
    
    # Handle missing Z dimension by creating a default height vector
    if len(corners) > 4 and corners[4]:
        vec_z = subtract_vectors(corners[4], corners[0])  # Height
    else:
        vec_z = [corners[0][0], corners[0][1], (corners[0][2] if len(corners[0]) > 2 else 0) + 20]
        vec_z = subtract_vectors(vec_z, corners[0])  # Default height
    
    # Normalize basis vectors
    basis_x = normalize_vector(vec_x)
    basis_y = normalize_vector(vec_y)
    basis_z = normalize_vector(vec_z)
    
    # Calculate half-widths along each axis
    half_width = vector_length(vec_x) / 2
    half_height = vector_length(vec_y) / 2
    half_depth = vector_length(vec_z) / 2
    
    return {
        "center": center,
        "basis": [basis_x, basis_y, basis_z],
        "half_size": [half_width, half_height, half_depth]
    }


def find_closest_point_on_obb(point, obb):
    """
    Find the closest point on an OBB to a given point
    
    Args:
        point (list): [x, y, z] coordinates of the point
        obb (dict): Oriented bounding box
        
    Returns:
        list: [x, y, z] coordinates of closest point on OBB
    """
    # Direction vector from box center to point
    direction = subtract_vectors(point, obb["center"])
    
    # For each axis of the box, calculate the projection
    # and clamp it to the box's half-width along that axis
    closest_point = obb["center"].copy()
    
    for i in range(3):
        # Project direction onto basis vector
        projection = dot_product(direction, obb["basis"][i])
        
        # Clamp projection to box half-size
        clamped = max(-obb["half_size"][i], min(projection, obb["half_size"][i]))
        
        # Add the clamped contribution of this basis vector
        scaled_basis = scale_vector(obb["basis"][i], clamped)
        closest_point = add_vectors(closest_point, scaled_basis)
    
    return closest_point


def analyze_audio_for_collision(audio_base64):
    """
    Analyze audio data to detect ball-bat collision
    
    Args:
        audio_base64 (str): Base64 encoded audio data
        
    Returns:
        dict: Audio-based collision detection result
    """
    # In a real implementation, you would:
    # 1. Decode the base64 audio
    # 2. Process the audio signal to detect impact sounds
    # 3. Use characteristics like amplitude spike, frequency profile, etc.
    
    # Placeholder implementation
    return {
        "collision": False,  # Would be determined by audio analysis
        "confidence": "low",
        "method": "audio",
        "details": "Audio analysis not implemented in this example"
    }


def calculate_distance(point_a, point_b):
    """
    Calculate distance between two 3D points
    
    Args:
        point_a (list): [x, y, z] coordinates
        point_b (list): [x, y, z] coordinates
        
    Returns:
        float: Distance
    """
    # Ensure points have 3 dimensions
    a = point_a + [0] * (3 - len(point_a)) if len(point_a) < 3 else point_a
    b = point_b + [0] * (3 - len(point_b)) if len(point_b) < 3 else point_b
    
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def calculate_box_center(corners):
    """
    Calculate center of a box from its corners
    
    Args:
        corners (list): Array of corner coordinates
        
    Returns:
        list: [x, y, z] center coordinates
    """
    sum_x, sum_y, sum_z = 0, 0, 0
    
    for corner in corners:
        sum_x += corner[0]
        sum_y += corner[1]
        # Handle case where z might be missing
        sum_z += corner[2] if len(corner) > 2 else 0
    
    return [
        sum_x / len(corners),
        sum_y / len(corners),
        sum_z / len(corners)
    ]


def subtract_vectors(a, b):
    """
    Subtract vector B from vector A
    
    Args:
        a (list): [x, y, z] vector
        b (list): [x, y, z] vector
        
    Returns:
        list: Resulting vector
    """
    # Ensure vectors have 3 dimensions
    a_vec = a + [0] * (3 - len(a)) if len(a) < 3 else a
    b_vec = b + [0] * (3 - len(b)) if len(b) < 3 else b
    
    return [a_vec[0] - b_vec[0], a_vec[1] - b_vec[1], a_vec[2] - b_vec[2]]


def add_vectors(a, b):
    """
    Add two vectors
    
    Args:
        a (list): [x, y, z] vector
        b (list): [x, y, z] vector
        
    Returns:
        list: Resulting vector
    """
    # Ensure vectors have 3 dimensions
    a_vec = a + [0] * (3 - len(a)) if len(a) < 3 else a
    b_vec = b + [0] * (3 - len(b)) if len(b) < 3 else b
    
    return [a_vec[0] + b_vec[0], a_vec[1] + b_vec[1], a_vec[2] + b_vec[2]]


def scale_vector(vector, scalar):
    """
    Scale a vector by a scalar
    
    Args:
        vector (list): [x, y, z] vector
        scalar (float): Scale factor
        
    Returns:
        list: Scaled vector
    """
    # Ensure vector has 3 dimensions
    vec = vector + [0] * (3 - len(vector)) if len(vector) < 3 else vector
    
    return [vec[0] * scalar, vec[1] * scalar, vec[2] * scalar]


def vector_length(vector):
    """
    Calculate length of a vector
    
    Args:
        vector (list): [x, y, z] vector
        
    Returns:
        float: Vector length
    """
    # Ensure vector has 3 dimensions
    vec = vector + [0] * (3 - len(vector)) if len(vector) < 3 else vector
    
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def normalize_vector(vector):
    """
    Normalize a vector (make unit length)
    
    Args:
        vector (list): [x, y, z] vector
        
    Returns:
        list: Normalized vector
    """
    # Ensure vector has 3 dimensions
    vec = vector + [0] * (3 - len(vector)) if len(vector) < 3 else vector
    
    length = vector_length(vec)
    if length == 0:
        return [0, 0, 0]
    
    return [vec[0] / length, vec[1] / length, vec[2] / length]


def dot_product(a, b):
    """
    Calculate dot product of two vectors
    
    Args:
        a (list): [x, y, z] vector
        b (list): [x, y, z] vector
        
    Returns:
        float: Dot product
    """
    # Ensure vectors have 3 dimensions
    a_vec = a + [0] * (3 - len(a)) if len(a) < 3 else a
    b_vec = b + [0] * (3 - len(b)) if len(b) < 3 else b
    
    return a_vec[0] * b_vec[0] + a_vec[1] * b_vec[1] + a_vec[2] * b_vec[2]


def cross_product(a, b):
    """
    Calculate cross product of two vectors
    
    Args:
        a (list): [x, y, z] vector
        b (list): [x, y, z] vector
        
    Returns:
        list: Cross product vector
    """
    # Ensure vectors have 3 dimensions
    a_vec = a + [0] * (3 - len(a)) if len(a) < 3 else a
    b_vec = b + [0] * (3 - len(b)) if len(b) < 3 else b
    
    return [
        a_vec[1] * b_vec[2] - a_vec[2] * b_vec[1],
        a_vec[2] * b_vec[0] - a_vec[0] * b_vec[2],
        a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]
    ]


def update_trajectory(input_data, collision_result):
    """
    Update the ball trajectory based on collision with the bat
    
    Args:
        input_data (dict): The input data with ball and bat information
        collision_result (dict): The collision detection result
        
    Returns:
        dict: Updated trajectory data
    """
    if not collision_result["collision"]:
        # No collision, return original velocity
        return {
            "updated": False,
            "velocity": input_data["detection"].get("velocity", [0, 0, 0]),
            "details": "No collision detected"
        }
    
    # Extract ball data
    ball_center = input_data["detection"]["center"]
    ball_velocity = input_data["detection"].get("velocity", [0, 0, 0])
    
    # Extract bat data
    bat_swing_velocity = input_data["bat"].get("swing_velocity", [0, 0, 0])
    bat_obb = collision_result["bat_obb"]
    collision_point = collision_result["collision_point"]
    
    # Physics parameters
    restitution = input_data.get("physics", {}).get("restitution", 0.8)
    friction = input_data.get("physics", {}).get("friction", 0.2)
    
    # Calculate normal vector at collision point (from collision point to ball center)
    normal = normalize_vector(subtract_vectors(ball_center, collision_point))
    
    # Calculate reflected velocity component
    # Use the normal vector to reflect the incoming velocity
    v_normal = scale_vector(normal, dot_product(ball_velocity, normal))
    v_tangent = subtract_vectors(ball_velocity, v_normal)
    
    # Calculate the new velocity by reflecting the normal component
    # and adding the bat's contribution
    reflected_normal = scale_vector(v_normal, -restitution)
    
    # Add bat swing velocity contribution
    # Project bat velocity onto collision normal
    bat_normal_v = dot_product(bat_swing_velocity, normal)
    bat_contribution = scale_vector(normal, max(0, bat_normal_v))
    
    # Apply friction to tangential component
    tangent_factor = 1.0 - friction
    reduced_tangent = scale_vector(v_tangent, tangent_factor)
    
    # Combine all components for new velocity
    new_velocity = add_vectors(
        add_vectors(reflected_normal, reduced_tangent),
        bat_contribution
    )
    
    # Add some spin effect based on collision point relative to bat center
    # This is a simplified approach - could be made more sophisticated
    offset = subtract_vectors(collision_point, bat_obb["center"])
    
    # Project offset onto bat surface to determine spin direction
    x_offset = dot_product(offset, bat_obb["basis"][0])
    y_offset = dot_product(offset, bat_obb["basis"][1])
    
    spin_effect = cross_product(
        bat_obb["basis"][2],  # Bat normal/face direction
        [x_offset, y_offset, 0]  # Offset on bat surface
    )
    
    # Scale spin effect and add to velocity
    spin_factor = 0.2  # Spin influence factor
    spin_contribution = scale_vector(spin_effect, spin_factor)
    final_velocity = add_vectors(new_velocity, spin_contribution)
    
    # Calculate speed and direction
    speed = vector_length(final_velocity)
    direction = normalize_vector(final_velocity)
    
    return {
        "updated": True,
        "previous_velocity": ball_velocity,
        "velocity": final_velocity,
        "speed": speed,
        "direction": direction,
        "collision_point": collision_point,
        "normal": normal,
        "restitution_applied": restitution,
        "friction_applied": friction,
        "spin_effect": spin_contribution,
        "details": "Trajectory updated based on collision physics"
    }


def process_input(json_input):
    """
    Process JSON input for collision detection and trajectory update
    
    Args:
        json_input (str or dict): JSON input string or dictionary
        
    Returns:
        dict: Collision detection and trajectory update result
    """
    try:
        input_data = json.loads(json_input) if isinstance(json_input, str) else json_input
        
        # First detect collision
        collision_result = detect_collision(input_data)
        
        # Then update trajectory if collision occurred
        trajectory_result = update_trajectory(input_data, collision_result)
        
        # Combine results
        result = {
            "collision": collision_result,
            "trajectory": trajectory_result
        }
        
        return result
    except Exception as e:
        return {
            "error": f"Error processing input: {str(e)}"
        }


def predict_trajectory(position, velocity, time_steps=10, time_step=0.1, gravity=[0, -9.8, 0]):
    """
    Predict ball trajectory without collision for visualization
    
    Args:
        position (list): [x, y, z] initial position
        velocity (list): [vx, vy, vz] initial velocity
        time_steps (int): Number of time steps to predict
        time_step (float): Time delta between steps in seconds
        gravity (list): Gravity vector [gx, gy, gz]
        
    Returns:
        list: List of predicted positions
    """
    trajectory = [position]
    current_position = position.copy()
    current_velocity = velocity.copy()
    
    for _ in range(time_steps):
        # Update velocity with gravity
        current_velocity = add_vectors(current_velocity, scale_vector(gravity, time_step))
        
        # Update position with velocity
        current_position = add_vectors(current_position, scale_vector(current_velocity, time_step))
        
        # Add to trajectory
        trajectory.append(current_position.copy())
    
    return trajectory


# Example usage
if __name__ == "__main__":
    # Example with a collision
    collision_example = {
        "frame_id": 71,
        "detection": {
            "center": [470, 890, 22],
            "radius": 11.0,
            "velocity": [-5, 10, -2]  # incoming velocity [vx, vy, vz]
        },
        "bat": {
            "box_vectors": [
                [450, 880, 20],
                [500, 880, 20],
                [500, 980, 40],
                [450, 980, 40]
            ],
            "swing_velocity": [2, 15, 5]  # bat swing velocity [vx, vy, vz]
        },
        "physics": {
            "restitution": 0.8,
            "friction": 0.2
        }
    }
    
    result = process_input(collision_example)
    print(json.dumps(result, indent=2))
    
    # If collision occurred, visualize the new trajectory
    if result["collision"]["collision"] and result["trajectory"]["updated"]:
        new_trajectory = predict_trajectory(
            collision_example["detection"]["center"],
            result["trajectory"]["velocity"]
        )
        print(f"Predicted trajectory after collision (first 5 points):")
        for i, pos in enumerate(new_trajectory[:5]):
            print(f"  Step {i}: {pos}")