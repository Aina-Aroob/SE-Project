"""
Cricket Ball-Bat Collision and Trajectory Update System

This module detects collisions between a cricket ball and bat
based on JSON input containing ball position and bat box vectors,
and updates the ball trajectory after collisions.
"""
import json
import math
import numpy as np
import base64
import io
from pydub import AudioSegment
import scipy.signal as signal
import matplotlib.pyplot as plt
import tempfile
import os

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
    if "ball" not in input_data:
        return {
            "collision": False,
            "confidence": "none", 
            "method": "none",
            "details": "No ball data provided"
        }
    
    ball_center = input_data["ball"]["center"]
    ball_radius = input_data["ball"]["radius"]
    
    # Extract bat data
    if "bat" not in input_data:
        return {
            "collision": False,
            "confidence": "none", 
            "method": "none",
            "details": "No bat data provided"
        }
    
    bat_corners = input_data["bat"]["corners"]
    
    # Check if we have audio data
    has_audio = "audio" in input_data and input_data["audio"]
    
    # Perform spatial collision detection
    spatial_result = detect_ball_bat_collision(ball_center, ball_radius, bat_corners)
    
    # If audio is available, use it to enhance detection
    audio_result = {"collision": False, "confidence": "none", "method": "none"}
    if has_audio:
        audio_result = analyze_audio_for_collision(input_data["audio"])
    
    # Combine results
    combined_collision = spatial_result["collision"]
    confidence = spatial_result["confidence"]
    
    return {
        "collision": combined_collision,
        "confidence": confidence,
        "spatial_detection": spatial_result,
        "audio_detection": audio_result,
        "method": "combined" if has_audio else "spatial",
        "details": "Collision detected" if combined_collision else "No collision detected"
    }

def detect_ball_bat_collision(ball_center, ball_radius, bat_box):
    """
    Detect collision between ball and bat based on spatial data
    """
    bat_obb = create_oriented_bounding_box(bat_box)
    closest_point = find_closest_point_on_obb(ball_center, bat_obb)
    distance = calculate_distance(ball_center, closest_point)
    collision = distance <= ball_radius
    
    return {
        "collision": collision,
        "distance": distance,
        "collision_point": closest_point,  # Always return the closest point
        "bat_obb": bat_obb,
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
    # 1. Decode the base64 audio
    try:
        audio_bytes = base64.b64decode(audio_base64)
        
        # 2. Convert bytes to audio using pydub
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_filename = temp_file.name
        
        # 3. Load audio file and convert to numpy array
        audio = AudioSegment.from_file(temp_filename)
        os.unlink(temp_filename)  # Delete temp file
        
        # Convert to mono and get samples as numpy array
        audio = audio.set_channels(1)
        samples = np.array(audio.get_array_of_samples())
        
        # 4. Analyze for impact spike
        # Detect rapid amplitude changes (characteristic of ball-bat contact)
        window_size = 100  # Adjust based on audio sample rate
        threshold_factor = 3.0  # How many times above average is considered a spike
        
        # Calculate rolling standard deviation to detect sudden changes
        rolling_std = np.std([samples[i:i+window_size] for i in range(0, len(samples)-window_size)])
        std_threshold = rolling_std * threshold_factor
        
        # Find segments with high standard deviation (indicating rapid changes)
        high_std_segments = []
        for i in range(0, len(samples)-window_size):
            segment_std = np.std(samples[i:i+window_size])
            if segment_std > std_threshold:
                high_std_segments.append((i, segment_std))
        
        # 5. Analyze frequency content of potential impact segments
        # Ball-bat impacts typically produce high-frequency components
        has_impact = len(high_std_segments) > 0
        
        # For more precise analysis, examine the frequency content
        freq_match = False
        if has_impact:
            # Take the segment with highest std dev
            max_segment = max(high_std_segments, key=lambda x: x[1])
            segment_start = max_segment[0]
            
            # Analyze frequency using FFT
            segment_data = samples[segment_start:segment_start+window_size]
            freqs = np.fft.fftfreq(len(segment_data), 1/audio.frame_rate)
            fft_data = np.abs(np.fft.fft(segment_data))
            
            # Check if there's significant energy in higher frequencies (4kHz-8kHz)
            # This frequency range is typical for ball-bat impacts
            high_freq_indices = np.where((freqs > 4000) & (freqs < 8000))[0]
            low_freq_indices = np.where((freqs > 100) & (freqs < 1000))[0]
            
            high_freq_energy = np.sum(fft_data[high_freq_indices])
            low_freq_energy = np.sum(fft_data[low_freq_indices])
            
            # Bat-ball contact typically has good high/low frequency ratio
            freq_match = (high_freq_energy / low_freq_energy) > 0.5
        
        # 6. Calculate confidence based on analysis
        confidence = "none"
        if has_impact and freq_match:
            confidence = "high"
        elif has_impact:
            confidence = "medium"
        
        # Generate output
        result = {
            "collision": has_impact and freq_match,
            "confidence": confidence,
            "method": "audio",
            "details": {
                "amplitude_spike": has_impact,
                "frequency_match": freq_match,
                "spike_count": len(high_std_segments),
                "max_deviation": max(segment[1] for segment in high_std_segments) if high_std_segments else 0
            }
        }
        
        return result
    
    except Exception as e:
        return {
            "collision": False,
            "confidence": "none",
            "method": "audio",
            "error": str(e),
            "details": "Failed to process audio data"
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
    """
    if not collision_result["collision"]:
        return {
            "updated": False,
            "velocity": input_data["ball"].get("velocity", [0, 0, 0]),
            "details": "No collision detected"
        }
    
    # Get collision point or use ball center as fallback
    collision_point = collision_result.get("collision_point", input_data["ball"]["center"])
    
    ball_center = input_data["ball"]["center"]
    ball_velocity = input_data["ball"].get("velocity", [0, 0, 0])
    bat_swing_velocity = input_data["bat"].get("swing_velocity", [0, 0, 0])
    bat_obb = collision_result["bat_obb"]
    
    # Physics parameters with defaults
    physics = input_data.get("physics", {})
    restitution = physics.get("restitution", 0.8)
    friction = physics.get("friction", 0.2)
    
    # Calculate normal vector
    normal = normalize_vector(subtract_vectors(ball_center, collision_point))

    # Physics parameters
    restitution = input_data.get("physics", {}).get("restitution", 0.8)
    friction = input_data.get("physics", {}).get("friction", 0.2)
    
    # Calculate normal vector at collision point
    normal = normalize_vector(subtract_vectors(ball_center, collision_point))
    
    # Calculate reflected velocity components
    v_normal = scale_vector(normal, dot_product(ball_velocity, normal))
    v_tangent = subtract_vectors(ball_velocity, v_normal)
    
    reflected_normal = scale_vector(v_normal, -restitution)
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
    
    # Add spin effect
    offset = subtract_vectors(collision_point, bat_obb["center"])
    x_offset = dot_product(offset, bat_obb["basis"][0])
    y_offset = dot_product(offset, bat_obb["basis"][1])
    
    spin_effect = cross_product(
        bat_obb["basis"][2],
        [x_offset, y_offset, 0]
    )
    
    spin_factor = 0.2
    spin_contribution = scale_vector(spin_effect, spin_factor)
    final_velocity = add_vectors(new_velocity, spin_contribution)
    
    return {
        "updated": True,
        "previous_velocity": ball_velocity,
        "velocity": final_velocity,
        "speed": vector_length(final_velocity),
        "direction": normalize_vector(final_velocity),
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
    """
    try:
        input_data = json.loads(json_input) if isinstance(json_input, str) else json_input
        frames = input_data.get("frames", [])
        audio_data = input_data.get("audio_base64", None)
        
        # Find first frame with both ball and bat data
        collision_frame = next(
            (frame for frame in frames if frame.get("ball") and frame.get("bat")), 
            None
        )
        
        if not collision_frame:
            return {
                "error": "No frame with both ball and bat data found",
                "available_frames": len(frames)
            }
        
        # Add audio data if available
        if audio_data:
            collision_frame["audio"] = audio_data
        
        # Build trajectory history
        previous_trajectory = [
            frame["ball"]["center"] for frame in frames 
            if frame.get("ball") and frame["ball"].get("center")
        ]
        
         # Collect leg data from all frames
        leg_data = [
            {
                "frame_id": frame["frame_id"],
                "leg_corners": frame["leg"]["corners"],
                "timestamp": frame["leg"]["corners"][0][2]  # Using the z-coordinate as timestamp
            } 
            for frame in frames if frame.get("leg") and frame["leg"].get("corners")
        ]
        # Detect collision
        collision_result = detect_collision(collision_frame)
        #print(collision_result)
        # Ensure we have bat OBB
        if collision_result["collision"] and "bat_obb" not in collision_result:
            bat_corners = collision_frame["bat"]["corners"]
            collision_result["bat_obb"] = create_oriented_bounding_box(bat_corners)
        
        # Update trajectory
        trajectory_result = update_trajectory(collision_frame, collision_result)
        
        # Add trajectory prediction if collision occurred
        if collision_result["collision"] and trajectory_result["updated"]:
            starting_position = (
                collision_result.get("collision_point") or
                collision_frame["ball"]["center"]
            )
            
            trajectory_steps = predict_trajectory(
                starting_position,
                trajectory_result["velocity"],
                time_steps=10,
                time_step=0.1,
                gravity=[0, -9.8, 0]
            )
            
            result["trajectory_prediction"] = {
                "previous_trajectory": previous_trajectory,
                "leg_data": leg_data,
                "steps": trajectory_steps,
                "collision_index": len(previous_trajectory) - 1 if previous_trajectory else 0,
                "history_steps": len(previous_trajectory),
                "future_steps": len(trajectory_steps),
                "starting_from": "collision_point" if "collision_point" in collision_result else "ball_position"
            }
        
            return result
        
        #previous trajectory being returned if collision is false
        elif collision_result["collision"] == False:
            # Prepare result
            result = {
                "previous_trajectory": previous_trajectory,
                "leg_data": leg_data,
                "collision": collision_result,
                "trajectory": trajectory_result,
                "field_setup": {
                    "stumps_position": collision_frame.get("stumps", {}).get("corners"),
                    "batsman_orientation": collision_frame.get("batsman_orientation", "unknown")
                }
            }
            return result
            
    
    except Exception as e:
        return {
            "error": f"Error processing input: {str(e)}",
            "type": type(e).__name__
        }    
    except Exception as e:
        return {
            "error": f"Error processing input: {str(e)}",
            "type": type(e).__name__
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
    #json.loads(json_data)
    with open('correct_input.json', 'r') as file:
        data = json.load(file)
    result = process_input(data)
    print("Printing Result: ")
    print(result)
    # with open('data.json', 'w') as file:
    #     json.dump(data, file, indent=4)  # 'indent=4' makes the output pretty
    # If collision occurred, visualize the new trajectory
    # if result["collision"]["collision"] and result["trajectory"]["updated"]:
    #     new_trajectory = predict_trajectory(
    #         collision_example["detection"]["center"],
    #         result["trajectory"]["velocity"]
    #     )
    #     steps_dict = {f"Step {i}": pos for i, pos in enumerate(new_trajectory)}
    #     json_output = json.dumps(steps_dict + collision_example["stumps"], indent=2)
    #     #print(json_output)
    #     combined = {**result, "new_trajectory_steps": steps_dict}

    #     # Dump merged result to JSON
    #     json_output = json.dumps(combined, indent=2)
    #     print(json_output)
