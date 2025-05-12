from flask import Flask, request, jsonify
from bat_detection import process_input, predict_trajectory
import json

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_collision():
    """
    REST API endpoint for bat detection and collision processing
    """
    try:
        data = request.json
        result = process_input(data)
        
        # Add field setup information to the result
        field_setup = {
            "stumps_position": data.get("stumps", {}).get("corners") if "stumps" in data else None,
            "batsman_orientation": data.get("batsman_orientation", "unknown")
        }
        result["field_setup"] = field_setup
        
        # If collision occurred, visualize the new trajectory
        if result["collision"]["collision"] and result["trajectory"]["updated"]:
            # Use collision point as starting position if available
            starting_position = None
            if "collision_point" in result["collision"].get("spatial_detection", {}):
                starting_position = result["collision"]["spatial_detection"]["collision_point"]
            else:
                starting_position = data.get("ball", {}).get("center") or data.get("detection", {}).get("center")
            
            # Generate trajectory prediction
            new_trajectory = predict_trajectory(
                starting_position,
                result["trajectory"]["velocity"]
            )
            
            # Use trajectory steps array directly instead of creating a dictionary with "Step X" keys
            result["trajectory_prediction"] = {
                "steps": new_trajectory,
                "starting_from": "collision_point" if "collision_point" in result["collision"].get("spatial_detection", {}) else "original_position"
            }
            
            return jsonify(result)
        else:
            return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)