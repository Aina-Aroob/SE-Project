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
        
        # If collision occurred, visualize the new trajectory
        if result["collision"]["collision"] and result["trajectory"]["updated"]:
            new_trajectory = predict_trajectory(
                data["detection"]["center"],
                result["trajectory"]["velocity"]
            )
            steps_dict = {f"Step {i}": pos for i, pos in enumerate(new_trajectory)}
            combined = {**result, "new_trajectory_steps": steps_dict}
            return jsonify(combined)
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