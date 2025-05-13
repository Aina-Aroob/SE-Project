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
        # Get input data
        data = request.json
        
        # Process the input using bat_detection module
        result = process_input(data)
        
        # Return the result directly since process_input() already handles:
        # - Collision detection
        # - Trajectory updates
        # - Field setup information
        # - Trajectory prediction
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
