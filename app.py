from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/decision', methods=['POST'])
def decision_module():
    try:
        data = request.get_json()

        batEdge = data.get("batEdge")
        predictedTraj = data.get("predictedTraj")

        if not batEdge or not predictedTraj:
            return jsonify({"error": "Missing batEdge or predictedTraj input"}), 400

        # Load JSON if string input
        if isinstance(batEdge, str):
            batEdge = json.loads(batEdge)
        if isinstance(predictedTraj, str):
            predictedTraj = json.loads(predictedTraj)

        # Initialize response
        pitch = ""
        impact = ""
        output = {
            "Decision": "",
            "Reason": "",
            "BallPitch": "",
            "BallPitchPoint": None,
            "PadImpact": "",
            "PadImpactPoint": predictedTraj["leg_contact_position"],
            "HittingStumps": predictedTraj["verdict"]["will_hit_stumps"],
            "batsman_orientation": predictedTraj.get("batsman_orientation", "R"),
            "batEdge": batEdge,
            "predictedTraj": predictedTraj
        }

        if batEdge["decision_flag"][0] == True:
            output["Decision"] = "NOT OUT"
            output["Reason"] = "Ball hit the bat"
            return jsonify(output)

        # Read data
        ball_trajectory = batEdge["original_trajectory"]
        left_stump_x = batEdge["stumps"][0]["x"]
        right_stump_x = batEdge["stumps"][2]["x"]

        leg_contact_position = predictedTraj["leg_contact_position"]
        batsman_type = predictedTraj.get("batsman_orientation", "R")

        # Check ball pitch point
        for i in range(1, len(ball_trajectory)):
            if ball_trajectory[i][1] >= ball_trajectory[i - 1][1]:
                pitch_point = ball_trajectory[i]
                break
        else:
            pitch_point = ball_trajectory[0]

        pitch_x = pitch_point[0]
        output["BallPitchPoint"] = pitch_point

        if pitch_x < left_stump_x:
            output["BallPitch"] = "Outside Off" if batsman_type == "R" else "Outside Leg"
        elif pitch_x > right_stump_x:
            output["BallPitch"] = "Outside Leg" if batsman_type == "R" else "Outside Off"
        else:
            output["BallPitch"] = "InLine"

        if output["BallPitch"] == "Outside Leg":
            output["Decision"] = "NOT OUT"
            output["Reason"] = "Ball pitched outside Leg stump"
            return jsonify(output)

        if left_stump_x <= leg_contact_position[0] <= right_stump_x:
            output["PadImpact"] = "InLine"
        else:
            output["PadImpact"] = "Outside Line"

        if output["PadImpact"] == "Outside Line":
            output["Decision"] = "NOT OUT"
            output["Reason"] = "Impact outside the line of stumps"
        elif predictedTraj["verdict"]["will_hit_stumps"]:
            output["Decision"] = "OUT"
            output["Reason"] = "Ball would have hit the stumps"
        else:
            output["Decision"] = "NOT OUT"
            output["Reason"] = "Ball missing the stumps"

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

#Input
#{
# "batEdge": { ... },
#  "predictedTraj": { ... }
#}

#Output
#{
#  "Decision": "OUT",
#  "Reason": "Ball would have hit the stumps",
#  "BallPitch": "InLine",
#  "BallPitchPoint": [1, 1, 1],
#  ...
#}
