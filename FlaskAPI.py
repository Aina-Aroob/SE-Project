#Flask endpoint: '/decision'
#Input:
#{
#  "batEdge": "{...}",
#  "predictedTraj": "{...}"
#}
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/decision', methods=['POST'])
def decision_module():
    try:
        # Get JSON data from the request
        data = request.get_json()

        batEdge = data.get("batEdge")
        predictedTraj = data.get("predictedTraj")

        # Check if the necessary data is provided
        if not batEdge or not predictedTraj:
            return jsonify({"error": "Missing batEdge or predictedTraj input"}), 400

        # Load the JSON strings into Python objects
        loaded_batEdge = json.loads(batEdge)
        loaded_predictedTraj = json.loads(predictedTraj)

        # Initialize variables
        pitch = ""
        pitch_point = None
        impact = ""
        decision = ""
        reason = ""
        hitting_stumps = loaded_predictedTraj["verdict"]["will_hit_stumps"]

        # Bat Contact Point and flag
        bat_contact_flag = loaded_batEdge["collision"]["collision"]  # Bat contact flag

        if bat_contact_flag == True:
            decision = "NOT OUT"
            reason = "Ball hit the bat"

        # Stumps Line
        ball_trajectory = loaded_batEdge["trajectory_prediction"]["steps"]  # List of ball trajectory points
        left_stump_x = loaded_batEdge["field_setup"]["stumps_position"][0][0]  # Line of stumps left bound
        right_stump_x = loaded_batEdge["field_setup"]["stumps_position"][1][0]  # Line of stumps right bound
        stumps_position_z = loaded_batEdge["field_setup"]["stumps_position"][0][2]  # Z point
        batsman_orientation = loaded_batEdge["field_setup"]["batsman_orientation"]  # Batsman orientation

        if batsman_orientation == 'U':
            batsman_orientation = 'R'

        leg_contact_position = None  # No data from module 4

        # Check if ball is pitched outside the line of stumps or not
        for i in range(1, len(ball_trajectory)):
            if ball_trajectory[i][2] <= stumps_position_z:
                # Check if ball bounces or not
                if ball_trajectory[i][1] >= ball_trajectory[i-1][1]:  # Change of Y signals bounce
                    pitch_point = ball_trajectory[i]
                    if ball_trajectory[i][0] < left_stump_x:
                        if batsman_orientation == 'R':
                            pitch = "Outside Off"
                        else:
                            pitch = "Outside Leg"
                    elif ball_trajectory[i][0] > right_stump_x:
                        if batsman_orientation == 'R':
                            pitch = "Outside Leg"
                        else:
                            pitch = "Outside Off"
                    else:
                        pitch = "Inline"
                    break

        if len(pitch) == 0:
            pitch = "Inline"
        if pitch == "Outside Leg":
            decision = "NOT OUT"
            if bat_contact_flag == True:
                reason += " and "
            reason += "Ball pitched outside Leg stump"

        if bat_contact_flag == False:
            # Check if impact is outside the line of stumps or not
            if leg_contact_position == None:
                impact = "Inline"
            else:
                if leg_contact_position[0] >= left_stump_x and leg_contact_position[0] <= right_stump_x:
                    impact = "Inline"
                else:
                    impact = "Outside Line"

                if impact == "Outside Line":
                    decision = "NOT OUT"
                    reason = "Impact outside the line of stumps"

            # Check if ball is hitting the stumps or not
            if hitting_stumps == True:
                decision = "OUT"
                reason = "Ball hitting the stumps"
            else:
                decision = "NOT OUT"
                reason = "Ball not hitting the stumps"
        else:
            impact = "No Contact"

        output = {
            "Decision": decision,
            "Reason": reason,
            "BallPitch": pitch,
            "BallPitchPoint": pitch_point,
            "PadImpact": impact,
            "PadImpactPoint": leg_contact_position,
            "HittingStumps": hitting_stumps,
            "BatEdge": loaded_batEdge,
            "PredictedTraj": loaded_predictedTraj
        }

        # Return the result as a JSON response
        return jsonify(output)

    except Exception as e:
        # Return error message in case of an exception
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



#{
  #"batEdge": "{\"collision\": {\"collision\": false}, \"trajectory_prediction\": {\"steps\": [[0,0,1.5],[0.1,0.5,0.8],[0.2,1.0,0.3],[0.3,1.5,0.6]]}, \"field_setup\": {\"stumps_position\": [[-0.15, 0, 0], [0.15, 0, 0]], \"batsman_orientation\": \"R\"}}",
 # "predictedTraj": "{\"verdict\": {\"will_hit_stumps\": true}, \"leg_contact_position\": [0.05, 1.2, 0.2]}"
#}

