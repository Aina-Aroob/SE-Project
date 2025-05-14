from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/decision', methods=['POST'])
def decision_module():
    try:
        # Get data from JSON body
        data = request.get_json()
        batEdge = data.get("batEdge")
        predictedTraj = data.get("predictedTraj")
        
        #Load Data from Json
        loaded_batEdge = json.loads(batEdge)
        loaded_predictedTraj = json.loads(predictedTraj)
        #intiliaze variables    
    
        pitch = ""
        pitch_point= None
        leg_contact_position = None
        leg_contact_point= None
        decision = None 
        reason = ""
        hitting_stumps = loaded_predictedTraj["will_hit_stumps"]
        hitting_stumps_point= None
        
        #Bat Contact Point and flag
        bat_contact_flag = loaded_batEdge["collision"]["collision"] #bat contact flag
        #
        leg_contact_flag= loaded_predictedTraj["will_hit_leg"]
        if leg_contact_flag and not bat_contact_flag:
            leg_contact_point= loaded_predictedTraj["leg_impact_location"] #no data from module 4
        #
        if bat_contact_flag== True:
            decision = "NOT OUT"
            reason = "Ball hit the bat"
    
        if bat_contact_flag == False:
            #Stumps Line
            left_stump_x = loaded_batEdge["field_setup"]["stumps_position"][0][0]   #line of stumps left bound
            right_stump_x = loaded_batEdge["field_setup"]["stumps_position"][1][0]   #  # .. right bound
            #Batsman Orientation
            batsman_orientation = loaded_batEdge["field_setup"]["batsman_orientation"] #batsman orientation
            if batsman_orientation == 'U':
                batsman_orientation = 'R'
            
            #Ball Trajectory
            ball_trajectory_before=loaded_batEdge["previous_trajectory"]
            #
            #BALL Pitch POSITION
            if pitch_point == None:
                for i in range(1,len(ball_trajectory_before)):
                    if ball_trajectory_before[i][1]>=ball_trajectory_before[i-1][1]:
                        pitch_point=ball_trajectory_before[i]
                        if(pitch_point[0] < left_stump_x):
                            if(batsman_orientation == 'R'):
                                pitch = "Outside Off"
                            else:
                                pitch= "Outside Leg"
                        elif(pitch_point[0] > right_stump_x):
                            if(batsman_orientation == 'R'):
                                pitch = "Outside Leg"
                            else:
                                pitch = "Outside Off"
                        else:
                            pitch = "Inline"
                        break
            if(pitch == "Outside Leg"):
                decision = "NOT OUT"
                if(bat_contact_flag == True):
                    reason += " and "
                reason += "Ball pitched outside Leg stump"
                
            #LEG IMPACT
            if(leg_contact_flag==True):
                if(leg_contact_point[0] >= left_stump_x and leg_contact_point[0] <= right_stump_x):
                    leg_contact_position = "Inline"
                else:
                    leg_contact_position = "Outside Line"
                
                if(not pitch):
                    if(leg_contact_point[0] < left_stump_x):
                            if(batsman_orientation == 'R'):
                                pitch = "Outside Off"
                            else:
                                pitch= "Outside Leg"
                    elif(leg_contact_point[0] > right_stump_x):
                        if(batsman_orientation == 'R'):
                            pitch = "Outside Leg"
                        else:
                            pitch = "Outside Off"
                    else:
                        pitch = "Inline"
            else:
                leg_contact_position="Pads Missing"
                    
            if(leg_contact_position == "Outside Line"):
                decision="NOT OUT"
                reason="Impact on Pad Outside Line"
    
            if (decision != "NOT OUT"):
                if(hitting_stumps == True):
                    #hitting_stumps_point = [loaded_predictedTraj["stump_impact"]["impact_point"]["x"],loaded_predictedTraj["stump_impact"]["impact_point"]["y"],loaded_predictedTraj["stump_impact"]["impact_point"]["z"]]
                    decision = "OUT"
                    reason = "Ball hitting the stumps"
                else:
                    decision = "NOT OUT"
                    reason = "Ball not hitting the stumps"
        
        output ={
            "Decision": decision,
            "Reason": reason,
            "BallPitch": pitch,
            "BallPitchPoint": pitch_point,
            "PadImpact": leg_contact_position,
            "PadImpactPoint": leg_contact_point,
            "HittingStumps": hitting_stumps,
            "HittingStumpsPoint": hitting_stumps_point,
            "BatEdge": loaded_batEdge,
            "PredictedTraj": loaded_predictedTraj
        }
        # with open('DecisionMakingModuleOutput.json', 'w') as file:
        #     json.dump(output, file, indent=4)
        # return json.dumps(output)
        return jsonify(output)

    except Exception as e:
        # Return error message in case of an exception
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
