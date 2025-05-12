import json

###Data Formats


# Predicted Trajectory Module
predictedTraj = {
    "verdict": {"status": "Out",
               "will_hit_stumps": True,
                 "impact_region": "middle",
                 "confidence": 0.85
                 },
}
predictedTraj = json.dumps(predictedTraj)

#############
#Bat Edge Module
batEdge = { "collision": { 
                            "collision": True,
                            "confidence": "high",
                            "spatial_detection": {  "collision": True,
                                                    "distance": 10.23,
                                                    "collision_point": [982.5, 644.3, 8.7],
                                                     "bat_obb": {  } 
                                                },
                            "audio_detection": {  },
                            "method": "spatial", "details": "Collision detected by spatial analysis only"
                        }, 
            "trajectory": { 
                            "updated": True,
                            "previous_velocity": [5.2, -2.7, 0.6],
                            "velocity": [-3.98, -2.16, 6.84],
                            "speed": 8.2,
                            "direction": [-0.49, -0.26, 0.83],
                            "collision_point": [982.5, 644.3, 8.7] 
                            },
            "field_setup": { 
                        "stumps_position": [[931, 691, 2432.2], [940, 691, 2432.2], [940, 750, 2432.2], [931, 750, 2432.2]],
                        "batsman_orientation": "L" 
            },
            "trajectory_prediction": { 
                "steps": [ [952, 67, 3.656], [976, 161, 3.957],[1040, 435, 3.957],[1056, 506, 3.957],[1455, 807, 6.116]],
                "history_steps": 22, 
                "future_steps": 10, "starting_from": "collision_point" 
            } 
}

batEdge= json.dumps(batEdge)
#########


#############

def Decision_Making_Module(batEdge,predictedTraj):
    #Load Data from Json
    loaded_batEdge = json.loads(batEdge)
    loaded_predictedTraj = json.loads(predictedTraj)
    #intiliaze variables    

    pitch = ""
    pitch_point= None
    impact = ""
    decision = "" 
    reason = ""
    hitting_stumps = loaded_predictedTraj["verdict"]["will_hit_stumps"]

    #Bat Contact Point and flag
    bat_contact_flag = loaded_batEdge["collision"]["collision"] #bat contact flag
    
    if bat_contact_flag== True:
        decision = "NOT OUT"
        reason = "Ball hit the bat"

    #Stumps Line
    ball_trajectory = loaded_batEdge["trajectory_prediction"]["steps"] #list of ball trajectory points
    left_stump_x = loaded_batEdge["field_setup"]["stumps_position"][0][0]   #line of stumps left bound
    right_stump_x = loaded_batEdge["field_setup"]["stumps_position"][1][0]   #  # .. right bound
    stumps_position_z = loaded_batEdge["field_setup"]["stumps_position"][0][2] #z point
    #Batsman Orientation
    batsman_orientation = loaded_batEdge["field_setup"]["batsman_orientation"] #batsman orientation
    if batsman_orientation == 'U':
        batsman_orientation = 'R'
    
    #
    leg_contact_position= None #no data from module 4
    #

    #check if ball is pitched outside the line of stumps or not
    for i in range((1,len(ball_trajectory))):
        if(ball_trajectory[i][2] <= stumps_position_z):
            #check if ball bounces or not
            if(ball_trajectory[i][1] >= ball_trajectory[i-1][1]): #change of y signals bounce
                pitch_point = ball_trajectory[i]
                if(ball_trajectory[i][0] < left_stump_x):
                    if(batsman_orientation == 'R'):
                        pitch = "Outside Off"
                    else:
                        pitch= "Outside Leg"
                elif(ball_trajectory[i][0] > right_stump_x):
                    if(batsman_orientation == 'R'):
                        pitch = "Outside Leg"
                    else:
                        pitch = "Outside Off"
                else:
                    pitch = "Inline "
                break
            
    if(len(pitch) == 0):
        pitch = "Inline"
    if(pitch == "Outside Leg"):
        decision = "NOT OUT"
        if(bat_contact_flag == True):
            reason += " and "
        reason += "Ball pitched outside Leg stump"
    
    if (bat_contact_flag == False):
        #check if impact is outside the line of stumps or not
        if(leg_contact_position == None):
            impact="Inline"
        else:
            if(leg_contact_position[0] >= left_stump_x and leg_contact_position[0] <= right_stump_x):
                impact = "Inline"
            else:
                impact = "Outside Line"

            if(impact == "Outside Line"):
                decision = "NOT OUT"
                reason = "Impact outside the line of stumps"

        #check if ball is hitting the stumps or not
        if(hitting_stumps == True):
            decision = "OUT"
            reason = "Ball hitting the stumps"
        else:
            decision = "NOT OUT"
            reason = "Ball not hitting the stumps"
    else:
        impact = "No Contact"
    
    output ={
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
    return json.dumps(output)
    



########### Testing



#def analyze_pitch_position(ball_trajectory, stumps_position, batsman_orientation):
 #   left_stump_x = stumps_position[0][0]
  #  right_stump_x = stumps_position[1][0]
   # stumps_z = stumps_position[0][2]
    
    #for i in range(1, len(ball_trajectory)):
     #   if ball_trajectory[i][2] <= stumps_z:
      #      if ball_trajectory[i][1] >= ball_trajectory[i-1][1]:  # Bounce detected
       #         pitch_point = ball_trajectory[i]
        #        x_pos = ball_trajectory[i][0]
                
         #       if x_pos < left_stump_x:
          #          return "Outside Off" if batsman_orientation == 'R' else "Outside Leg", pitch_point
           #     elif x_pos > right_stump_x:
            #        return "Outside Leg" if batsman_orientation == 'R' else "Outside Off", pitch_point
             #   else:
              #      return "Inline", pitch_point
    #return "Inline", None


#if __name__ == "__main__":
 #   print( Decision_Making_Module(batEdge,predictedTraj ))


#def analyze_impact_position(leg_contact_position, stumps_position):
    #if not leg_contact_position:
     #   return "Inline"
    #x_impact = leg_contact_position[0]
    #left_stump_x = stumps_position[0][0]
   # right_stump_x = stumps_position[1][0]
    
  #  if left_stump_x <= x_impact <= right_stump_x:
 #       return "Inline"
#    return "Outside Line"

