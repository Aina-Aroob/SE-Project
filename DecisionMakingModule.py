import json

###Data Formats


# Predicted Trajectory Module
predictedTraj = {
    "verdict": {
        "status": "Out",
        "will_hit_stumps": True,
        "impact_region": "middle",
        "confidence": 0.85
    },
    "leg_contact_position": (1, 4, 5),
    "batsman_orientation": "R"
}
predictedTraj = json.dumps(predictedTraj)

#############
#Bat Edge Module
batEdge = { 
    #"original_trajectory": ,
    #"leg_position_data": ,
    "decision_flag": [True, None],
    "original_trajectory": [(0,0,0),(1,1,1),(2,2,2),(3,3,3),(4,4,4),(5,5,5),(6,6,6),(7,7,7),(8,8,8),(9,9,9),(10,10,10)],
    "stumps": [
       :"corners"[ {"x": 0, "y": 0, "z": 0},
        {"x": 1, "y": 1, "z": 1},
        {"x": 2, "y": 2, "z": 2}
    ],
    ]}
batEdge= json.dumps(batEdge)
#########


#############

def Decision_Making_Module(batEdge,predictedTraj):
    #Load Data from Json
    loaded_batEdge = json.loads(batEdge)
    loaded_predictedTraj = json.loads(predictedTraj)
    #intiliaze variables    
    pitch = ""
    impact = ""
    output ={
        "decision" : "" ,
        "Reason" : "",
    }

    if loaded_batEdge["decision_flag"][0] == True:
        output["decision"] = "NOT OUT"
        output["Reason"] = "Ball hit the bat"
    else:
        #check pitch of ball
        ball_trajectory = loaded_batEdge["original_trajectory"]
        left_stump_x = loaded_batEdge["stumps"]["corners"][0][0]   #line of stumps left bound
        right_stump_x = loaded_batEdge["stumps"]["corners"][1][0]  # .. right bound

        #assumed
        batsman_orientation = 'R' #Right Handed batsman dummy
        leg_contact_position= (1,4,5) #dummy leg contact position
        verdict= {"status": "Out", "will_hit_stumps": True, "impact_region": "middle","confidence": 0.85} #dummy 
        #

        #check if ball is pitched outside the line of stumps or not
        for i in range((1,len(ball_trajectory))):
            #check if ball bounces or not
            if(ball_trajectory[i][1] >= ball_trajectory[i-1][1]):
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
                    pitch = "InLine "
                break
        if(len(pitch) == 0):
            pitch = "InLine"
        if(pitch == "Outside Leg"):
            output["decision"] = "NOT OUT"
            output["Reason"] = "Ball pitched outside Leg stump"
        
        #check if impact is outside the line of stumps or not
        
        if(leg_contact_position[0] >= left_stump_x and leg_contact_position[0] <= right_stump_x):
            impact = "InLine"
        else:
            impact = "Outside Line"

        if(impact == "Outside Line"):
            output["decision"] = "NOT OUT"
            output["Reason"] = "Impact outside the line of stumps"

        #check if ball is hitting the stumps or not
        if[verdict["will_hit_stumps"] == True]:
            output["decision"] = "OUT"
            output["Reason"] = "Ball hitting the stumps"
        else:
            output["decision"] = "NOT OUT"
            output["Reason"] = "Ball not hitting the stumps"
            
    return json.dumps(output)
    



########### Testing

if __name__ == "__main__":
    print( Decision_Making_Module(batEdge,predictedTraj ))



# """
# Sample Output:
#
# {
#   "Decision": "OUT",
#   "Reason": "Ball would have hit the stumps",
#   "BallPitch": "InLine",
#   "BallPitchPoint": [1, 1, 1],
#   "PadImpact": "InLine",
#   "PadImpactPoint": [1, 4, 5],
#   "HittingStumps": true,
#   "batsman_orientation": "R",
#   "batEdge": {
#     "decision_flag": [false, null],
#     "original_trajectory": [
#       [0, 0, 0],
#       [1, 1, 1],
#       [2, 2, 2],
#       [3, 3, 3]
#     ],
#     "stumps": [
#       {"x": 0, "y": 0, "z": 0},
#       {"x": 1, "y": 1, "z": 1},
#       {"x": 2, "y": 2, "z": 2}
#     ]
#   },
#   "predictedTraj": {
#     "verdict": {
#       "status": "Out",
#       "will_hit_stumps": true,
#       "impact_region": "middle",
#       "confidence": 0.85
#     },
#     "leg_contact_position": [1, 4, 5],
#     "batsman_orientation": "R"
#   }
# }
# """



# Output of the program when run
# The function Decision_Making_Module takes a JSON string with decision_flag = [True, None]
# Since decision_flag[0] is True, the output is set to "NOT OUT" with reason "Ball hit the bat"

#print( Decision_Making_Module(batEdge) )
# Output:
# {
#     "decision": "NOT OUT",
#     "Reason": "Ball hit the bat"
# }
# {"decision": "NOT OUT", "Reason": "Ball hit the bat"}


#-------------------------------------------------------#

# import json
# 
# ### Data Formats (Genericized Using Provided Structs)
# 
# # Predicted Trajectory Module
# predictedTraj = {
#     "verdict": {
#         "status": "Out",
#         "will_hit_stumps": True,
#         "impact_region": "middle",
#         "confidence": 0.85
#     },
#     "leg_contact_position": (1, 4, 5),
#     "batsman_orientation": "R"
# }
# predictedTraj = json.dumps(predictedTraj)
# 
# # Bat Edge Module
# batEdge = {
#     "decision_flag": [False, None],
#     "original_trajectory": [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5),
#                              (6, 6, 6), (7, 7, 7), (8, 8, 8), (9, 9, 9), (10, 10, 10)],
#     "stumps": [
#         {"x": 0, "y": 0, "z": 0},
#         {"x": 1, "y": 1, "z": 1},
#         {"x": 2, "y": 2, "z": 2}
#     ]
# }
# batEdge = json.dumps(batEdge)
# 
# # Collision Detection Module
# collision = {
#     "collision": True,
#     "distance": 8.0,
#     "collision_point": [470.0, 890.0, 14.0],
#     "bat_obb": {
#         "center": [475.0, 930.0, 30.0],
#         "basis": [
#             [1.0, 0.0, 0.0],
#             [0.0, 0.9805806756909201, 0.19611613513818404],
#             [0.0, 0.0, 1.0]
#         ],
#         "half_size": [25.0, 50.99019513592785, 10.0]
#     },
#     "confidence": "high",
#     "method": "spatial",
#     "details": "Ball intersects bat by 3.00 units"
# }
# 
# #############
# 
# def Decision_Making_Module(batEdge, predictedTraj):
#     loaded_batEdge = json.loads(batEdge)
#     loaded_predictedTraj = json.loads(predictedTraj)
# 
#     pitch = ""
#     impact = ""
#     output = {
#         "Decision": "",
#         "Reason": "",
#         "BallPitch": "",
#         "BallPitchPoint": None,
#         "PadImpact": "",
#         "PadImpactPoint": loaded_predictedTraj["leg_contact_position"],
#         "HittingStumps": loaded_predictedTraj["verdict"]["will_hit_stumps"],
#         "batsman_orientation": loaded_predictedTraj.get("batsman_orientation", "R")
#     }
# 
#     if collision["collision"]:
#         output["Decision"] = "NOT OUT"
#         output["Reason"] = "Ball hit the bat"
#         return json.dumps(output, indent=2)
# 
#     ball_trajectory = loaded_batEdge["original_trajectory"]
#     left_stump_x = loaded_batEdge["stumps"][0]["x"]
#     right_stump_x = loaded_batEdge["stumps"][2]["x"]
# 
#     batsman_type = output["batsman_orientation"]
#     leg_contact_position = loaded_predictedTraj["leg_contact_position"]
#     verdict = loaded_predictedTraj["verdict"]
# 
#     for i in range(1, len(ball_trajectory)):
#         if ball_trajectory[i][1] >= ball_trajectory[i - 1][1]:
#             pitch_point = ball_trajectory[i]
#             break
#     else:
#         pitch_point = ball_trajectory[0]
# 
#     pitch_x = pitch_point[0]
#     output["BallPitchPoint"] = pitch_point
# 
#     if pitch_x < left_stump_x:
#         output["BallPitch"] = "Outside Off" if batsman_type == "R" else "Outside Leg"
#     elif pitch_x > right_stump_x:
#         output["BallPitch"] = "Outside Leg" if batsman_type == "R" else "Outside Off"
#     else:
#         output["BallPitch"] = "InLine"
# 
#     if output["BallPitch"] == "Outside Leg":
#         output["Decision"] = "NOT OUT"
#         output["Reason"] = "Ball pitched outside leg stump"
#         return json.dumps(output, indent=2)
# 
#     if left_stump_x <= leg_contact_position[0] <= right_stump_x:
#         output["PadImpact"] = "InLine"
#     else:
#         output["PadImpact"] = "Outside Line"
# 
#     if output["PadImpact"] == "Outside Line":
#         output["Decision"] = "NOT OUT"
#         output["Reason"] = "Impact outside the line of stumps"
#     elif verdict["will_hit_stumps"]:
#         output["Decision"] = "OUT"
#         output["Reason"] = "Ball would have hit the stumps"
#     else:
#         output["Decision"] = "NOT OUT"
#         output["Reason"] = "Ball missing the stumps"
# 
#     return json.dumps(output, indent=2)
# 
# ########### Testing
# 
# if __name__ == "__main__":
#     print(Decision_Making_Module(batEdge, predictedTraj))

#above code SAMPLE OUTPUT:
# Sample Output:
# {
#   "Decision": "OUT",
#   "Reason": "Ball would have hit the stumps",
#   "BallPitch": "InLine",
#   "BallPitchPoint": [1, 1, 1],
#   "PadImpact": "InLine",
#   "PadImpactPoint": [1, 4, 5],
#   "HittingStumps": true,
#   "batsman_orientation": "R",
#   "batEdge": {
#     "decision_flag": [false, null],
#     "original_trajectory": [
#       [0, 0, 0],
#       [1, 1, 1],
#       [2, 2, 2],
#       [3, 3, 3],
#       [4, 4, 4],
#       [5, 5, 5],
#       [6, 6, 6],
#       [7, 7, 7],
#       [8, 8, 8],
#       [9, 9, 9],
#       [10, 10, 10]
#     ],
#     "stumps": [
#       {"x": 0, "y": 0, "z": 0},
#       {"x": 1, "y": 1, "z": 1},
#       {"x": 2, "y": 2, "z": 2}
#     ]
#   },
#   "predictedTraj": {
#     "verdict": {
#       "status": "Out",
#       "will_hit_stumps": true,
#       "impact_region": "middle",
#       "confidence": 0.85
#     },
#     "leg_contact_position": [1, 4, 5],
#     "batsman_orientation": "R"
#   }
# }
#


# Importing the json module to handle JSON data
#import json

### Data Formats

# Creating a dictionary to represent the batEdge module
#batEdge = { 
    # 'original_trajectory' and 'leg_position_data' are commented out or unused here
    # "original_trajectory": ,
    # "leg_position_data": ,
  #  "decision_flag": [True, None]  # A list with the first value True indicating ball hit the bat
#}

# Converting the dictionary to a JSON-formatted string
#batEdge = json.dumps(batEdge)

#############

# Defining the Decision_Making_Module function which takes batEdge data as input
#def Decision_Making_Module(batEdge):
    # Load data from the JSON string into a Python dictionary
 #   loaded_batEdge = json.loads(batEdge)

    # Initialize a dictionary to store the output decision and reason
  #  output = {
   #     "decision": "",  # Initially empty decision
    #    "Reason": "",    # Initially empty reason
    #}

    # Check if the first element of decision_flag is True
    #if loaded_batEdge["decision_flag"][0] == True:
        # If true, set decision to "NOT OUT"
     #   output["decision"] = "NOT OUT"
        # Set the reason for the decision
      #  output["Reason"] = "Ball hit the bat"
    
    # Return the output dictionary as a JSON string
    #return json.dumps(output)

########### Testing

# If this file is run directly (not imported), run the test
#if __name__ == "__main__":
    # Call the Decision_Making_Module with batEdge input and print the result
    #print(Decision_Making_Module(batEdge))
