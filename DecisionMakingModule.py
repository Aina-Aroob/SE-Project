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
    "batsman_type": "RH"
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
        {"x": 0, "y": 0, "z": 0},
        {"x": 1, "y": 1, "z": 1},
        {"x": 2, "y": 2, "z": 2}
    ],
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
        left_stump_x = loaded_batEdge["stumps"][0]["x"]   #line of stumps left bound
        right_stump_x = loaded_batEdge["stumps"][2]["x"]  # .. right bound

        #assumed
        batsman_type = 'RH' #Right Handed batsman dummy
        leg_contact_position= (1,4,5) #dummy leg contact position
        verdict= {"status": "Out", "will_hit_stumps": True, "impact_region": "middle","confidence": 0.85} #dummy 
        #

        #check if ball is pitched outside the line of stumps or not
        for i in range((1,len(ball_trajectory))):
            #check if ball bounces or not
            if(ball_trajectory[i][1] >= ball_trajectory[i-1][1]):
                if(ball_trajectory[i][0] < left_stump_x):
                    if(batsman_type == 'RH'):
                        pitch = "Outside Off"
                    else:
                        pitch= "Outside Leg"
                elif(ball_trajectory[i][0] > right_stump_x):
                    if(batsman_type == 'RH'):
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
    print( Decision_Making_Module(batEdge) )
