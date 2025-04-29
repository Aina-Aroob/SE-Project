import json

###Data Formats


#Bat Edge Module
batEdge = { 
    #"original_trajectory": ,
    #"leg_position_data": ,
    "decision_flag": [True, None]
}
batEdge= json.dumps(batEdge)
#########





#############

def Decision_Making_Module(batEdge):
    #Load Data from Json
    loaded_batEdge = json.loads(batEdge)

    #intiliaze variables
    output ={
        "decision" : "" ,
        "Reason" : "",
    }

    if loaded_batEdge["decision_flag"][0] == True:
        output["decision"] = "NOT OUT"
        output["Reason"] = "Ball hit the bat"
    
    return json.dumps(output)
    



########### Testing

if __name__ == "__main__":
    print( Decision_Making_Module(batEdge) )
