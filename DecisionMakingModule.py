import json

###Data Formats
# predTraj={
# "is_lbw": false,
# "confidence_score": 0.5861000903702026,
# "trajectory_analysis": {
# "original_points": [
# {
# "x": 707.0,
# "y": 312.0,
# "z": 6.657804685443537,
# "t": 0.0
# },
# {
# "x": 739.0,
# "y": 205.0,
# "z": 7.402829268292683,
# "t": 0.1
# },
# {
# "x": 761.0,
# "y": 129.0,
# "z": 7.7122601956549355,
# "t": 0.2
# },
# {
# "x": 782.9999999999999,
# "y": 70.0,
# "z": 8.882788492493196,
# "t": 0.30000000000000004
# },
# {
# "x": 801.0,
# "y": 27.0,
# "z": 9.197454545454544,
# "t": 0.4
# },
# {
# "x": 820.0,
# "y": 8.0,
# "z": 9.484874999999999,
# "t": 0.5
# },
# {
# "x": 833.0000000000001,
# "y": 8.0,
# "z": 13.190039546303941,
# "t": 0.6000000000000001
# },
# {
# "x": 847.0,
# "y": 0.0,
# "z": 13.289955337595234,
# "t": 0.7000000000000001
# },
# {
# "x": 853.9999999999999,
# "y": 5.0,
# "z": 11.568684250647962,
# "t": 0.8
# },
# {
# "x": 862.0,
# "y": 23.999999999999996,
# "z": 10.839857142857142,
# "t": 0.9
# },
# {
# "x": 871.0,
# "y": 51.0,
# "z": 14.453142857142856,
# "t": 1.0
# },
# {
# "x": 880.0,
# "y": 85.0,
# "z": 13.796181818181818,
# "t": 1.1
# },
# {
# "x": 884.0,
# "y": 129.0,
# "z": 15.974526315789472,
# "t": 1.2000000000000002
# },
# {
# "x": 889.0,
# "y": 177.0,
# "z": 15.175799999999999,
# "t": 1.3
# },
# {
# "x": 895.0000000000001,
# "y": 228.99999999999997,
# "z": 17.853882352941174,
# "t": 1.4000000000000001
# },
# {
# "x": 898.0000000000001,
# "y": 288.0,
# "z": 16.862,
# "t": 1.5
# },
# {
# "x": 902.0,
# "y": 354.0,
# "z": 17.853882352941174,
# "t": 1.6
# },
# {
# "x": 904.0000000000001,
# "y": 418.00000000000006,
# "z": 17.853882352941174,
# "t": 1.7000000000000002
# },
# {
# "x": 906.0,
# "y": 488.0,
# "z": 18.969749999999998,
# "t": 1.8
# },
# {
# "x": 906.0,
# "y": 562.0,
# "z": 18.969749999999998,
# "t": 1.9000000000000001
# },
# {
# "x": 911.0,
# "y": 640.9999999999999,
# "z": 21.679714285714283,
# "t": 2.0
# },
# {
# "x": 909.0,
# "y": 714.9999999999999,
# "z": 20.234399999999997,
# "t": 2.1
# },
# {
# "x": 909.0,
# "y": 798.0,
# "z": 20.234399999999997,
# "t": 2.2
# },
# {
# "x": 914.0,
# "y": 840.0,
# "z": 20.234399999999997,
# "t": 2.3000000000000003
# },
# {
# "x": 929.0,
# "y": 775.0,
# "z": 21.679714285714283,
# "t": 2.4000000000000004
# },
# {
# "x": 942.0,
# "y": 717.0,
# "z": 20.234399999999997,
# "t": 2.5
# },
# {
# "x": 955.0,
# "y": 664.0,
# "z": 21.679714285714283,
# "t": 2.6
# },
# {
# "x": 969.0000000000001,
# "y": 633.0,
# "z": 21.679714285714283,
# "t": 2.7
# },
# {
# "x": 976.0,
# "y": 656.0,
# "z": 20.414043583535108,
# "t": 2.8000000000000003
# },
# {
# "x": 977.0,
# "y": 679.0000000000001,
# "z": 21.679714285714283,
# "t": 2.9000000000000004
# },
# {
# "x": 979.0,
# "y": 707.0,
# "z": 20.234399999999997,
# "t": 3.0
# },
# {
# "x": 982.0,
# "y": 743.0,
# "z": 16.862,
# "t": 3.1
# },
# {
# "x": 976.0,
# "y": 797.0,
# "z": 11.673692307692306,
# "t": 3.2
# },
# {
# "x": 973.0,
# "y": 784.0,
# "z": 13.050522423356409,
# "t": 3.3000000000000003
# },
# {
# "x": 969.0000000000001,
# "y": 767.9999999999999,
# "z": 18.097668594597813,
# "t": 3.4000000000000004
# },
# {
# "x": 965.0,
# "y": 761.0,
# "z": 23.347384615384613,
# "t": 3.5
# },
# {
# "x": 958.0,
# "y": 753.0000000000001,
# "z": 20.234399999999997,
# "t": 3.6
# },
# {
# "x": 959.0,
# "y": 758.0,
# "z": 17.853882352941174,
# "t": 3.7
# },
# {
# "x": 955.0,
# "y": 757.0,
# "z": 22.786486486486485,
# "t": 3.8000000000000003
# },
# {
# "x": 954.0,
# "y": 767.9999999999999,
# "z": 18.969749999999998,
# "t": 3.9000000000000004
# }
# ],
# "predicted_points": [
# {
# "x": 952.7500000000001,
# "y": 769.5344488188975,
# "z": 19.248716911764703,
# "t": 3.95
# },
# {
# "x": 951.5000000000001,
# "y": 770.1033464566929,
# "z": 19.527683823529408,
# "t": 4.0
# },
# {
# "x": 950.2500000000002,
# "y": 769.7066929133857,
# "z": 19.806650735294113,
# "t": 4.05
# },
# {
# "x": 949.0000000000002,
# "y": 768.3444881889762,
# "z": 20.085617647058818,
# "t": 4.1
# },
# {
# "x": 947.7500000000002,
# "y": 766.0167322834644,
# "z": 20.364584558823523,
# "t": 4.1499999999999995
# },
# {
# "x": 946.5000000000003,
# "y": 762.7234251968501,
# "z": 20.64355147058823,
# "t": 4.199999999999999
# },
# {
# "x": 945.2500000000003,
# "y": 758.4645669291336,
# "z": 20.922518382352933,
# "t": 4.249999999999999
# },
# {
# "x": 944.0000000000005,
# "y": 753.2401574803148,
# "z": 21.20148529411764,
# "t": 4.299999999999999
# },
# {
# "x": 942.7500000000005,
# "y": 747.0501968503935,
# "z": 21.480452205882347,
# "t": 4.349999999999999
# },
# {
# "x": 941.5000000000005,
# "y": 739.8946850393698,
# "z": 21.759419117647052,
# "t": 4.399999999999999
# },
# {
# "x": 940.2500000000006,
# "y": 731.7736220472437,
# "z": 22.038386029411758,
# "t": 4.449999999999998
# },
# {
# "x": 939.0000000000006,
# "y": 722.6870078740153,
# "z": 22.317352941176463,
# "t": 4.499999999999998
# },
# {
# "x": 937.7500000000007,
# "y": 712.6348425196846,
# "z": 22.596319852941168,
# "t": 4.549999999999998
# },
# {
# "x": 936.5000000000007,
# "y": 701.6171259842516,
# "z": 22.875286764705873,
# "t": 4.599999999999998
# },
# {
# "x": 935.2500000000007,
# "y": 689.6338582677162,
# "z": 23.154253676470578,
# "t": 4.649999999999998
# },
# {
# "x": 934.0000000000008,
# "y": 676.6850393700784,
# "z": 23.433220588235283,
# "t": 4.6999999999999975
# },
# {
# "x": 932.7500000000008,
# "y": 662.7706692913381,
# "z": 23.712187499999988,
# "t": 4.749999999999997
# },
# {
# "x": 931.5000000000009,
# "y": 647.8907480314955,
# "z": 23.991154411764693,
# "t": 4.799999999999997
# },
# {
# "x": 930.2500000000009,
# "y": 632.0452755905507,
# "z": 24.2701213235294,
# "t": 4.849999999999997
# },
# {
# "x": 929.0000000000009,
# "y": 615.2342519685034,
# "z": 24.549088235294104,
# "t": 4.899999999999997
# }
# ]
# },
# "bounce_analysis": {
# "pos_x": 21.5138,
# "pos_y": 0.0,
# "pos_z": 0.33756486557491894,
# "timestamp": 0.7000000000000001
# },
# "impact_analysis": {
# "bat_collision": {
# "detected": false,
# "point": null,
# "confidence": 0.0
# },
# "stump_impact": {
# "will_hit": false,
# "impact_point": {
# "x": 0,
# "y": 0,
# "z": 0
# }
# }
# },
# "swing_characteristics": {
# "swing_angle": 2.579606779154457,
# "swing_magnitude": 0.5930235692307692,
# "swing_direction": 1.0,
# "swing_rate": 0.66143763568063,
# "swing_consistency": 0.13866748262901102,
# "late_swing": 1.157860895304164
# }
# }
# batEdge={
# "collision": {
# "collision": false,
# "confidence": "none",
# "spatial_detection": {
# "collision": false,
# "distance": 282.271589076903,
# "collision_point": [
# 894.5,
# 523.0,
# 6.657804685443537
# ],
# "bat_obb": {
# "center": [
# 982.0,
# 517.5,
# 5.655577639751553
# ],
# "basis": [
# [
# 0.8846205354008864,
# -0.4663116000562812,
# 0.0
# ],
# [
# 0.6280192682591458,
# 0.7781977889298112,
# 0.0
# ],
# [
# 0.0,
# 0.0,
# 1.0
# ]
# ],
# "half_size": [
# 72.91261893527073,
# 36.62308015445998,
# 10.0
# ]
# },
# "confidence": "none",
# "method": "spatial",
# "details": "Ball is 236.68 units away from bat"
# },
# "audio_detection": {
# "collision": false,
# "confidence": "none",
# "method": "audio",
# "details": {
# "amplitude_spike": false,
# "frequency_match": false,
# "spike_count": 0,
# "max_deviation": 0
# }
# },
# "method": "combined",
# "details": "No collision detected"
# },
# "trajectory": {
# "updated": false,
# "velocity": [
# 0,
# 0,
# 0
# ],
# "details": "No collision detected"
# },
# "field_setup": {
# "stumps_position": [
# [
# 953,
# 783,
# 18.182326415094337
# ],
# [
# 1048,
# 784,
# 18.182326415094337
# ],
# [
# 1058,
# 513,
# 18.182326415094337
# ],
# [
# 952,
# 516,
# 18.182326415094337
# ]
# ],
# "batsman_orientation": "R"
# },
# "previous_trajectory": [
# [
# 707,
# 312,
# 6.657804685443537
# ],
# [
# 739,
# 205,
# 7.402829268292682
# ],
# [
# 761,
# 129,
# 7.7122601956549355
# ],
# [
# 783,
# 70,
# 8.882788492493196
# ],
# [
# 801,
# 27,
# 9.197454545454544
# ],
# [
# 820,
# 8,
# 9.484874999999999
# ],
# [
# 833,
# 8,
# 13.190039546303941
# ],
# [
# 847,
# 0,
# 13.289955337595234
# ],
# [
# 854,
# 5,
# 11.568684250647962
# ],
# [
# 862,
# 24,
# 10.839857142857142
# ],
# [
# 871,
# 51,
# 14.453142857142856
# ],
# [
# 880,
# 85,
# 13.796181818181816
# ],
# [
# 884,
# 129,
# 15.974526315789472
# ],
# [
# 889,
# 177,
# 15.175799999999999
# ],
# [
# 895,
# 229,
# 17.853882352941174
# ],
# [
# 898,
# 288,
# 16.862
# ],
# [
# 902,
# 354,
# 17.853882352941174
# ],
# [
# 904,
# 418,
# 17.853882352941174
# ],
# [
# 906,
# 488,
# 18.969749999999998
# ],
# [
# 906,
# 562,
# 18.969749999999998
# ],
# [
# 911,
# 641,
# 21.679714285714283
# ],
# [
# 909,
# 715,
# 20.234399999999997
# ],
# [
# 909,
# 798,
# 20.234399999999997
# ],
# [
# 914,
# 840,
# 20.234399999999997
# ],
# [
# 929,
# 775,
# 21.679714285714283
# ],
# [
# 942,
# 717,
# 20.234399999999997
# ],
# [
# 955,
# 664,
# 21.679714285714283
# ],
# [
# 969,
# 633,
# 21.679714285714283
# ],
# [
# 976,
# 656,
# 20.414043583535108
# ],
# [
# 977,
# 679,
# 21.679714285714283
# ],
# [
# 979,
# 707,
# 20.234399999999997
# ],
# [
# 982,
# 743,
# 16.862
# ],
# [
# 976,
# 797,
# 11.673692307692306
# ],
# [
# 973,
# 784,
# 13.050522423356407
# ],
# [
# 969,
# 768,
# 18.097668594597813
# ],
# [
# 965,
# 761,
# 23.347384615384613
# ],
# [
# 958,
# 753,
# 20.234399999999997
# ],
# [
# 959,
# 758,
# 17.853882352941174
# ],
# [
# 955,
# 757,
# 22.786486486486485
# ],
# [
# 954,
# 768,
# 18.969749999999998
# ]
# ]
# }


#############

def Decision_Making_Module(batEdge,predictedTraj):
    #Load Data from Json
    loaded_batEdge = json.loads(batEdge)
    loaded_predictedTraj = json.loads(predictedTraj)
    #intiliaze variables    

    pitch = ""
    pitch_point= None
    impact = ""
    decision = "OUT" 
    reason = ""
    hitting_stumps = loaded_predictedTraj["impact_analysis"]["stump_impact"]["will_hit"]
    hitting_stumps_point= None

    #Bat Contact Point and flag
    bat_contact_flag = loaded_batEdge["collision"]["collision"] #bat contact flag
    
    if bat_contact_flag== True:
        decision = "NOT OUT"
        reason = "Ball hit the bat"
    elif(loaded_predictedTraj["is_lbw"]==False):
        decision = "NOT OUT"
        reason = "Ball not touching the Pads"

    #Stumps Line
    left_stump_x = loaded_batEdge["field_setup"]["stumps_position"][0][0]   #line of stumps left bound
    right_stump_x = loaded_batEdge["field_setup"]["stumps_position"][1][0]   #  # .. right bound
    #Batsman Orientation
    batsman_orientation = loaded_batEdge["field_setup"]["batsman_orientation"] #batsman orientation
    if batsman_orientation == 'U':
        batsman_orientation = 'R'
    
    #
    leg_contact_position= None #no data from module 4
    #
    if(loaded_predictedTraj.get("bounce_analysis")):
        pitch_point =[loaded_predictedTraj["bounce_analysis"]["pos_x"],loaded_predictedTraj["bounce_analysis"]["pos_y"],loaded_predictedTraj["bounce_analysis"]["pos_z"]]        
    
    #check if ball is pitched outside the line of stumps or not
    if pitch_point == None:
        if(bat_contact_flag == True):
            if(loaded_batEdge["collision"]["spatial_detection"]["collision_point"][0] < left_stump_x):
                pitch= "Outside Off"
            elif(loaded_batEdge["collision"]["spatial_detection"]["collision_point"][0] > right_stump_x):
                pitch = "Outside Leg"
            else:
                pitch = "Inline"
        else:
            pitch = "Inline"
    else:
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

    if(pitch == "Outside Leg"):
        decision = "NOT OUT"
        if(bat_contact_flag == True):
            reason += " and "
        reason += "Ball pitched outside Leg stump"
    # if (bat_contact_flag == False and pitch != "Outside Leg"):
        #check if impact is outside the line of stumps or not
        # if(leg_contact_position == None):
        #     impact="Inline"
        # else:
        #     if(leg_contact_position[0] >= left_stump_x and leg_contact_position[0] <= right_stump_x):
        #         impact = "Inline"
        #     else:
        #         impact = "Outside Line"

        #     if(impact == "Outside Line"):
        #         decision = "NOT OUT"
        #         reason = "Impact outside the line of stumps"
        #check if ball is hitting the stumps or not
    

    if (decision != "NOT OUT"):
        if(hitting_stumps == True):
            hitting_stumps_point = [loaded_predictedTraj["stump_impact"]["impact_point"]["x"],loaded_predictedTraj["stump_impact"]["impact_point"]["y"],loaded_predictedTraj["stump_impact"]["impact_point"]["z"]]
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
        "HittingStumpsPoint": hitting_stumps_point,
        "BatEdge": loaded_batEdge,
        "PredictedTraj": loaded_predictedTraj
    }
    return json.dumps(output)
    



########### Testing

if __name__ == "__main__":
    with open('batEdge.json', 'r') as file:
        batEdge = json.load(file)
    with open('predTraj.json', 'r') as file:
        predTraj = json.load(file)

    batEdge_File = json.dumps(batEdge)
    predTraj_File = json.dumps(predTraj)
    print( Decision_Making_Module(batEdge_File,predTraj_File ))
