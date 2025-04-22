from flask import Flask, request, jsonify, make_response
from bat_detection import detect_bat_edge_contact, update_trajectory
from test_detection import test_no_contact, test_reflection
app = Flask(__name__, debug=True)

@app.route("/detect")
def detect():
    ball_traj = request.args.get('ball_trajectory')
    bat_edges = request.args.get('bat_edges')
    detected, ball_position = detect_bat_edge_contact(ball_traj, bat_edges)
    if(detected == True):
        response = make_response(jsonify({'detected': detected, 'ball_position': ball_position}), 200)
    elif(detected == False):
        response = make_response(jsonify({'detected': detected, 'ball_position': None}), 200)
    return response

@app.route("/test_detect")
def test_detect():
    ball_traj = request.args.get('ball_trajectory')
    bat_edges = request.args.get('bat_edges')
    output = test_no_contact(ball_traj, bat_edges)
    if output == True:
        return "Contact detection test passed"
    else:
        return "Contact detection test failed"
    
@app.route("/update")
def update():
    ball_velocity = request.args.get("ball_velocity")
    bat_normal = request.args.get("bat_normal")
    reflection = update_trajectory(ball_velocity, bat_normal)
    response = make_response(jsonify({'updated_trajectory': reflection}))
    return response

@app.route("/test_update")
def test_update():
    ball_velocity = request.args.get("ball_velocity")
    bat_normal = request.args.get("bat_normal")
    output = test_reflection(ball_velocity, bat_normal)
    if output == True:
        return "Update test succeeded"
    else:
        return "Update test failed"


