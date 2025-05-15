import tkinter as tk
import threading
import requests
import webbrowser
import cv2
import os
import json
from pathlib import Path


#=============== Import custom modules
#from lbw_predictor import LBWPredictor
from DecisionMakingModule import Decision_Making_Module
from ballAndObjectTrackingModule.combined_tracker import track_objects 
from bat_detection import process_input, predict_trajectory
from trajectory_analysis_module.predictor import LBWPredictor
from trajectory_predictor import TrajectoryPredictor
from draw_trajectory_new import TrajectoryOverlayRenderer

SERVER_URL = "http://127.0.0.1:5000"
videopath=os.path.abspath("recordings/updated_video.mp4")
TRAJECTORY_OUTPUT_DIR = Path("trajectory_analysis_output") 

# ------------ Camera Control Functions -------------- #
def start_camera():
    threading.Thread(target=lambda: webbrowser.open(f"{SERVER_URL}/video")).start()

def pause_camera():
    threading.Thread(target=lambda: requests.get(f"{SERVER_URL}/pause")).start()

def resume_camera():
    threading.Thread(target=lambda: requests.get(f"{SERVER_URL}/resume")).start()

def stop_camera():
    threading.Thread(target=lambda: requests.get(f"{SERVER_URL}/stop")).start()

# ------------ Navigation Functions -------------- #
def show_main_menu():
    for frame in all_frames:
        frame.pack_forget()
    main_menu_frame.pack(fill="both", expand=True)

    

def show_camera_controls():
    for frame in all_frames:
        frame.pack_forget()
    camera_controls_frame.pack(fill="both", expand=True)

def show_video_playback_controls():
    for frame in all_frames:
        frame.pack_forget()
    video_controls_frame.pack(fill="both", expand=True)

def show_tracking_controls():
    for frame in all_frames:
        frame.pack_forget()
    tracking_controls_frame.pack(fill="both", expand=True)

# Add this navigation function
def show_bat_detection():
    for frame in all_frames:
        frame.pack_forget()
    bat_detection_frame.pack(fill="both", expand=True)

def show_trajectory_analysis():
    for frame in all_frames:
        frame.pack_forget()
    trajectory_analysis_frame.pack(fill="both", expand=True)

def show_decision_making():
    for frame in all_frames:
        frame.pack_forget()
    decision_making_frame.pack(fill="both", expand=True)

def show_overlay_controls():
    for frame in all_frames:
        frame.pack_forget()
    overlay_controls_frame.pack(fill="both", expand=True)

# ------------ Video Playback -------------- #
def play_video(path):
    def run():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Failed to open video file: {path}")
            return

        max_width = 800
        max_height = 600

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if w > max_width or h > max_height:
                scale = min(max_width / w, max_height / h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            cv2.imshow("Video ", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=run).start()

#=================bat edge func=====================
# First, add this new function for bat edge detection processing
def run_bat_edge_detection():
    bat_status_label.config(text="Starting bat edge detection...", fg="blue")
    
    def task():
        try:
            bat_status_label.config(text="Processing bat edges...", fg="orange")
            
            json_path = os.path.abspath("server files/correct_input.json")
            output_dir = os.path.abspath("bat_detection_output")  # Centralized output location
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Process and save (output_dir triggers saving)
            result = process_input(data)
            
            # Standardized saving in GUI code
            output_dir = os.path.abspath("server files/bat_detection_output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "detection_results.json")
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
            
            bat_status_label.config(
                text=f"Complete!",
                fg="green"
            )
        except Exception as e:
            bat_status_label.config(text=f"System error:\n{str(e)}", fg="red")
    
    threading.Thread(target=task, daemon=True).start()
# ------------ Trajectory Analysis Function -------------- #
def run_trajectory_analysis():
    def task():
        try:
            # Update status
            verdict_label.config(text="Starting trajectory analysis...", fg="blue")
            
            # 1. Set up paths - using your existing structure but pointing to bat detection output
            DATA_DIR = Path("Server files")
            INPUT_DIR = DATA_DIR
            OUTPUT_DIR = Path("Server files")    
            INPUT_FILENAME = "trajectory_input.json"  # From bat detection
            OUTPUT_FILENAME = "trajectory_output.json"
            
            # 2. Load input data (from bat detection output)
            input_path = INPUT_DIR / INPUT_FILENAME
            """Main function to demonstrate trajectory prediction."""
            # Initialize predictor with data file
            predictor = TrajectoryPredictor('Server files/trajectory_input.json')
            
            # Predict trajectory
            trajectory = predictor.predict_trajectory(future_frames=10)
            
            # Print results
            print(f"Predicted trajectory from last frame (frame {predictor.last_frame['frame_id']})")
            print(f"Historical ball positions: {len(trajectory.historical_points)} frames")
            print(f"Predicted ball positions: {len(trajectory.trajectory_points)} frames")
            
            # Detail the impact predictions
            impact_results = []
            if trajectory.will_hit_leg:
                impact_results.append(f"WILL HIT LEG at ({trajectory.leg_impact_location[0]:.2f}, {trajectory.leg_impact_location[1]:.2f}, {trajectory.leg_impact_location[2]:.2f})")
            
            if trajectory.will_hit_stumps and trajectory.will_hit_leg:
                impact_results.append(f"WOULD HIT STUMPS at ({trajectory.impact_location[0]:.2f}, {trajectory.impact_location[1]:.2f}, {trajectory.impact_location[2]:.2f}) - LBW CANDIDATE")
            elif trajectory.will_hit_stumps:
                impact_results.append(f"WILL HIT STUMPS at ({trajectory.impact_location[0]:.2f}, {trajectory.impact_location[1]:.2f}, {trajectory.impact_location[2]:.2f})")
            
            if not impact_results:
                print("Prediction: WILL MISS ALL TARGETS")
            else:
                print("Prediction: " + " & ".join(impact_results))
            
            if trajectory.bounce_point:
                print(f"Bounce point: ({trajectory.bounce_point[0]:.2f}, {trajectory.bounce_point[1]:.2f}, {trajectory.bounce_point[2]:.2f})")
            
            print(f"Swing characteristics:")
            for key, value in trajectory.swing_characteristics.items():
                if isinstance(value, (int, float)):
                    print(f"  - {key}: {value:.2f}")
                else:
                    print(f"  - {key}: {value}")
            
            
            # Prepare output for next module
            output = {
                "previous_trajectory": trajectory.historical_points,
                "predicted_trajectory": trajectory.trajectory_points,
                "bounce_point": trajectory.bounce_point,
                "stump_impact_location": trajectory.impact_location,
                "leg_impact_location": trajectory.leg_impact_location,
                "will_hit_stumps": trajectory.will_hit_stumps,
                "will_hit_leg": trajectory.will_hit_leg,
                "swing_characteristics": trajectory.swing_characteristics
            }
            
            # Add collision information if it exists in the original data
            if 'collision' in predictor.data:
                output["collision"] = predictor.data["collision"]
            
            # Save output to JSON file for next module
            with open('Server files/trajectory_output.json', 'w') as f:
                # Convert tuples to lists for JSON serialization
                json_output = {
                    "previous_trajectory": [[float(x), float(y), float(z)] for x, y, z in output["previous_trajectory"]],
                    "predicted_trajectory": [[float(x), float(y), float(z)] for x, y, z in output["predicted_trajectory"]],
                    "bounce_point": [float(x) for x in output["bounce_point"]] if output["bounce_point"] else None,
                    "stump_impact_location": [float(x) for x in output["stump_impact_location"]] if output["stump_impact_location"] else None,
                    "leg_impact_location": [float(x) for x in output["leg_impact_location"]] if output["leg_impact_location"] else None,
                    "will_hit_stumps": output["will_hit_stumps"],
                    "will_hit_leg": output["will_hit_leg"],
                    "swing_characteristics": {k: float(v) if isinstance(v, (int, float)) else v 
                                            for k, v in output["swing_characteristics"].items()}
                }

                # Add collision from output if it exists
                if "collision" in output:
                    json_output["collision"] = output["collision"]
                    
                json.dump(json_output, f, indent=2)

           
            
        except Exception as e:
            verdict_label.config(text=f"System error in trajectory analysis: {str(e)}", fg="red")

    threading.Thread(target=task, daemon=True).start()

# ------------ Decision Making Function -------------- #

def run_decision_making():
    def task():
        try:
            decision_label.config(text="Starting decision making...", fg="blue")
            
            # Load data from previous modules
            bat_detection_path = Path("Server files/bat_detection_output/detection_results.json")
            trajectory_path = Path("Server files/trajectory_output.json")
            
            # Verify input files exist
            if not bat_detection_path.exists():
                decision_label.config(text="Error: Bat detection data not found", fg="red")
                return
            if not trajectory_path.exists():
                decision_label.config(text="Error: Trajectory data not found", fg="red")
                return
            
            # Load data from files
            with open(bat_detection_path, 'r') as f:
                batEdge = json.load(f)
            with open(trajectory_path, 'r') as f:
                predictedTraj = json.load(f)
            
            # Initialize variables
            pitch = ""
            pitch_point = None
            leg_contact_position = None
            leg_contact_point = None
            decision = None 
            reason = ""
            hitting_stumps = predictedTraj["will_hit_stumps"]
            hitting_stumps_point = None
            
            # Bat Contact Point and flag
            bat_contact_flag = batEdge["collision"]["collision"] if "collision" in batEdge else False
            
            # Leg contact
            leg_contact_flag = predictedTraj["will_hit_leg"]
            if leg_contact_flag and not bat_contact_flag:
                leg_contact_point = predictedTraj["leg_impact_location"]
            
            # Decision logic
            if bat_contact_flag == True:
                decision = "NOT OUT"
                reason = "Ball hit the bat"
            
            if bat_contact_flag == False:
                # Stumps Line
                if "field_setup" in batEdge and "stumps_position" in batEdge["field_setup"]:
                    left_stump_x = batEdge["field_setup"]["stumps_position"][0][0]  # left bound
                    right_stump_x = batEdge["field_setup"]["stumps_position"][1][0]  # right bound
                else:
                    # Default stump positions if not available
                    left_stump_x = -5
                    right_stump_x = 5
                
                # Batsman Orientation
                batsman_orientation = batEdge.get("field_setup", {}).get("batsman_orientation", "RH")
                if batsman_orientation == 'U':
                    batsman_orientation = 'R'
                
                # Ball Trajectory
                ball_trajectory_before = batEdge.get("previous_trajectory", [])
                
                # BALL PITCH POSITION
                if pitch_point is None and ball_trajectory_before:
                    for i in range(1, len(ball_trajectory_before)):
                        if ball_trajectory_before[i][1] >= ball_trajectory_before[i-1][1]:
                            pitch_point = ball_trajectory_before[i]
                            if pitch_point[0] < left_stump_x:
                                if batsman_orientation == 'R':
                                    pitch = "Outside Off"
                                else:
                                    pitch = "Outside Leg"
                            elif pitch_point[0] > right_stump_x:
                                if batsman_orientation == 'R':
                                    pitch = "Outside Leg"
                                else:
                                    pitch = "Outside Off"
                            else:
                                pitch = "Inline"
                            break
                
                if pitch == "Outside Leg":
                    decision = "NOT OUT"
                    if bat_contact_flag == True:
                        reason += " and "
                    reason += "Ball pitched outside Leg stump"
                
                # LEG IMPACT
                if leg_contact_flag and leg_contact_point:
                    if leg_contact_point[0] >= left_stump_x and leg_contact_point[0] <= right_stump_x:
                        leg_contact_position = "Inline"
                    else:
                        leg_contact_position = "Outside Line"
                    
                    if not pitch:
                        if leg_contact_point[0] < left_stump_x:
                            if batsman_orientation == 'R':
                                pitch = "Outside Off"
                            else:
                                pitch = "Outside Leg"
                        elif leg_contact_point[0] > right_stump_x:
                            if batsman_orientation == 'R':
                                pitch = "Outside Leg"
                            else:
                                pitch = "Outside Off"
                        else:
                            pitch = "Inline"
                else:
                    leg_contact_position = "Pads Missing"
                
                if leg_contact_position == "Outside Line":
                    decision = "NOT OUT"
                    reason = "Impact on Pad Outside Line"
                
                if decision != "NOT OUT":
                    if hitting_stumps:
                        hitting_stumps_point = predictedTraj.get("stump_impact_location")
                        decision = "OUT"
                        reason = "Ball hitting the stumps"
                    else:
                        decision = "NOT OUT"
                        reason = "Ball not hitting the stumps"
            
            # Prepare output
            output = {
                "Decision": decision,
                "Reason": reason,
                "BallPitch": pitch,
                "BallPitchPoint": pitch_point,
                "PadImpact": leg_contact_position,
                "PadImpactPoint": leg_contact_point,
                "HittingStumps": hitting_stumps,
                "HittingStumpsPoint": hitting_stumps_point
            }
            
            # Save output
            output_path = Path("Server files/decision_output.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=4)
            
            # Display results
            decision_text = f"Decision: {output['Decision']}\nReason: {output['Reason']}"
            decision_label.config(text=decision_text, fg="green")
            
        except Exception as e:
            decision_label.config(text=f"Error in decision making:\n{str(e)}", fg="red")
    
    threading.Thread(target=task, daemon=True).start()
# ------------  Function -------------- #

def run_tracking_module():
    # Update status immediately when button is clicked
    tracking_status_label.config(text="Starting video processing...", fg="blue")
    
    def task():
        try:
            # Update status when processing begins
            tracking_status_label.config(text="Processing video...", fg="orange")
            
            # Run the actual tracking
            result = track_objects(videopath)
            
            # Update status when complete
            if result:
                tracking_status_label.config(
                    text="Processing complete!\nResults saved to JSON",
                    fg="green"
                )
            else:
                tracking_status_label.config(
                    text="Processing completed with warnings",
                    fg="orange"
                )
                
        except Exception as e:
            # Show error message if something fails
            tracking_status_label.config(
                text=f"Error processing video:\n{str(e)}",
                fg="red"
            )
    
    # Run the task in a separate thread (fixed missing parenthesis)
    threading.Thread(target=task, daemon=True).start()

def back_from_decision_making():
    decision_label.config(text="")

    # Explicitly hide the decision_making_frame
    decision_making_frame.pack_forget()

    # Ensure all frames are hidden before switching
    for frame in all_frames:
        frame.pack_forget()

    # Show only the main menu frame
    main_menu_frame.pack(fill="both", expand=True)

# ------------ UI Setup -------------- #
root = tk.Tk()
root.title("DRS App")
root.geometry("360x640")
root.configure(bg="#f0f0f0")

btn_style = {
    "font": ("Helvetica", 12),
    "width": 25,
    "height": 2,
    "bg": "#1976D2",
    "fg": "white",
    "activebackground": "#1565C0",
    "bd": 0
}

# ------------ Frame: Main Menu -------------- #
main_menu_frame = tk.Frame(root, bg="#f0f0f0")

modules = [
    ("Camera", show_camera_controls),
    ("Ball and Object Tracking", show_tracking_controls),
    ("Bat’s Edge Detection", show_bat_detection),
    ("Trajectory Analysis", show_trajectory_analysis),
    ("Decision Making", show_decision_making),
    ("Video Overlay", show_overlay_controls),
]

tk.Label(main_menu_frame, text="DRS Modules", font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=20)

for text, command in modules:
    tk.Button(main_menu_frame, text=text, command=command, **btn_style).pack(pady=8)

# ------------ Frame: Camera Controls -------------- #
camera_controls_frame = tk.Frame(root, bg="#f0f0f0")

tk.Label(camera_controls_frame, text="Camera Controls", font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=20)
tk.Button(camera_controls_frame, text="Start Camera", command=start_camera, **btn_style).pack(pady=8)
tk.Button(camera_controls_frame, text="Pause Camera", command=pause_camera, **btn_style).pack(pady=8)
tk.Button(camera_controls_frame, text="Resume Camera", command=resume_camera, **btn_style).pack(pady=8)
tk.Button(camera_controls_frame, text="Stop Camera", command=stop_camera, **btn_style).pack(pady=8)
tk.Button(camera_controls_frame, text="Back to Modules", command=show_main_menu,
          font=("Helvetica", 12), width=25, height=2, bg="#E91E63", fg="white", bd=0).pack(pady=20)

# ------------ Frame: Stream & Overlay Video Controls -------------- #

def run_streamOverlay():
    def task():
        try:
            overlay_status_label.config(text="Starting video overlay...", fg="blue")
            
            # Define paths
            video_path = Path("recordings/updated_video.mp4")
            module4_path = Path("Server files/trajectory_output.json")  # From trajectory analysis
            module5_path = Path("Server files/bat_detection_output/detection_results.json")  # From bat detection
            output_path = Path("output videos/augmented_video.avi")
            
            # Verify input files exist
            if not video_path.exists():
                overlay_status_label.config(text="Error: Input video not found", fg="red")
                return
            if not module4_path.exists():
                overlay_status_label.config(text="Error: Trajectory data not found", fg="red")
                return
            if not module5_path.exists():
                overlay_status_label.config(text="Error: Bat detection data not found", fg="red")
                return
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run the overlay renderer
            from draw_trajectory_new import TrajectoryOverlayRenderer  # Import your renderer class
            renderer = TrajectoryOverlayRenderer(
                video_path=str(video_path),
                module4_json=str(module4_path),
                module5_json=str(module5_path),
                output_path=str(output_path),
                slow_factor=3  # Adjust as needed
            )
            renderer.run()
            
            # Update status
            overlay_status_label.config(
                text=f"Video overlay complete!",
                fg="green"
            )
            
            # Add button to play the output video
            play_button = tk.Button(
                overlay_controls_frame,
                text="Play Augmented Video",
                command=lambda: play_video(str(output_path)),
                **btn_style
            )
            play_button.pack(pady=10)
            
        except Exception as e:
            overlay_status_label.config(
                text=f"Error in video overlay:\n{str(e)}",
                fg="red"
            )
    
    threading.Thread(target=task, daemon=True).start()
# ------------ Frame: Stream & Overlay" -------------- #
overlay_controls_frame = tk.Frame(root, bg="#f0f0f0")

overlay_status_label = tk.Label(
    overlay_controls_frame,
    text="Ready to create video overlay",
    font=("Helvetica", 12),
    bg="#f0f0f0",
    fg="black",
    wraplength=300,
    justify="left"
)
overlay_status_label.pack(pady=10)

video_controls_frame = tk.Frame(root, bg="#f0f0f0")

tk.Label(video_controls_frame, text="Stream & Overlay", font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=20)

tk.Label(overlay_controls_frame, text="Video Overlay", font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=20)
tk.Button(overlay_controls_frame, text="Create Video Overlay", command=run_streamOverlay, **btn_style).pack(pady=10)
overlay_status_label.pack(pady=10)
tk.Button(overlay_controls_frame, text="Back to Modules", command=show_main_menu,
          font=("Helvetica", 12), width=25, height=2, bg="#E91E63", fg="white", bd=0).pack(pady=30)


# ------------ Frame: Ball and Object Tracking -------------- #

# Tracking controls frame setup
tracking_controls_frame = tk.Frame(root, bg="#f0f0f0")

# Title label (packed directly since we don't need to reference it later)
tk.Label(tracking_controls_frame, 
        text="Ball & Object Tracking", 
        font=("Helvetica", 18, "bold"), 
        bg="#f0f0f0").pack(pady=20)

# Status label - created and packed separately to maintain reference
tracking_status_label = tk.Label(
    tracking_controls_frame,
    text="Ready to process video",
    font=("Helvetica", 12),
    bg="#f0f0f0",
    fg="black",
    wraplength=300,  # Allows text to wrap
    justify="left"
)
tracking_status_label.pack(pady=10)

# Tracking button
tk.Button(
    tracking_controls_frame, 
    text="Run Ball & Object Tracking", 
    command=run_tracking_module, 
    **btn_style
).pack(pady=10)

# Video playback button
tk.Button(
    tracking_controls_frame, 
    text="Play Input Video", 
    command=lambda: play_video(videopath), 
    **btn_style
).pack(pady=10)

# Back button
tk.Button(
    tracking_controls_frame, 
    text="Back to Modules", 
    command=show_main_menu,
    font=("Helvetica", 12), 
    width=25, 
    height=2, 
    bg="#E91E63", 
    fg="white", 
    bd=0
).pack(pady=30)


#=================bat edge detection===================#
# Create the bat detection frame
bat_detection_frame = tk.Frame(root, bg="#f0f0f0")

# Title label
tk.Label(bat_detection_frame, 
        text="Bat Edge Detection", 
        font=("Helvetica", 18, "bold"), 
        bg="#f0f0f0").pack(pady=20)

# Status label
bat_status_label = tk.Label(
    bat_detection_frame,
    text="Ready to detect bat edges",
    font=("Helvetica", 12),
    bg="#f0f0f0",
    fg="black",
    wraplength=300,
    justify="left"
)
bat_status_label.pack(pady=10)

# Process button
tk.Button(
    bat_detection_frame, 
    text="Run Bat Edge Detection", 
    command=run_bat_edge_detection, **btn_style).pack(pady=10)

# Back button
tk.Button(bat_detection_frame, text="Back to Modules", command=show_main_menu,
          font=("Helvetica", 12), width=25, height=2, bg="#E91E63", fg="white", bd=0).pack(pady=30)

# ========================Frame: Trajectory Analysis ========================
trajectory_analysis_frame = tk.Frame(root, bg="#f0f0f0")

tk.Label(trajectory_analysis_frame, text="Trajectory Analysis", font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=20)
tk.Button(trajectory_analysis_frame, text="Run Analysis", command=run_trajectory_analysis, **btn_style).pack(pady=10)
verdict_label = tk.Label(trajectory_analysis_frame, text="", font=("Helvetica", 12), bg="#f0f0f0", justify="left")
verdict_label.pack(pady=10)
tk.Button(trajectory_analysis_frame, text="Back to Modules", command=show_main_menu,
          font=("Helvetica", 12), width=25, height=2, bg="#E91E63", fg="white", bd=0).pack(pady=30)

# =========================== Frame: Decision Making ========================
decision_making_frame = tk.Frame(root, bg="#f0f0f0")

tk.Label(decision_making_frame, text="Decision Making", font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=20)
tk.Button(decision_making_frame, text="Run Decision Making", command=run_decision_making, **btn_style).pack(pady=10)
decision_label = tk.Label(decision_making_frame, text="", font=("Helvetica", 12), bg="#f0f0f0", justify="left")
decision_label.pack(pady=10)
#tk.Button(decision_making_frame, text="Back to Modules", command=show_main_menu,
#          font=("Helvetica", 12), width=25, height=2, bg="#E91E63", fg="white", bd=0).pack(pady=30)
tk.Button(
    decision_making_frame,
    text="Back to Modules",
    command=back_from_decision_making,
    font=("Helvetica", 12),
    width=25,
    height=2,
    bg="#E91E63",
    fg="white",
    bd=0
).pack(pady=30)

# ------------ Track All Frames for Navigation -------------- #
all_frames = [main_menu_frame, camera_controls_frame, video_controls_frame, tracking_controls_frame, trajectory_analysis_frame, bat_detection_frame ,overlay_controls_frame]

# ------------ Start at Main Menu -------------- #
main_menu_frame.pack(fill="both", expand=True)

root.mainloop()
