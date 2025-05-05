import tkinter as tk
import threading
import requests
import webbrowser
import cv2
import os
import json

#=============== Import custom modules
from lbw_predictor import LBWPredictor
from DecisionMakingModule import Decision_Making_Module

SERVER_URL = "http://127.0.0.1:5000"

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

def show_trajectory_analysis():
    for frame in all_frames:
        frame.pack_forget()
    trajectory_analysis_frame.pack(fill="both", expand=True)

def show_decision_making():
    for frame in all_frames:
        frame.pack_forget()
    decision_making_frame.pack(fill="both", expand=True)

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

# ------------ Trajectory Analysis Function -------------- #


def run_trajectory_analysis():
    def task():
        predictor = LBWPredictor()

        # Load trajectory data from JSON file
        try:
            with open("trajectory.json", "r") as file:
                input_data = json.load(file)
        except Exception as e:
            verdict_label.config(text=f"Error loading trajectory data: {e}")
            return
        
        # Process the trajectory data using predictor
        result = predictor.process_input(input_data)
        verdict = result['verdict']
        
        # Display the verdict details
        verdict_text = (
            f"Status: {verdict['status']}\n"
            f"Will Hit Stumps: {verdict['will_hit_stumps']}\n"
            f"Impact Region: {verdict['impact_region']}\n"
            f"Confidence: {verdict['confidence']:.2f}"
        )
        verdict_label.config(text=verdict_text)

    threading.Thread(target=task).start()


# ------------ Decision Making Function -------------- #
def run_decision_making():
    def task():
        batEdge = {
            "decision_flag": [True, None]
        }
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
        batEdge_json = json.dumps(batEdge)
        result_json = Decision_Making_Module(batEdge_json, predictedTraj )
        result = json.loads(result_json)
        decision_text = f"Decision: {result['decision']}\nReason: {result['Reason']}"
        decision_label.config(text=decision_text)

    #threading.Thread(target=task).start()
    task()
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
    ("Bat’s Edge Detection", lambda: print("Coming Soon...")),
    ("Trajectory Analysis", show_trajectory_analysis),
    ("Decision Making", show_decision_making),
    ("Stream Analysis and Overlay", show_video_playback_controls),
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
video_controls_frame = tk.Frame(root, bg="#f0f0f0")

tk.Label(video_controls_frame, text="Stream & Overlay", font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=20)

input_video_path = os.path.abspath("input_video.mp4")
output1_video_path = os.path.abspath("output1_video.mp4")

tk.Button(video_controls_frame, text="Play Input Video", command=lambda: play_video(input_video_path), **btn_style).pack(pady=10)
tk.Button(video_controls_frame, text="Play Output Video", command=lambda: play_video(output1_video_path), **btn_style).pack(pady=10)
tk.Button(video_controls_frame, text="Back to Modules", command=show_main_menu,
          font=("Helvetica", 12), width=25, height=2, bg="#E91E63", fg="white", bd=0).pack(pady=30)

# ------------ Frame: Ball and Object Tracking -------------- #
tracking_controls_frame = tk.Frame(root, bg="#f0f0f0")

tk.Label(tracking_controls_frame, text="Ball & Object Tracking", font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=20)
input_video_path1 = os.path.abspath("front.mp4")
output_video_path1 = os.path.abspath("detections_overlaid.mp4")

tk.Button(tracking_controls_frame, text="Play Input Video", command=lambda: play_video(input_video_path1), **btn_style).pack(pady=10)
tk.Button(tracking_controls_frame, text="Play Output Video", command=lambda: play_video(output_video_path1), **btn_style).pack(pady=10)
tk.Button(tracking_controls_frame, text="Back to Modules", command=show_main_menu,
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
all_frames = [main_menu_frame, camera_controls_frame, video_controls_frame, tracking_controls_frame, trajectory_analysis_frame]

# ------------ Start at Main Menu -------------- #
main_menu_frame.pack(fill="both", expand=True)

root.mainloop()