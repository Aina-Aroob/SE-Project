import cv2
import json

# Load trajectory and markers from JSON
with open('ball_trajectory.json', 'r') as f:
    data = json.load(f)

trajectory = data["trajectory"]
bounce_point = data.get("bounce_point")
impact_point = data.get("impact_point")
decision = data.get("decision")
metadata = data.get("metadata", {})


#------HANDLING METADATA HERE----
trajectory_color = tuple(metadata.get("trajectory_color", [255, 0, 0]))             # Blue
trajectory_thickness = metadata.get("trajectory_thickness", 12)
ball_dot_radius = metadata.get("ball_dot_radius", 6)
bounce_color = tuple(metadata.get("bounce_color", [0, 255, 255]))                   # Yellow
impact_color = tuple(metadata.get("impact_color", [0, 0, 255]))                     # Red
marker_radius = metadata.get("marker_radius", 10)
top_box_color = tuple(metadata.get("decision_box_top_color", [255, 0, 0]))          # Blue
bottom_box_color_out = tuple(metadata.get("decision_box_bottom_color_out", [0, 0, 255]))         # Red
bottom_box_color_not_out = tuple(metadata.get("decision_box_bottom_color_not_out", [0, 255, 0])) # Green

# Open input video
cap = cv2.VideoCapture('input_video.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0
bounce_shown = False
impact_shown = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(trajectory):
        break

    overlay = frame.copy()
    point = trajectory[frame_idx]
    current_pos = (point['x'], point['y'])

    # Draw trajectory so far
    for i in range(1, frame_idx + 1):
        pt1 = (trajectory[i - 1]['x'], trajectory[i - 1]['y'])
        pt2 = (trajectory[i]['x'], trajectory[i]['y'])
        cv2.line(overlay, pt1, pt2, trajectory_color, trajectory_thickness)

    # Current ball dot
    cv2.circle(overlay, current_pos, ball_dot_radius, trajectory_color, -1)

    # Bounce point marker
    if bounce_point:
        if current_pos == (bounce_point['x'], bounce_point['y']):
            bounce_shown = True
        if bounce_shown:
            cv2.circle(overlay, (bounce_point['x'], bounce_point['y']), marker_radius, bounce_color, -1)

    # Impact point marker
    if impact_point:
        if current_pos == (impact_point['x'], impact_point['y']):
            impact_shown = True
        if impact_shown:
            cv2.circle(overlay, (impact_point['x'], impact_point['y']), marker_radius, impact_color, -1)

    # Translucent blending
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    last_frame = frame.copy()
    out.write(frame)
    frame_idx += 1

cap.release()

# Pause on last frame and show decision box
if decision and last_frame is not None:
    pause_duration_frames = int(fps * 5)
    final_frame = last_frame.copy()

    text = "Decision"
    text2 = f"{decision}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_size2, _ = cv2.getTextSize(text2, font, font_scale, thickness)

    x = final_frame.shape[1] - text_size[0] - 100
    y = ((final_frame.shape[0] + text_size[1]) // 2) - text_size[1] - 20
    y2 = (final_frame.shape[0] + text_size[1]) // 2

    # Top blue box
    cv2.rectangle(final_frame, 
                  (x - 10, y - text_size[1] - 10),
                  (x + text_size[0] + 10, y + 10),
                  top_box_color, -1)
    cv2.putText(final_frame, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Bottom box (color based on decision)
    bottom_color = bottom_box_color_out if decision.lower() == "out" else bottom_box_color_not_out
    cv2.rectangle(final_frame, 
                  (x - 10, y2 - text_size2[1] - 10),
                  (x + text_size2[0] + 10, y2 + 10),
                  bottom_color, -1)
    cv2.putText(final_frame, text2, (x, y2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    for _ in range(pause_duration_frames):
        out.write(final_frame)

out.release()
print("Output video saved as output_video.mp4.")
