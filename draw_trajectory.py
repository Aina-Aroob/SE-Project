import cv2
import json

# Load trajectory and markers from JSON
with open('ball_trajectory.json', 'r') as f:
    data = json.load(f)

trajectory = data["trajectory"]
bounce_point = data.get("bounce_point")
impact_point = data.get("impact_point")
decision = data.get("decision")  # Optional decision text

# Open input video
cap = cv2.VideoCapture('input_video.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0

# Flags to start showing bounce/impact once they appear
bounce_shown = False
impact_shown = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(trajectory):
        break

    # Current point in trajectory
    point = trajectory[frame_idx]
    current_pos = (point['x'], point['y'])

    # Draw current trajectory point
    cv2.circle(frame, current_pos, 5, (255, 0, 0), -1)  # Blue dot

    # Draw trajectory line up to current point
    for i in range(1, frame_idx + 1):
        pt1 = (trajectory[i - 1]['x'], trajectory[i - 1]['y'])
        pt2 = (trajectory[i]['x'], trajectory[i]['y'])
        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Blue line

    # Show bounce point if it's already passed
    if bounce_point:
        if current_pos == (bounce_point['x'], bounce_point['y']):
            bounce_shown = True
        if bounce_shown:
            cv2.circle(frame, (bounce_point['x'], bounce_point['y']), 10, (0, 255, 255), -1)  # Yellow

    # Show impact point if it's already passed
    if impact_point:
        if current_pos == (impact_point['x'], impact_point['y']):
            impact_shown = True
        if impact_shown:
            cv2.circle(frame, (impact_point['x'], impact_point['y']), 10, (0, 0, 255), -1)  # Red

    # Show decision after the impact point (or last 10 frames)
    if decision and (frame_idx >= len(trajectory) - 10 or
        (impact_point and point['x'] == impact_point['x'] and point['y'] == impact_point['y'])):
        
        text = f"Decision: {decision}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Position of the text
        x, y = 50, 50

        # Draw white rectangle behind text
        cv2.rectangle(frame, (x - 10, y - text_size[1] - 10), 
                    (x + text_size[0] + 10, y + 10), (255, 255, 255), -1)

        # Draw text over the white box (black text for contrast)
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


        out.write(frame)
        frame_idx += 1

cap.release()
out.release()
print("Output video with persistent trajectory, bounce, and impact markers saved as output_video.mp4")
