import cv2
import json
import numpy as np


# ---INPUT module 4 & 5 ----
def load_json_from_module4():
    with open('module4_output.json') as f:
        return json.load(f)

def load_json_from_module5():
    with open('module5_output.json') as f:
        return json.load(f)


# Convert the input into 2D format
def remove_z_from_trajectory(data):
    trajectory_2d = [{"x": point["x"], "y": point["y"]} for point in data["predicted_trajectory"]]
    data["predicted_trajectory"] = trajectory_2d
    return data


# 3D coordinates to 2D canvas
def project_coordinates(x, y):
    return int(x * 100 + 320), int(240 - y * 100)

def project_coordinates_3d(x, y, z):
    return int(x * 100 + 320), int(240 - y * 100)


# Check if two points are close enough
def is_near(point1, point2, tolerance=10):
    return abs(point1[0] - point2[0]) < tolerance and abs(point1[1] - point2[1]) < tolerance


# ---- OVERLAY FUNCTION ----
def draw_overlay(frame, data_module4, data_module5, frame_idx, bounce_shown, impact_shown):
    data_module5 = remove_z_from_trajectory(data_module5)
    trajectory_points = data_module5.get("predicted_trajectory", [])

    if not trajectory_points or frame_idx >= len(trajectory_points):
        return frame, bounce_shown, impact_shown

    # Ball position in current frame
    current_point = trajectory_points[frame_idx]
    current_pos = project_coordinates(**current_point)

    # Trajectory up to current frame
    for i in range(1, frame_idx + 1):
        pt1 = project_coordinates(**trajectory_points[i - 1])
        pt2 = project_coordinates(**trajectory_points[i])
        overlay = frame.copy()
        cv2.line(overlay, pt1, pt2, (255, 0, 0), 7)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Draw ball
    cv2.circle(frame, current_pos, 7, (255, 0, 0), -1)

    # BOUNCE POINT
    if 'bounce_point' in data_module4:
        bounce = data_module4['bounce_point']
        bounce_2d = project_coordinates_3d(bounce['pos_x'], bounce['pos_y'], bounce['pos_z'])
        if not bounce_shown and is_near(current_pos, bounce_2d):
            bounce_shown = True
        if bounce_shown:
            cv2.circle(frame, bounce_2d, 7, (255, 0, 0), -1)

    # IMPACT POINT
    if 'impact_point' in data_module4:
        impact = data_module4['impact_point']
        impact_2d = project_coordinates_3d(impact['pos_x'], impact['pos_y'], impact['pos_z'])
        if not impact_shown and is_near(current_pos, impact_2d):
            impact_shown = True
        if impact_shown:
            cv2.circle(frame, impact_2d, 7, (255, 0, 0), -1)

    # Decision Box
    height, width, _ = frame.shape
    box_x_start = width - 350
    box_y_start = 20
    box_width = 330
    box_height = 200
    cv2.rectangle(frame, (box_x_start, box_y_start), (box_x_start + box_width, box_y_start + box_height), (0, 0, 0), -1)

    decision = data_module5.get("decision", "Pending")
    reason = data_module5.get("reason", "")
    max_line_length = 30
    reason_lines = []
    while len(reason) > max_line_length:
        space_index = reason.rfind(' ', 0, max_line_length)
        if space_index == -1:
            space_index = max_line_length
        reason_lines.append(reason[:space_index].strip())
        reason = reason[space_index:].strip()
    if reason:
        reason_lines.append(reason)

    cv2.putText(frame, f"Decision: {decision}", (box_x_start + 10, box_y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    line_y = box_y_start + 70
    for line in reason_lines:
        cv2.putText(frame, line, (box_x_start + 10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += 30

    return frame, bounce_shown, impact_shown


# --- MAIN FUNCTION ---
def run_overlay():
    cap = cv2.VideoCapture("input_video.mp4")

    data_module4 = load_json_from_module4()
    data_module5 = load_json_from_module5()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    bounce_shown = False
    impact_shown = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, bounce_shown, impact_shown = draw_overlay(frame, data_module4, data_module5, frame_idx, bounce_shown, impact_shown)

        cv2.imshow("Ball Tracking Overlay", annotated_frame)
        out.write(annotated_frame)

        frame_idx += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_overlay()
