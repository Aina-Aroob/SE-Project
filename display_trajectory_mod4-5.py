# import cv2
# import json

# # Load Module 4 and Module 5 JSON data
# def load_json_from_module4():
#     with open('module4_output.json') as f:
#         return json.load(f)

# def load_json_from_module5():
#     with open('module5_output.json') as f:
#         return json.load(f)

# # Convert 3D coordinates to 2D canvas (simple projection)
# def project_coordinates(x, y, z):
#     # For simplicity, use x and y (ignoring depth or using z for brightness/size)
#     # Scale to fit frame size (assumes 640x480)
#     return int(x * 100 + 320), int(240 - y * 100)

# # Overlay drawing function
# def draw_overlay(frame, data_module4, data_module5, frame_idx):
#     # 1. Draw trajectory from Module 5's predicted trajectory
#     trajectory_points = data_module5.get("predicted_trajectory", [])

#     # Draw the trajectory progressively: only up to the current frame_idx
#     for i in range(1, frame_idx + 1):
#         pt1 = project_coordinates(**trajectory_points[i - 1])
#         pt2 = project_coordinates(**trajectory_points[i])
#         cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Blue line connecting points
    
#     # Draw each point as a circle (blue) up to the current point
#     if frame_idx < len(trajectory_points):
#         point = trajectory_points[frame_idx]
#         current_pos = project_coordinates(**point)
#         cv2.circle(frame, current_pos, 5, (255, 0, 0), -1)  # Blue circle at current point

#     # 2. Show decision text
#     decision = data_module5.get("decision", "Pending")
#     reason = data_module5.get("reason", "")
#     cv2.putText(frame, f"Decision: {decision}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#     cv2.putText(frame, f"Reason: {reason}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # 3. Show confidence score
#     conf = data_module5.get("confidence_score", None)
#     if conf:
#         cv2.putText(frame, f"Confidence: {conf:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 100), 2)

#     return frame

# # Main loop
# def run_overlay():
#     cap = cv2.VideoCapture("sample_ball_video.mp4")  # Or 0 for webcam

#     data_module4 = load_json_from_module4()
#     data_module5 = load_json_from_module5()

#     # Video writer setup
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#     frame_idx = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Draw overlays progressively
#         annotated_frame = draw_overlay(frame, data_module4, data_module5, frame_idx)

#         # Show frame
#         cv2.imshow("Ball Tracking Overlay", annotated_frame)

#         # Write frame to output video
#         out.write(annotated_frame)

#         # Progress to next frame
#         frame_idx += 1

#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     run_overlay()


















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


# converting the input into 2d format
def remove_z_from_trajectory(data):
    trajectory_2d = [{"x": point["x"], "y": point["y"]} for point in data["predicted_trajectory"]]
    data["predicted_trajectory"] = trajectory_2d
    return data


# 3D coordinates to 2D canvas (simple projection)
def project_coordinates(x, y):
    return int(x * 100 + 320), int(240 - y * 100)

def project_coordinates_3d(x, y, z):
    return int(x * 100 + 320), int(240 - y * 100)



# ---- OVERLAY FUNCTION ----
def draw_overlay(frame, data_module4, data_module5, frame_idx, bounce_shown, impact_shown):
    data_module5 = remove_z_from_trajectory(data_module5)
    trajectory_points = data_module5.get("predicted_trajectory", [])     # Get trajectory points

    if not trajectory_points:
        print("Warning: No predicted trajectory data found.")
        return frame, bounce_shown, impact_shown


    # ball trajectory frame by frame
    for i in range(1, frame_idx + 1):
        if i < len(trajectory_points):
            pt1 = project_coordinates(**trajectory_points[i - 1])
            pt2 = project_coordinates(**trajectory_points[i])
            cv2.line(frame, pt1, pt2, (0, 0, 255), 4)                   # Red line for trajectory


    # ball in blue
    if frame_idx < len(trajectory_points):
        point = trajectory_points[frame_idx]
        current_pos = project_coordinates(**point)
        cv2.circle(frame, current_pos, 7, (128, 0, 0), -1)              # Dark blue ball


    # BOUNCE POINT
    if 'bounce_point' in data_module4:
        bounce = data_module4['bounce_point']
        bounce_2d = project_coordinates_3d(bounce['pos_x'], bounce['pos_y'], bounce['pos_z'])

        # Checking if the ball has reched the bounce point
        if not bounce_shown and is_near(current_pos, bounce_2d):  
            bounce_shown = True
        if bounce_shown:
            cv2.circle(frame, bounce_2d, 7, (128, 0, 0), -1)            # ball stays at bounce point

            # Continue drawing trajectory from bounce point onwards
            next_points = trajectory_points[frame_idx:]
            for i in range(1, len(next_points)):
                pt1 = project_coordinates(**next_points[i - 1])
                pt2 = project_coordinates(**next_points[i])
                cv2.line(frame, pt1, pt2, (0, 0, 255), 4)  


    # IMPACT POINT
    if 'impact_point' in data_module4:
        impact = data_module4['impact_point']
        impact_2d = project_coordinates_3d(impact['pos_x'], impact['pos_y'], impact['pos_z'])

        # checking if ball reached impact point
        if not impact_shown and is_near(current_pos, impact_2d):  # Add tolerance checking
            impact_shown = True
        if impact_shown:
            cv2.circle(frame, impact_2d, 7, (128, 0, 0), -1)  


    # WRITING DECISION IN A BOX ON RIGHT SIDE OF SCREEN
    height, width, _ = frame.shape
    box_x_start = width - 350
    box_y_start = 20
    box_width = 330
    box_height = 200  
    cv2.rectangle(frame, (box_x_start, box_y_start), (box_x_start + box_width, box_y_start + box_height), (0, 0, 0), -1)

    # Add text inside the box
    decision = data_module5.get("decision", "Pending")
    reason = data_module5.get("reason", "")

    # Split reason into lines that fit within the box width  - if reason big ho
    max_line_length = 30        # Maximum characters per line
    reason_lines = []
    while len(reason) > max_line_length:
        space_index = reason.rfind(' ', 0, max_line_length)   # Find the last space within max_line_length
        if space_index == -1:                                 # No space found, break at max_line_length
            space_index = max_line_length
        reason_lines.append(reason[:space_index].strip())
        reason = reason[space_index:].strip()
    if reason:
        reason_lines.append(reason)                           # Add the remaining part of the reason

    # Positioning decision text
    cv2.putText(frame, f"Decision: {decision}", (box_x_start + 10, box_y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Positioning reason text line by line
    line_y = box_y_start + 70  # Start position for reason
    for line in reason_lines:
        cv2.putText(frame, line, (box_x_start + 10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += 30  # Move to the next line

    return frame, bounce_shown, impact_shown



# Function to check if two points are close enough (within a tolerance)
def is_near(point1, point2, tolerance=10):
    return abs(point1[0] - point2[0]) < tolerance and abs(point1[1] - point2[1]) < tolerance


# ---MAIN----
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
