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

# Load Module 4 and Module 5 JSON data
def load_json_from_module4():
    with open('module4_output.json') as f:
        return json.load(f)

def load_json_from_module5():
    with open('module5_output.json') as f:
        return json.load(f)
    
def remove_z_from_trajectory(data):
    # Remove z and keep only x and y for 2D
    trajectory_2d = [{"x": point["x"], "y": point["y"]} for point in data["predicted_trajectory"]]
    data["predicted_trajectory"] = trajectory_2d
    return data


# Convert 3D coordinates to 2D canvas (simple projection)
def project_coordinates(x, y):
    # For simplicity, use x and y (ignoring depth or using z for brightness/size)
    # Scale to fit frame size (assumes 640x480)
    return int(x * 100 + 320), int(240 - y * 100)



# Overlay drawing function
def draw_overlay(frame, data_module4, data_module5, frame_idx):
    # Remove z and work only with x and y for 2D
    data_module5 = remove_z_from_trajectory(data_module5)

    trajectory_points = data_module5.get("predicted_trajectory", [])

    if not trajectory_points:
        print("Warning: No predicted trajectory data found.")
        return frame
    


    for i in range(1, frame_idx + 1):
        pt1 = project_coordinates(**trajectory_points[i - 1])
        pt2 = project_coordinates(**trajectory_points[i])
        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Blue line connecting points

    if frame_idx < len(trajectory_points):
        point = trajectory_points[frame_idx]
        current_pos = project_coordinates(**point)
        cv2.circle(frame, current_pos, 5, (255, 0, 0), -1)  # Blue circle at current point

    # # displaying decision text
    # decision = data_module5.get("decision", "Pending")
    # reason = data_module5.get("reason", "")
    # cv2.putText(frame, f"Decision: {decision}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    # cv2.putText(frame, f"Reason: {reason}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # # Show confidence score
    # conf = data_module5.get("confidence_score", None)
    # if conf:
    #     cv2.putText(frame, f"Confidence: {conf:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 100), 2)

    return frame


# Main loop
def run_overlay():
    cap = cv2.VideoCapture("sample_ball_video.mp4")  # Or 0 for webcam

    data_module4 = load_json_from_module4()
    data_module5 = load_json_from_module5()

    # Video writer setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw overlays progressively
        annotated_frame = draw_overlay(frame, data_module4, data_module5, frame_idx)

        # Show frame
        cv2.imshow("Ball Tracking Overlay", annotated_frame)

        # Write frame to output video
        out.write(annotated_frame)

        # Progress to next frame
        frame_idx += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_overlay()
