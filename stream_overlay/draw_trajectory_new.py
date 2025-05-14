import cv2
import json
import numpy as np

class TrajectoryOverlayRenderer:
    def __init__(self, video_path, module4_json, module5_json, output_path, slow_factor=3):
        self.video_path = video_path
        self.module4_json = module4_json
        self.module5_json = module5_json
        self.output_path = output_path
        self.slow_factor = slow_factor
        self._load_data()
        self._setup_video()

    def _load_data(self):
        with open(self.module4_json, 'r') as f4:
            data4 = json.load(f4)
        with open(self.module5_json, 'r') as f5:
            data5 = json.load(f5)

        prev_traj = data4["previous_trajectory"]
        pred_traj = data4["predicted_trajectory"]

        self.real_trajectory = [
            {"x": int(x), "y": int(y)} for (x, y, _) in prev_traj
        ]
        self.predicted_trajectory = [
            {"x": int(x), "y": int(y)} for (x, y, _) in pred_traj
        ]

        collision_pt = data4.get("collision", {}).get("spatial_detection", {}).get("collision_point")
        leg_impact_pt = data4.get("leg_impact_location")

        self.collision_point = (
            {"x": int(collision_pt[0]), "y": int(collision_pt[1])}
            if collision_pt else None
        )
        self.impact_point = (
            {"x": int(leg_impact_pt[0]), "y": int(leg_impact_pt[1])}
            if leg_impact_pt else None
        )

        self.pitching_result = data5.get("BallPitch", "N/A")
        self.impact_result = data5.get("PadImpact", "N/A")
        self.wickets_result = "HITTING" if data5.get("HittingStumps", False) else "MISSING"
        self.final_decision = data5.get("Decision", "N/A")

        self.real_color = (255, 0, 0)
        self.predicted_color = (255, 0, 255)  # bright magenta-pink
        self.trajectory_thickness = 14
        self.ball_dot_radius = 6
        self.bounce_color = (0, 255, 255)
        self.impact_color = (0, 0, 255)
        self.collision_color = (0, 255, 0)
        self.marker_radius = 7

        self.impact_index = self._find_closest_index(self.real_trajectory, self.impact_point) if self.impact_point else -1
        self.collision_index = self._find_closest_index(self.real_trajectory, self.collision_point) if self.collision_point else -1

    def _find_closest_index(self, trajectory, target_point):
        min_dist = float('inf')
        min_idx = -1
        for i, pt in enumerate(trajectory):
            dist = (pt['x'] - target_point['x'])**2 + (pt['y'] - target_point['y'])**2
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return min_idx

    def _setup_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            self.fps,
            (self.width, self.height)
        )
        self.frame_idx = 0

    def draw_gradient_box(self, frame, x, y, width, height, color_top, color_bottom):
        for i in range(height):
            alpha = i / height
            color = [
                int((1 - alpha) * color_top[j] + alpha * color_bottom[j])
                for j in range(3)
            ]
            cv2.line(frame, (x, y + i), (x + width, y + i), color, 1)

    def draw_decision_boxes(self, frame):
        font = cv2.FONT_HERSHEY_DUPLEX
        label_font_scale = 0.6
        value_font_scale = 0.6
        thickness = 1

        labels = ["PITCHING", "IMPACT", "WICKETS", "FINAL DECISION"]
        values = [
            self.pitching_result.upper(),
            self.impact_result.upper(),
            self.wickets_result.upper(),
            self.final_decision.upper()
        ]

        value_colors = {
            "OUT": (0, 0, 255),
            "NOT OUT": (26, 158, 26),
            "HITTING": (0, 0, 255),
            "MISSING": (0, 0, 255),
            "IN-LINE": (26, 158, 26),
            "INLINE": (18, 118, 18),
            "OUTSIDE OFF": (0, 0, 255),
            "OUTSIDE LEG": (0, 0, 255),
            "N/A": (128, 128, 128)
        }

        box_width = 180
        box_height = 30
        spacing = 12
        start_x = frame.shape[1] - box_width - 50
        start_y = frame.shape[0] // 5

        for i in range(len(labels)):
            y = start_y + i * (box_height * 2 + spacing)
            self.draw_gradient_box(frame, start_x, y, box_width, box_height,
                                   color_top=(195, 47, 47), color_bottom=(84, 18, 18))

            label_size = cv2.getTextSize(labels[i], font, label_font_scale, thickness)[0]
            label_x = start_x + (box_width - label_size[0]) // 2
            label_y = y + (box_height + label_size[1]) // 2 - 2
            cv2.putText(frame, labels[i], (label_x, label_y), font, label_font_scale, (255, 255, 255), thickness)

            y_val = y + box_height
            value = values[i]
            color = value_colors.get(value, (100, 100, 100))
            cv2.rectangle(frame, (start_x, y_val), (start_x + box_width, y_val + box_height), color, -1)

            value_size = cv2.getTextSize(value, font, value_font_scale, thickness)[0]
            value_x = start_x + (box_width - value_size[0]) // 2
            value_y = y_val + (box_height + value_size[1]) // 2 - 2
            cv2.putText(frame, value, (value_x, value_y), font, value_font_scale, (255, 255, 255), thickness)

        return frame

    def draw_overlay(self):
        pause_frames = 30  # number of frames to pause at end to show predicted path

        full_trajectory = self.real_trajectory + self.predicted_trajectory

        last_real_frame = None

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if ret and self.frame_idx < len(self.real_trajectory):
                # 🎥 Normal playback: show real trajectory frame by frame
                last_real_frame = frame.copy()
                overlay = frame.copy()
                visible_index = max(0, self.frame_idx - 3)

                # 🎨 Draw real trajectory up to current frame
                for i in range(1, min(visible_index + 1, len(self.real_trajectory))):
                    pt1 = (self.real_trajectory[i - 1]['x'], self.real_trajectory[i - 1]['y'])
                    pt2 = (self.real_trajectory[i]['x'], self.real_trajectory[i]['y'])
                    cv2.line(overlay, pt1, pt2, self.real_color, self.trajectory_thickness)

                # 🎯 Draw ball dot
                if visible_index < len(self.real_trajectory):
                    point = self.real_trajectory[visible_index]
                    current_pos = (point['x'], point['y'])
                    cv2.circle(overlay, current_pos, self.ball_dot_radius, self.real_color, -1)

                # 🎯 Draw overlay points
                if self.collision_point and visible_index >= self.collision_index:
                    cv2.circle(overlay, (self.collision_point['x'], self.collision_point['y']),
                            self.marker_radius, self.collision_color, -1)

                if self.impact_point and visible_index >= self.impact_index:
                    cv2.circle(overlay, (self.impact_point['x'], self.impact_point['y']),
                            self.marker_radius, self.impact_color, -1)

                blended_frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                final_frame = self.draw_decision_boxes(blended_frame)

                for _ in range(self.slow_factor):
                    self.out.write(final_frame)

                self.frame_idx += 1

            elif last_real_frame is not None:
                # 🎥 Ball has reached end → pause frame and show full real + predicted path
                for _ in range(pause_frames):
                    overlay = last_real_frame.copy()

                    # 🎨 Draw full real trajectory
                    for i in range(1, len(self.real_trajectory)):
                        pt1 = (self.real_trajectory[i - 1]['x'], self.real_trajectory[i - 1]['y'])
                        pt2 = (self.real_trajectory[i]['x'], self.real_trajectory[i]['y'])
                        cv2.line(overlay, pt1, pt2, self.real_color, self.trajectory_thickness)

                    # 🎨 Draw full predicted trajectory
                    for i in range(1, len(self.predicted_trajectory)):
                        pt1 = (self.predicted_trajectory[i - 1]['x'], self.predicted_trajectory[i - 1]['y'])
                        pt2 = (self.predicted_trajectory[i]['x'], self.predicted_trajectory[i]['y'])
                        cv2.line(overlay, pt1, pt2, self.predicted_color, int(self.trajectory_thickness * 0.5))

                    blended_frame = cv2.addWeighted(overlay, 0.3, last_real_frame, 0.7, 0)
                    final_frame = self.draw_decision_boxes(blended_frame)

                    for _ in range(self.slow_factor):
                        self.out.write(final_frame)

                break  # 🎬 Exit after pause

            else:
                break

        self.cap.release()
        self.out.release()
        print(f"Output video saved as: {self.output_path}")




    def run(self):
        self.draw_overlay()


if __name__ == "__main__":
    renderer = TrajectoryOverlayRenderer(
        video_path="input_video3.avi",
        module4_json="module4new.json",
        module5_json="module5_output.json",
        output_path="output_video3.avi",
        slow_factor=3
    )
    renderer.run()
