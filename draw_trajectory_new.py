import cv2
import json

#-----TRAJECTORY CLASS -------
class TrajectoryOverlayRenderer:

    def __init__(self, video_path, json_path, output_path):
        self.video_path = video_path
        self.json_path = json_path
        self.output_path = output_path
        self._load_data()
        self._setup_video()

    # Handling input here
    def _load_data(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        self.trajectory = data["trajectory"]
        self.bounce_point = data.get("bounce_point")
        self.impact_point = data.get("impact_point")
        self.pitching_result = data.get("pitching_result", "N/A")
        self.impact_result = data.get("impact_result", "N/A")
        self.wickets_result = data.get("wickets_result", "N/A")
        self.final_decision = data.get("final_decision", "N/A")

        metadata = data.get("metadata", {})

        self.trajectory_color = tuple(metadata.get("trajectory_color", [255, 0, 0]))
        self.trajectory_thickness = metadata.get("trajectory_thickness", 14)
        self.ball_dot_radius = metadata.get("ball_dot_radius", 6)
        self.bounce_color = tuple(metadata.get("bounce_color", [0, 255, 255]))
        self.impact_color = tuple(metadata.get("impact_color", [0, 0, 255]))
        self.marker_radius = metadata.get("marker_radius", 7)
        self.top_box_color = tuple(metadata.get("decision_box_top_color", [255, 0, 0]))
        self.bottom_box_color_out = tuple(metadata.get("decision_box_bottom_color_out", [0, 0, 255]))
        self.bottom_box_color_not_out = tuple(metadata.get("decision_box_bottom_color_not_out", [0, 255, 0]))

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
        self.bounce_shown = False
        self.impact_shown = False
        self.last_frame = None

    def pause_at_frame(self, frame, duration_seconds):
        pause_frames = int(self.fps * duration_seconds)
        for _ in range(pause_frames):
            self.out.write(frame)

    def write_slow_motion_segment(self, start_idx, end_idx, slow_factor=3):
        for idx in range(start_idx, end_idx):
            if idx >= len(self.trajectory):
                break

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                break

            point = self.trajectory[idx]
            current_pos = (point['x'], point['y'])

            overlay = frame.copy()
            for i in range(1, idx + 1):
                pt1 = (self.trajectory[i - 1]['x'], self.trajectory[i - 1]['y'])
                pt2 = (self.trajectory[i]['x'], self.trajectory[i]['y'])
                cv2.line(overlay, pt1, pt2, self.trajectory_color, self.trajectory_thickness)
            cv2.circle(overlay, current_pos, self.ball_dot_radius, self.trajectory_color, -1)

            alpha = 0.3
            overlayed_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            for _ in range(slow_factor):
                self.out.write(overlayed_frame)

    def draw_overlay(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or self.frame_idx >= len(self.trajectory):
                break

            overlay = frame.copy()
            point = self.trajectory[self.frame_idx]
            current_pos = (point['x'], point['y'])

            for i in range(1, self.frame_idx + 1):
                pt1 = (self.trajectory[i - 1]['x'], self.trajectory[i - 1]['y'])
                pt2 = (self.trajectory[i]['x'], self.trajectory[i]['y'])
                cv2.line(overlay, pt1, pt2, self.trajectory_color, self.trajectory_thickness)

            cv2.circle(overlay, current_pos, self.ball_dot_radius, self.trajectory_color, -1)

            if self.bounce_point and self.bounce_shown:
                cv2.circle(overlay, (self.bounce_point['x'], self.bounce_point['y']),
                           self.marker_radius, self.bounce_color, -1)

            if self.impact_point and self.impact_shown:
                cv2.circle(overlay, (self.impact_point['x'], self.impact_point['y']),
                           self.marker_radius, self.impact_color, -1)

            alpha = 0.3
            blended_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            if self.bounce_point and not self.bounce_shown and \
               current_pos == (self.bounce_point['x'], self.bounce_point['y']):
                slow_start = max(self.frame_idx - 5, 0)
                self.write_slow_motion_segment(slow_start, self.frame_idx + 1, slow_factor=5)
                self.pause_at_frame(blended_frame, 1)
                self.bounce_shown = True

            if self.impact_point and not self.impact_shown and \
               current_pos == (self.impact_point['x'], self.impact_point['y']):
                slow_start = max(self.frame_idx - 5, 0)
                self.write_slow_motion_segment(slow_start, self.frame_idx + 1, slow_factor=5)
                self.pause_at_frame(blended_frame, 1)
                self.impact_shown = True

            self.last_frame = blended_frame.copy()
            self.out.write(blended_frame)
            self.frame_idx += 1

        self.cap.release()

    def display_decision(self):
        if self.final_decision and self.last_frame is not None:
            pause_frames = int(self.fps * 5)
            frame = self.last_frame.copy()

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            thickness = 2

            labels = [
                "Pitching",
                "Impact",
                "Wickets",
                "Final Decision"
            ]
            values = [
                self.pitching_result,
                self.impact_result,
                self.wickets_result,
                self.final_decision
            ]

            text_sizes = [cv2.getTextSize(label, font, scale, thickness)[0] for label in labels]
            value_sizes = [cv2.getTextSize(value, font, scale, thickness)[0] for value in values]

            box_width = max(max(t[0], v[0]) for t, v in zip(text_sizes, value_sizes)) + 40
            box_height = text_sizes[0][1] + 30
            spacing = 10

            x = frame.shape[1] - box_width - 60
            y_start = (frame.shape[0] // 2) - ((box_height + spacing) * len(labels)) // 2

            def draw_rounded_box(img, x, y, w, h, color, radius=10, shadow=False):
                overlay = img.copy()
                shadow_offset = 4 if shadow else 0
                s_color = (50, 50, 50) if shadow else color

                cv2.rectangle(overlay, (x + shadow_offset + radius, y + shadow_offset),
                              (x + shadow_offset + w - radius, y + shadow_offset + h), s_color, -1)
                cv2.rectangle(overlay, (x + shadow_offset, y + shadow_offset + radius),
                              (x + shadow_offset + w, y + shadow_offset + h - radius), s_color, -1)

                cv2.ellipse(overlay, (x + shadow_offset + radius, y + shadow_offset + radius),
                            (radius, radius), 180, 0, 90, s_color, -1)
                cv2.ellipse(overlay, (x + shadow_offset + w - radius, y + shadow_offset + radius),
                            (radius, radius), 270, 0, 90, s_color, -1)
                cv2.ellipse(overlay, (x + shadow_offset + radius, y + shadow_offset + h - radius),
                            (radius, radius), 90, 0, 90, s_color, -1)
                cv2.ellipse(overlay, (x + shadow_offset + w - radius, y + shadow_offset + h - radius),
                            (radius, radius), 0, 0, 90, s_color, -1)

                return overlay

            box_colors = []
            for label, value in zip(labels, values):
                if label == "Final Decision":
                    color = self.bottom_box_color_out if value.lower() == "out" else self.bottom_box_color_not_out
                else:
                    color = self.top_box_color
                box_colors.append(color)

            for i in range(len(labels)):
                y = y_start + i * (box_height + spacing)
                frame = draw_rounded_box(frame, x, y, box_width, box_height, (0, 0, 0), radius=15, shadow=True)
                frame = draw_rounded_box(frame, x, y, box_width, box_height, box_colors[i], radius=15)
                cv2.putText(frame, labels[i], (x + 20, y + 20), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
                cv2.putText(frame, values[i], (x + 20, y + box_height - 10), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

            for _ in range(pause_frames):
                self.out.write(frame)

        self.out.release()
        print(f"Output video saved as {self.output_path}")

    def run(self):
        self.draw_overlay()
        self.display_decision()


if __name__ == "__main__":
    renderer = TrajectoryOverlayRenderer(
        video_path="input_video3.avi",
        json_path="ball_trajectory.json",
        output_path="output_video3.avi"
    )
    renderer.run()
