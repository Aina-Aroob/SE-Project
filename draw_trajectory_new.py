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
        self.decision = data.get("decision")
        metadata = data.get("metadata", {})

        # Metadata default values
        self.trajectory_color = tuple(metadata.get("trajectory_color", [255, 0, 0]))
        self.trajectory_thickness = metadata.get("trajectory_thickness", 14)  # widened
        self.ball_dot_radius = metadata.get("ball_dot_radius", 6)
        self.bounce_color = tuple(metadata.get("bounce_color", [0, 255, 255]))
        self.impact_color = tuple(metadata.get("impact_color", [0, 0, 255]))
        self.marker_radius = metadata.get("marker_radius", 7)  # slightly smaller
        self.top_box_color = tuple(metadata.get("decision_box_top_color", [255, 0, 0]))
        self.bottom_box_color_out = tuple(metadata.get("decision_box_bottom_color_out", [0, 0, 255]))
        self.bottom_box_color_not_out = tuple(metadata.get("decision_box_bottom_color_not_out", [0, 255, 0]))

    def _setup_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Use XVID codec for AVI output
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

    # Adding overlays here
    def draw_overlay(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or self.frame_idx >= len(self.trajectory):
                break

            overlay = frame.copy()
            point = self.trajectory[self.frame_idx]
            current_pos = (point['x'], point['y'])

            # Ball trajectory (semi-transparent thick blue lines)
            for i in range(1, self.frame_idx + 1):
                pt1 = (self.trajectory[i - 1]['x'], self.trajectory[i - 1]['y'])
                pt2 = (self.trajectory[i]['x'], self.trajectory[i]['y'])
                cv2.line(overlay, pt1, pt2, self.trajectory_color, self.trajectory_thickness)

            # Ball dot
            cv2.circle(overlay, current_pos, self.ball_dot_radius, self.trajectory_color, -1)

            # Bounce point
            if self.bounce_point:
                if current_pos == (self.bounce_point['x'], self.bounce_point['y']):
                    self.bounce_shown = True
                if self.bounce_shown:
                    cv2.circle(overlay, (self.bounce_point['x'], self.bounce_point['y']), self.marker_radius, self.bounce_color, -1)

            # Impact point
            if self.impact_point:
                if current_pos == (self.impact_point['x'], self.impact_point['y']):
                    self.impact_shown = True
                if self.impact_shown:
                    cv2.circle(overlay, (self.impact_point['x'], self.impact_point['y']), self.marker_radius, self.impact_color, -1)

            # Blend overlay
            alpha = 0.3  # More translucent
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            self.last_frame = frame.copy()
            self.out.write(frame)
            self.frame_idx += 1

        self.cap.release()

    # Displaying decision
    def display_decision(self):
        if self.decision and self.last_frame is not None:
            pause_frames = int(self.fps * 5)
            frame = self.last_frame.copy()

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.5
            thickness = 3

            text1 = "Decision"
            text2 = self.decision
            size1, _ = cv2.getTextSize(text1, font, scale, thickness)
            size2, _ = cv2.getTextSize(text2, font, scale, thickness)

            x = frame.shape[1] - size1[0] - 100
            y1 = ((frame.shape[0] + size1[1]) // 2) - size1[1] - 20
            y2 = (frame.shape[0] + size2[1]) // 2

            # Top box
            cv2.rectangle(frame, (x - 10, y1 - size1[1] - 10), (x + size1[0] + 10, y1 + 10), self.top_box_color, -1)
            cv2.putText(frame, text1, (x, y1), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # Bottom box based on decision
            bottom_color = self.bottom_box_color_out if text2.lower() == "out" else self.bottom_box_color_not_out
            cv2.rectangle(frame, (x - 10, y2 - size2[1] - 10), (x + size2[0] + 10, y2 + 10), bottom_color, -1)
            cv2.putText(frame, text2, (x, y2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

            for _ in range(pause_frames):
                self.out.write(frame)

        self.out.release()
        print(f"Output video saved as {self.output_path}")

    def run(self):
        self.draw_overlay()
        self.display_decision()


if __name__ == "__main__":
    renderer = TrajectoryOverlayRenderer(
        video_path="input_video3.avi",             # Input .avi video
        json_path="ball_trajectory.json",         # JSON with trajectory
        output_path="output_video3.avi"            # Output .avi video
    )
    renderer.run()
