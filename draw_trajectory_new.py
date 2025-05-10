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

        self.trajectory = [{"x": int(p[0]), "y": int(p[1])} for p in data4["predicted_path"]]

        ball_pitch_point = data5.get("BallPitchPoint")
        pad_impact_point = data5.get("PadImpactPoint")

        self.bounce_point = {"x": int(ball_pitch_point[0]), "y": int(ball_pitch_point[1])} if ball_pitch_point else None
        self.impact_point = {"x": int(pad_impact_point[0]), "y": int(pad_impact_point[1])} if pad_impact_point else None

        self.pitching_result = data5.get("BallPitch", "N/A")
        self.impact_result = data5.get("PadImpact", "N/A")
        self.wickets_result = "Hitting" if data5.get("HittingStumps", False) else "Missing"
        self.final_decision = data5.get("Decision", "N/A")

        self.trajectory_color = (255, 0, 0)
        self.trajectory_thickness = 14
        self.ball_dot_radius = 6
        self.bounce_color = (0, 255, 255)
        self.impact_color = (0, 0, 255)
        self.marker_radius = 7
        self.top_box_color = (255, 0, 0)
        self.bottom_box_color_out = (0, 0, 255)
        self.bottom_box_color_not_out = (0, 255, 0)

        self.bounce_index = self._find_closest_index(self.bounce_point) if self.bounce_point else -1
        self.impact_index = self._find_closest_index(self.impact_point) if self.impact_point else -1

    def _find_closest_index(self, target_point):
        min_dist = float('inf')
        min_idx = -1
        for i, pt in enumerate(self.trajectory):
            dist = (pt['x'] - target_point['x'])**2 + (pt['y'] - target_point['y'])**2
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return min_idx

    def _setup_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = self.original_fps
        self.out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            self.fps,
            (self.width, self.height)
        )
        self.frame_idx = 0
    
    def draw_gradient_box(self, frame, x, y, width, height, color_top, color_bottom):
        for i in range(height):
            # Calculate interpolation factor
            alpha = i / height
            # Interpolate between top and bottom colors
            color = [
                int((1 - alpha) * color_top[j] + alpha * color_bottom[j])
                for j in range(3)
            ]
            # Draw a 1-pixel-high line with the interpolated color
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
            "OUT": (0, 0, 255),             # Red
            "NOT OUT": (26, 158, 26),       # Green
            "HITTING": (0, 0, 255),         # Red
            "MISSING": (0, 0, 255),         # Red
            "IN-LINE": (26, 158, 26),       # Green
            "INLINE": (18, 118, 18),        # Green
            "OUTSIDE OFF": (0, 0, 255),     # Red
            "OUTSIDE LEG": (0, 0, 255),     # Red
            "N/A": (128, 128, 128)          # Grey
        }

        box_width = 180
        box_height = 30
        spacing = 12

        frame_width = frame.shape[1]  # Get the width of the video frame
        start_x = frame_width - box_width - 50  # 50 px right margin
        #start_x = 30
        start_y = frame.shape[0] // 5

        for i in range(len(labels)):
            y = start_y + i * (box_height * 2 + spacing)


            # ------------ Draw gradient label box (top to bottom blue fade)--------------------
            self.draw_gradient_box(frame, start_x, y, box_width, box_height,
                          color_top=(195, 47, 47), color_bottom=(84, 18, 18))
            
            # --------------Add Label text---------------------
            label_text = labels[i]
            label_size = cv2.getTextSize(label_text, font, label_font_scale, thickness)[0]
            label_x = start_x + (box_width - label_size[0]) // 2
            label_y = y + (box_height + label_size[1]) // 2 -2 
            cv2.putText(frame, label_text, (label_x, label_y), font, label_font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

            # --------------Draw Solid color value box below-----------------------
            y_val = y + box_height
            value = values[i]
            color = value_colors.get(value, (100, 100, 100))
            cv2.rectangle(frame, (start_x, y_val), (start_x + box_width, y_val + box_height), color, -1)

            # --------------Add Value text-------------------------------
            value_size = cv2.getTextSize(value, font, value_font_scale, thickness)[0]
            value_x = start_x + (box_width - value_size[0]) // 2
            value_y = y_val + (box_height + value_size[1]) // 2 -2 
            cv2.putText(frame, value, (value_x, value_y), font, value_font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            


        return frame

    # def draw_decision_boxes(self, frame):
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     label_font_scale = 0.7
    #     value_font_scale = 0.8
    #     thickness = 1

    #     labels = ["PITCHING", "IMPACT", "WICKETS", "FINAL DECISION"]
    #     values = [
    #         self.pitching_result.upper(),
    #         self.impact_result.upper(),
    #         self.wickets_result.upper(),
    #         self.final_decision.upper()
    #     ]

    #     label_color = (255, 0, 0)  # Blue
    #     value_colors = {
    #         "OUT": (0, 0, 255),         # Red
    #         "NOT OUT": (0, 255, 0),     # Green
    #         "HITTING": (0, 0, 255),
    #         "MISSING": (128, 128, 128),
    #         "IN-LINE": (0, 0, 255),
    #         "OUTSIDE OFF": (255, 0, 0),
    #         "OUTSIDE LEG": (255, 0, 0),
    #         "N/A": (128, 128, 128)
    #     }

    #     box_width = 280
    #     box_height = 60
    #     spacing = 10

    #     start_x = 40
    #     start_y = frame.shape[0] // 4

    #     for i in range(len(labels)):
    #         y = start_y + i * (box_height + spacing)
    #         # Draw label box (top)
    #         cv2.rectangle(frame, (start_x, y), (start_x + box_width, y + box_height // 2), label_color, -1)
    #         label_size = cv2.getTextSize(labels[i], font, label_font_scale, thickness)[0]
    #         label_x = start_x + (box_width - label_size[0]) // 2
    #         label_y = y + (box_height // 4 + label_size[1] // 2)
    #         cv2.putText(frame, labels[i], (label_x, label_y), font, label_font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    #         # Determine value box color
    #         value = values[i]
    #         value_color = value_colors.get(value.upper(), (100, 100, 100)) if labels[i] != "ORIGINAL DECISION" else (
    #             (0, 0, 255) if value == "OUT" else (0, 255, 0)
    #         )

    #         # Draw value box (bottom)
    #         y_bottom = y + box_height // 2
    #         cv2.rectangle(frame, (start_x, y_bottom), (start_x + box_width, y_bottom + box_height // 2), value_color, -1)
    #         value_size = cv2.getTextSize(value, font, value_font_scale, thickness)[0]
    #         value_x = start_x + (box_width - value_size[0]) // 2
    #         value_y = y_bottom + (box_height // 4 + value_size[1] // 2)
    #         cv2.putText(frame, value, (value_x, value_y), font, value_font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    #     return frame





#------------------------1st one----------------------------------------
    # def draw_decision_boxes(self, frame):
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     scale = 0.6
    #     thickness = 1

    #     labels = ["Pitching", "Impact", "Wickets", "Final Decision"]
    #     values = [self.pitching_result, self.impact_result, self.wickets_result, self.final_decision]

    #     top_color = (255, 200, 100)
    #     bottom_color = (144, 238, 144)

    #     text_sizes = [cv2.getTextSize(label, font, scale, thickness)[0] for label in labels]
    #     value_sizes = [cv2.getTextSize(value, font, scale, thickness)[0] for value in values]

    #     box_width = max(max(t[0], v[0]) for t, v in zip(text_sizes, value_sizes)) + 40
    #     box_height = (text_sizes[0][1] + value_sizes[0][1]) + 30
    #     spacing = 10

    #     x = frame.shape[1] - box_width - 40
    #     y_start = (frame.shape[0] // 2) - ((box_height + spacing) * len(labels)) // 2

    #     for i in range(len(labels)):
    #         y = y_start + i * (box_height + spacing)
    #         top_height = int(box_height * 0.45)
    #         bottom_height = box_height - top_height

    #         # Top half
    #         cv2.rectangle(frame, (x, y), (x + box_width, y + top_height), top_color, -1)
    #         text_size = cv2.getTextSize(labels[i], font, scale, thickness)[0]
    #         text_x = x + (box_width - text_size[0]) // 2
    #         text_y = y + (top_height + text_size[1]) // 2 - 4
    #         cv2.putText(frame, labels[i], (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    #         # Bottom half
    #         box_bottom_y = y + top_height
    #         value_color = bottom_color if labels[i] != "Final Decision" else (
    #             self.bottom_box_color_out if values[i].lower() == "out" else self.bottom_box_color_not_out
    #         )
    #         cv2.rectangle(frame, (x, box_bottom_y), (x + box_width, box_bottom_y + bottom_height), value_color, -1)
    #         value_size = cv2.getTextSize(values[i], font, scale, thickness)[0]
    #         value_x = x + (box_width - value_size[0]) // 2
    #         value_y = box_bottom_y + (bottom_height + value_size[1]) // 2 - 4
    #         cv2.putText(frame, values[i], (value_x, value_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    #     return frame

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

            if self.bounce_point and self.frame_idx >= self.bounce_index:
                cv2.circle(overlay, (self.bounce_point['x'], self.bounce_point['y']),
                           self.marker_radius, self.bounce_color, -1)

            if self.impact_point and self.frame_idx >= self.impact_index:
                cv2.circle(overlay, (self.impact_point['x'], self.impact_point['y']),
                           self.marker_radius, self.impact_color, -1)

            alpha = 0.3
            blended_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            frame_with_boxes = self.draw_decision_boxes(blended_frame)

            for _ in range(self.slow_factor):
                self.out.write(frame_with_boxes)

            self.frame_idx += 1

        self.cap.release()
        self.out.release()
        print(f"Output video saved as {self.output_path}")

    def run(self):
        self.draw_overlay()


if __name__ == "__main__":
    renderer = TrajectoryOverlayRenderer(
        video_path="input_video3.avi",
        module4_json="module4_output.json",
        module5_json="module5_output.json",
        output_path="output_video3.avi",
        slow_factor=3
    )
    renderer.run()
