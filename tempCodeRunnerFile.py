    def draw_overlay(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or self.frame_idx >= len(self.trajectory):
                break

            overlay = frame.copy()
            point = self.trajectory[self.frame_idx]
            current_pos = (point['x'], point['y'])

            # Draw trajectory so far
            for i in range(1, self.frame_idx + 1):
                pt1 = (self.trajectory[i - 1]['x'], self.trajectory[i - 1]['y'])
                pt2 = (self.trajectory[i]['x'], self.trajectory[i]['y'])
                cv2.line(overlay, pt1, pt2, self.trajectory_color, self.trajectory_thickness)

            # Ball dot
            cv2.circle(overlay, current_pos, self.ball_dot_radius, self.trajectory_color, -1)

            # Show bounce marker if bounce point has already occurred
            if self.bounce_point and self.bounce_shown:
                cv2.circle(overlay, (self.bounce_point['x'], self.bounce_point['y']),
                        self.marker_radius, self.bounce_color, -1)

            # Show impact marker if impact point has already occurred
            if self.impact_point and self.impact_shown:
                cv2.circle(overlay, (self.impact_point['x'], self.impact_point['y']),
                        self.marker_radius, self.impact_color, -1)

            # Blend overlay with original frame
            alpha = 0.3
            blended_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Slow motion and pause at bounce point (only once)
            if self.bounce_point and not self.bounce_shown and \
            current_pos == (self.bounce_point['x'], self.bounce_point['y']):
                slow_start = max(self.frame_idx - 5, 0)
                self.write_slow_motion_segment(slow_start, self.frame_idx + 1, slow_factor=5)
                self.pause_at_frame(blended_frame, 0.2)
                self.bounce_shown = True

            # Slow motion and pause at impact point (only once)
            if self.impact_point and not self.impact_shown and \
            current_pos == (self.impact_point['x'], self.impact_point['y']):
                slow_start = max(self.frame_idx - 5, 0)
                self.write_slow_motion_segment(slow_start, self.frame_idx + 1, slow_factor=5)
                self.pause_at_frame(blended_frame, 0.2)
                self.impact_shown = True

            self.last_frame = blended_frame.copy()
            self.out.write(blended_frame)
            self.frame_idx += 1

        self.cap.release()
