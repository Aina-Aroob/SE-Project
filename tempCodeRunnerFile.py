        self.top_box_color = tuple(metadata.get("decision_box_top_color", [0, 0, 255]))
        self.bottom_box_color_out = tuple(metadata.get("decision_box_bottom_color_out", [255, 0, 0]))
        self.bottom_box_color_not_out = tuple(metadata.get("decision_box_bottom_color_not_out", [255, 0, 0]))
