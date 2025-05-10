# module4_interface.py
import json

def load_bounce_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get("bounce_point")
