import json
from pathlib import Path
from trajectory_analysis_module import LBWPredictor

DATA_DIR = Path("./data")
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
FILENAME = "sample_inputs_bounce.json"

def load_input_data(filepath: Path) -> dict:
    with filepath.open('r') as f:
        return json.load(f)

def save_output_data(filepath: Path, data: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filepath}")

def main():
    input_path = INPUT_DIR / FILENAME
    input_data = load_input_data(input_path)

    predictor = LBWPredictor()
    result = predictor.process_input(input_data)
    output_path = OUTPUT_DIR / FILENAME
    save_output_data(output_path, result)

if __name__ == "__main__":
    main()