import json
from pathlib import Path
from trajectory_analysis_module import LBWPredictor

DATA_DIR = Path("./data")
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"    
INPUT_FILENAME = "sample_inputs.json"
OUTPUT_FILENAME = "sample_outputs.json"

def load_input_data(filepath: Path) -> dict:
    with filepath.open('r') as f:
        return json.load(f)

def save_output_data(filepath: Path, data: dict):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filepath}")

def main():
    input_path = INPUT_DIR / INPUT_FILENAME
    input_data = load_input_data(input_path)
    
    predictor = LBWPredictor()
    results = {}
    
    # Process each scenario separately
    for scenario_name, scenario_data in input_data.items():
        print(f"Processing {scenario_name}...")
        result = predictor.process_input(scenario_data)
        results[scenario_name] = result
    
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    save_output_data(output_path, results)

if __name__ == "__main__":
    main()