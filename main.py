#!/usr/bin/env python3

import json
import argparse
import sys
from lbw_predictor import LBWPredictor


def load_input_data(input_file: str) -> dict:
    """Load input data from a JSON file."""
    try:
        with open(input_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="LBW Prediction System")
    parser.add_argument(
        "input_file", help="Path to input JSON file containing trajectory data"
    )
    parser.add_argument("--output", "-o", help="Path to output JSON file (optional)")

    args = parser.parse_args()

    try:
        input_data = load_input_data(args.input_file)

        predictor = LBWPredictor()

        result = predictor.process_input(input_data)

        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nPrediction saved to {args.output}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
