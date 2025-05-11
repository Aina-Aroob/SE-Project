import json
import logging
from pathlib import Path
from typing import Dict, Any
from lbw_predictor import LBWPredictor
from utils import convert_position_meters_to_inches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data(input_file: str) -> Dict[str, Any]:
    """Load test data from a JSON file."""
    try:
        with open(input_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load test data from {input_file}: {str(e)}")
        raise

def save_test_output(result: Dict[str, Any], test_name: str) -> None:
    """Save test results to a JSON file."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Create filename from test name
        filename = test_name.lower().replace(" ", "_") + "_output.json"
        output_path = output_dir / filename
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Test results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save test results: {str(e)}")
        raise

def print_test_results(result: Dict[str, Any], test_name: str) -> None:
    """Print test results in a formatted way."""
    print(f"\n{'='*50}")
    print(f"Test Case: {test_name}")
    print(f"{'='*50}")
    
    # Print verdict
    print("\nLBW Prediction Results:")
    print("----------------------")
    print(f"Status: {result['verdict']['status']}")
    print(f"Will hit stumps: {result['verdict']['will_hit_stumps']}")
    
    # Print impact point if exists
    if result['verdict']['will_hit_stumps']:
        impact = result['verdict']['impact_point']
        print("\nImpact Point Details:")
        print(f"Coordinates (inches):")
        print(f"  X: {convert_position_meters_to_inches(impact['x']):.2f}")
        print(f"  Y: {convert_position_meters_to_inches(impact['y']):.2f}")
        print(f"  Z: {convert_position_meters_to_inches(impact['z']):.2f}")
        print(f"\nRelative Position:")
        print(f"  Height ratio: {impact['relative_height']:.2f}")
        print(f"  Width ratio: {impact['relative_width']:.2f}")
    
    print(f"\nConfidence: {result['verdict']['confidence']:.2f}")
    
    # Print bounce point if exists
    if result['bounce_point']:
        print("\nBounce Point Details:")
        print(f"Coordinates (inches):")
        print(f"  X: {convert_position_meters_to_inches(result['bounce_point']['pos_x']):.2f}")
        print(f"  Y: {convert_position_meters_to_inches(result['bounce_point']['pos_y']):.2f}")
        print(f"  Z: {convert_position_meters_to_inches(result['bounce_point']['pos_z']):.2f}")
        print(f"  Time: {result['bounce_point']['timestamp']:.2f}s")
    else:
        print("\nNo bounce point detected in trajectory")
    
    # Print swing characteristics
    print("\nSwing Characteristics:")
    print(f"Angle: {result['swing_characteristics']['swing_angle']:.2f} degrees")
    print(f"Magnitude: {convert_position_meters_to_inches(result['swing_characteristics']['swing_magnitude']):.2f} inches")
    print(f"Direction: {'Clockwise' if result['swing_characteristics']['swing_direction'] > 0 else 'Counter-clockwise'}")
    
    # Print trajectory points
    print("\nPredicted Trajectory (first 5 points):")
    print("-------------------------------------")
    for i, point in enumerate(result['predicted_path'][:5]):
        print(f"Point {i}:")
        print(f"  X: {convert_position_meters_to_inches(point['pos_x']):.2f}")
        print(f"  Y: {convert_position_meters_to_inches(point['pos_y']):.2f}")
        print(f"  Z: {convert_position_meters_to_inches(point['pos_z']):.2f}")
        print(f"  Time: {point['timestamp']:.2f}s")

def run_test_case(test_file: str, test_name: str) -> Dict[str, Any]:
    """Run a single test case and return the results."""
    try:
        logger.info(f"Running test case: {test_name}")
        input_data = load_test_data(test_file)
        
        predictor = LBWPredictor()
        result = predictor.process_input(input_data)
        
        print_test_results(result, test_name)
        
        # Save test results to JSON file
        save_test_output(result, test_name)
        
        return result
    except Exception as e:
        logger.error(f"Test case '{test_name}' failed: {str(e)}")
        raise

def verify_test_results(result: Dict[str, Any], test_name: str) -> None:
    """Verify test results meet expected criteria."""
    try:
        # Verify basic structure
        assert 'verdict' in result, "Missing verdict in results"
        assert 'predicted_path' in result, "Missing predicted path in results"
        assert 'swing_characteristics' in result, "Missing swing characteristics in results"
        
        # Verify verdict
        verdict = result['verdict']
        assert 'status' in verdict, "Missing status in verdict"
        assert 'will_hit_stumps' in verdict, "Missing will_hit_stumps in verdict"
        assert 'confidence' in verdict, "Missing confidence in verdict"
        
        # Verify predicted path
        path = result['predicted_path']
        assert len(path) > 0, "Empty predicted path"
        assert all('pos_x' in p and 'pos_y' in p and 'pos_z' in p for p in path), "Invalid point format in path"
        
        # Verify swing characteristics
        swing = result['swing_characteristics']
        assert 'swing_angle' in swing, "Missing swing angle"
        assert 'swing_magnitude' in swing, "Missing swing magnitude"
        assert 'swing_direction' in swing, "Missing swing direction"
        
        logger.info(f"Test case '{test_name}' passed verification")
    except AssertionError as e:
        logger.error(f"Test case '{test_name}' failed verification: {str(e)}")
        raise

def verify_edge_case_results(result: Dict[str, Any]) -> None:
    """Additional verification for edge case results."""
    try:
        # Verify high velocity handling
        path = result['predicted_path']
        for point in path:
            assert point['pos_x'] >= 0, "Invalid X position"
            assert abs(point['pos_y']) <= 1000, "Y position out of reasonable range"
            assert point['pos_z'] >= 0, "Invalid Z position"
        
        # Verify extreme angles
        swing = result['swing_characteristics']
        assert 0 <= swing['swing_angle'] <= 180, "Swing angle out of valid range"
        assert swing['swing_magnitude'] >= 0, "Invalid swing magnitude"
        assert swing['swing_direction'] in [-1, 0, 1], "Invalid swing direction"
        
        # Verify bounce point for high trajectories
        if result['bounce_point']:
            bounce = result['bounce_point']
            assert bounce['pos_y'] == 0, "Bounce point not at ground level"
            assert bounce['timestamp'] > 0, "Invalid bounce time"
        
        logger.info("Edge case verification passed")
    except AssertionError as e:
        logger.error(f"Edge case verification failed: {str(e)}")
        raise

def main():
    # Define test cases with their input files and names
    test_cases = [
        ('test_data/sample_inputs_hit.json', 'Direct Hit Test'),
        ('test_data/sample_inputs_bounce.json', 'Bounce Test'),
        ('test_data/sample_inputs_edge.json', 'Edge Case Test')
    ]
    
    try:
        # Create test_data directory if it doesn't exist
        test_data_dir = Path("test_data")
        test_data_dir.mkdir(exist_ok=True)
        
        # Run each test case
        for test_file, test_name in test_cases:
            result = run_test_case(test_file, test_name)
            verify_test_results(result, test_name)
            
            # Additional verification for edge cases
            if test_name == 'Edge Case Test':
                verify_edge_case_results(result)
                
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 