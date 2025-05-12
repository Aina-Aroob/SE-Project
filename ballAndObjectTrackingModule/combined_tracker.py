import os
import sys
import json
from datetime import datetime
import glob # To find generated files

# Ensure the module path is correctly handled if this script itself is run
# or imported from elsewhere.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.append(_CURRENT_DIR)

# Import the tracker classes from within the same module
try:
    from WicketDetection import WicketDetector3D
    from detection import YoloBallTracker
    from legdetectionwork import PoseLegTracker
    from bat_tracking import BatTracker
except ImportError as e:
    print(f"Error importing tracker classes within combined_tracker: {e}")
    # This suggests an issue with the environment or how the module is structured/called.
    raise

# Define the intermediate directory relative to this file
INTERMEDIATE_DIR = os.path.join(_CURRENT_DIR, "..", "ballTrackingIntermediate")
# Ensure the intermediate directory path is absolute and normalized
INTERMEDIATE_DIR = os.path.abspath(INTERMEDIATE_DIR)

def track_objects(video_path, output_dir="final_tracking_output"):
    """
    Runs all tracking modules (Wicket, Ball, Leg/Pose, Bat) on a video,
    merges their JSON outputs, saves the combined result, and cleans up.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the final merged JSON file.
                          Defaults to "final_tracking_output" in the parent
                          directory of the module.

    Returns:
        str: The path to the final merged JSON file, or None if failed.
    """
    print(f"Starting combined tracking for: {video_path}")

    # --- 1. Validate Input and Setup Paths ---
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None

    # Ensure intermediate and final output directories exist
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    # Make output_dir relative to the caller's location or provide an absolute path.
    # For simplicity, let's create it relative to the project root (one level above module).
    project_root = os.path.dirname(_CURRENT_DIR)
    final_output_dir = os.path.join(project_root, output_dir)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Intermediate files will be in: {INTERMEDIATE_DIR}")
    print(f"Final output will be in: {final_output_dir}")

    # Store paths of generated JSON files for later merging and cleanup
    generated_json_files = {}

    # --- 2. Run Individual Trackers --- 
    trackers_to_run = [
        ("Wicket", WicketDetector3D, "detect_wickets"),
        ("Ball", YoloBallTracker, "process"),
        ("Leg/Pose", PoseLegTracker, "process"),
        ("Bat", BatTracker, "process")
    ]

    for name, TrackerClass, process_method_name in trackers_to_run:
        print(f"\n--- Running {name} Tracking --- ")
        try:
            # Instantiate with video path and force output to intermediate dir
            tracker_instance = TrackerClass(video_path=video_path, output_dir=INTERMEDIATE_DIR)
            
            # Get the process method and call it
            process_method = getattr(tracker_instance, process_method_name)
            process_method()
            
            # Store the path to the generated JSON file
            if hasattr(tracker_instance, 'out_json_path') and os.path.exists(tracker_instance.out_json_path):
                 generated_json_files[name] = tracker_instance.out_json_path
                 print(f"--- {name} Tracking Finished. Output: {tracker_instance.out_json_path} ---")
            else:
                 print(f"--- {name} Tracking finished, but output JSON path not found or file doesn't exist. ---")
                 # Attempt to find it if the attribute wasn't standard (less robust)
                 pattern = os.path.join(INTERMEDIATE_DIR, f"{name.lower().split('/')[0]}*.json")
                 found_files = glob.glob(pattern)
                 if found_files:
                    # Get the most recently created file matching the pattern
                    latest_file = max(found_files, key=os.path.getctime)
                    generated_json_files[name] = latest_file
                    print(f"    (Found potential output: {latest_file})" )
                 else:
                     print(f"    (Could not find any output JSON matching {pattern})" )

        except Exception as e:
            print(f"!!! Error during {name} Tracking: {e} !!!")
            # Decide whether to continue or stop
            # For now, we print the error and continue to get partial results

    # --- 3. Merge JSON Outputs --- 
    print("\n--- Merging Results --- ")
    if not generated_json_files:
        print("Error: No intermediate JSON files were generated or found. Cannot merge.")
        return None
        
    merged_data = {}
    all_frame_ids = set()

    for name, json_path in generated_json_files.items():
        print(f"Reading {name} data from: {json_path}")
        try:
            with open(json_path, 'r') as f:
                tracker_data = json.load(f)
                
            if not isinstance(tracker_data, list):
                 print(f"Warning: Expected a list in {json_path}, got {type(tracker_data)}. Skipping file.")
                 continue
                 
            # Process data from this tracker
            for frame_info in tracker_data:
                if not isinstance(frame_info, dict) or "frame_id" not in frame_info:
                    print(f"Warning: Skipping invalid frame data in {json_path}: {frame_info}")
                    continue
                    
                frame_id = frame_info["frame_id"]
                all_frame_ids.add(frame_id)
                
                # Initialize frame in merged dict if not present
                if frame_id not in merged_data:
                    merged_data[frame_id] = {
                        "frame_id": frame_id,
                        "ball": None,
                        "bat": None,
                        "leg": None,
                        "stumps": None,
                        "batsman_orientation": None
                    }
                
                # Merge non-null data from current tracker into the master frame
                # Update only if the tracker provided data for this object
                if name == "Ball" and frame_info.get("ball") is not None:
                    merged_data[frame_id]["ball"] = frame_info["ball"]
                elif name == "Bat" and frame_info.get("bat") is not None:
                    merged_data[frame_id]["bat"] = frame_info["bat"]
                elif name == "Leg/Pose":
                    if frame_info.get("leg") is not None:
                        merged_data[frame_id]["leg"] = frame_info["leg"]
                    # Leg/Pose tracker also provides orientation
                    if frame_info.get("batsman_orientation") is not None:
                         merged_data[frame_id]["batsman_orientation"] = frame_info["batsman_orientation"]
                elif name == "Wicket" and frame_info.get("stumps") is not None:
                     merged_data[frame_id]["stumps"] = frame_info["stumps"]
                     
        except FileNotFoundError:
            print(f"Warning: Intermediate file not found: {json_path}")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {json_path}")
        except Exception as e:
            print(f"Warning: An unexpected error occurred reading {json_path}: {e}")

    if not merged_data:
        print("Error: No valid frame data found after attempting to read intermediate files.")
        return None
        
    # Convert merged dictionary to a list sorted by frame_id
    final_data_list = sorted(merged_data.values(), key=lambda x: x["frame_id"])

    # --- 4. Save Final Merged JSON --- 
    final_json_filename = f"combined_tracking.json"
    final_json_path = os.path.join(final_output_dir, final_json_filename)

    print(f"\nSaving merged data to: {final_json_path}")
    try:
        with open(final_json_path, 'w') as f:
            json.dump(final_data_list, f, indent=2)
        print("Successfully saved merged JSON.")
    except Exception as e:
        print(f"Error saving final JSON file: {e}")
        # Don't cleanup if saving failed
        return None 

    # --- 5. Cleanup Intermediate Files --- 
    print("\nCleaning up intermediate files...")
    cleanup_count = 0
    for name, json_path in generated_json_files.items():
        try:
            os.remove(json_path)
            print(f"  Removed: {os.path.basename(json_path)}")
            cleanup_count += 1
        except OSError as e:
            print(f"Warning: Could not remove intermediate file {json_path}: {e}")
            
    print(f"Cleanup complete. Removed {cleanup_count} intermediate files.")

    return final_json_path

# Example of how to use this function if the script is run directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run combined tracking and merging process.')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--output', default="final_tracking_output", 
                        help='Directory to save the final merged JSON (relative to project root).')
    
    args = parser.parse_args()

    # Check if the video exists before calling
    if not os.path.exists(args.video_path):
         print(f"FATAL ERROR: Video file specified does not exist: {args.video_path}")
         # Try to provide context if path is relative
         if not os.path.isabs(args.video_path):
             script_location = os.path.dirname(os.path.abspath(__file__))
             print(f"       (Searched relative to script location: {script_location})")
             # Also check relative to current working directory
             print(f"       (Current working directory: {os.getcwd()})" )
         sys.exit(1)

    # Call the main function
    final_file = track_objects(args.video_path, output_dir=args.output)

    if final_file:
        print(f"\nProcess finished successfully. Final data is in: {final_file}")
    else:
        print("\nProcess finished with errors.") 