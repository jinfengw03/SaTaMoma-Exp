import time
import numpy as np
import sys
import os

# Ensure tiago_client is in path
sys.path.append(os.getcwd())

from tiago_client.tiago_client import TiagoClient

def main():
    # Helper to clear terminal
    def clear():
        os.system('cls' if os.name == 'nt' else 'clear')

    print("Connecting to Tiago Server...")
    # Initialize client (disable teleop as we only want to read state)
    client = TiagoClient(server_url="http://192.168.0.110:1234/", use_teleop=False)
    
    print("\nStarting Perception Monitor...")
    print("This script will print the obstacles detected by the robot.")
    print("Press Ctrl+C to stop.\n")
    time.sleep(2)

    try:
        while True:
            start_time = time.time()
            
            # Fetch state without heavy visual data (images)
            # This is fast and contains the 'obstacles' field we added
            state = client.get_state_wo_vis()
            
            obstacles = state.get('obstacles')
            
            clear()
            print(f"--- Perception Monitor (Updated: {time.strftime('%H:%M:%S')}) ---")
            
            if obstacles is None:
                print("\n[!] 'obstacles' key not found in server response.")
                print("    Did you update tiago_server_node.py on the server?")
            elif len(obstacles) == 0:
                print("\n[?] No obstacles detected.")
                print("    - Is the camera blocked?")
                print("    - Is the pointcloud_to_sphere.py node running?")
                print("    - Are objects within range (< 0.87m)?")
            else:
                print(f"\n[+] Detected {len(obstacles)} spheres (torso_lift_link frame):")
                print(f"{'ID':<4} {'X (m)':<10} {'Y (m)':<10} {'Z (m)':<10} {'Radius (m)':<10}")
                print("-" * 50)
                
                # obstacles is a list of [x, y, z, r] lists
                for i, obs in enumerate(obstacles):
                    x, y, z, r = obs
                    print(f"{i:<4} {x:<10.3f} {y:<10.3f} {z:<10.3f} {r:<10.3f}")

            # Refresh rate ~5Hz
            time.sleep(0.2)
            
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
