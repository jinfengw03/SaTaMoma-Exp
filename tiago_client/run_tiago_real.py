import time
import numpy as np
from tiago_client.tiago_client.tiago_client import TiagoClient

def main():
    # Initialize the real robot client
    # Default URL is http://192.168.0.110:1234/
    client = TiagoClient(server_url="http://192.168.0.110:1234/", use_teleop=True)
    
    print("\n[REAL] Teleoperation started.")
    print("Controls:")
    print("- Right Trigger: Close Right Gripper")
    print("- Left Trigger: Close Left Gripper")
    print("- B Button: Reset Pose")
    print("- Start/Menu: Exit\n")

    try:
        while True:
            start_time = time.time()
            
            # 1. Get action from Oculus VR (includes IK and Safety Filter)
            # is_filter=True enables the teleop policy's internal smoothing
            action, buttons = client.get_teleop_action(is_filter=True)
            
            if action is not None:
                # 2. Send action to the robot via HTTP POST
                obs, info = client.step(action)
                
                # Optional: Handle specific button presses
                if buttons.get('B'):
                    print("[REAL] Resetting robot pose...")
                    # Define a default reset pose if needed
                    # client.reset(default_reset_pose)
            
            # Maintain control frequency (approx 20Hz)
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.05 - elapsed))
            
    except KeyboardInterrupt:
        print("\n[REAL] Shutting down...")
    finally:
        client.close()

if __name__ == "__main__":
    main()
