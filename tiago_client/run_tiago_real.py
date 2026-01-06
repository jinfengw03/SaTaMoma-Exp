import time
import numpy as np
import cv2
from tiago_client.tiago_client import TiagoClient
from intent_prediction.intent_predictor_integrated import IntentPredictorIntegrated

def main():
    # Initialize the real robot client
    # Default URL is http://192.168.0.110:1234/
    client = TiagoClient(server_url="http://192.168.0.110:1234/", use_teleop=True)
    
    # Initialize Intent Predictor
    # Note: Ensure 'ollama' is installed and 'llava:7b' model is pulled
    predictor = IntentPredictorIntegrated(model_name='llava:7b', analysis_interval=5.0)
    
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
                
                # 3. Update Intent Predictor with latest observation
                # obs usually contains 'tiago_head_image' if configured in server
                if 'tiago_head_image' in obs:
                    # Assuming image is decoded or needs decoding. 
                    # If it's raw bytes/base64, it might need processing in TiagoClient first.
                    # Here we assume obs['tiago_head_image'] is a numpy array (H,W,3)
                    img = obs['tiago_head_image']
                    
                    # Get joint positions for context
                    joints = obs.get('right_joints', [])
                    base_vel = obs.get('base_velocity', [0, 0, 0])
                   
                    torso_val = obs.get('torso', 0)
                    if isinstance(torso_val, np.ndarray):
                        torso_val = float(torso_val.reshape(-1)[0]) if torso_val.size else 0.0
                    elif isinstance(torso_val, list):
                        torso_val = torso_val[0] if torso_val else 0.0

                    # Update predictor state (non-blocking)
                    predictor.update_state(image=img, joints=joints, base_vel=base_vel, torso=torso_val)
                
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
        predictor.stop()
    finally:
        client.close()

if __name__ == "__main__":
    main()
