import rospy
import time
from tiago_client.tiago_client_sim import TiagoClientSim

def main():
    # Initialize the simulation client (connects to Gazebo via ROS)
    client = TiagoClientSim(use_teleop=True)
    
    # Frequency for the control loop
    rate = rospy.Rate(20) # 20Hz
    
    print("\n[SIM] Simulation Teleoperation started.")
    print("Ensure Gazebo and TIAGO controllers are running.")
    print("Controls:")
    print("- Right Trigger: Close Right Gripper")
    print("- Left Trigger: Close Left Gripper")
    print("- Start/Menu: Exit\n")

    try:
        while not rospy.is_shutdown():
            # 1. Get action from Oculus VR (includes IK and Safety Filter)
            # is_filter=True enables the teleop policy's internal smoothing
            action, buttons = client.get_teleop_action(is_filter=True)
            
            if action is not None:
                # 2. Publish action to ROS topics
                obs, info = client.step(action)
            
            rate.sleep()
            
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\n[SIM] Shutting down...")

if __name__ == "__main__":
    main()
