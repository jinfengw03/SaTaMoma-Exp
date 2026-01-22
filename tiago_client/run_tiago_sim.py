import rospy
import time
import numpy as np
from std_msgs.msg import Float64MultiArray
from tiago_client.tiago_client_sim import TiagoClientSim
import os

class ObstacleMonitor:
    def __init__(self):
        self.obstacles = []
        self.sub = rospy.Subscriber('/detected_spheres', Float64MultiArray, self.callback)
        
    def callback(self, msg):
        # Data comes in as flat list [x,y,z,r, x,y,z,r, ...]
        # Reshape to list of lists [[x,y,z,r], ...]
        data = np.array(msg.data)
        if data.size > 0:
            self.obstacles = data.reshape(-1, 4).tolist()
        else:
            self.obstacles = []

def main():
    # Initialize the simulation client (connects to Gazebo via ROS)
    client = TiagoClientSim(use_teleop=True)

    rviz_goal = None
    if os.environ.get("GOAL_STEP_FROM_RVIZ", "0") == "1":
        try:
            from tiago_client.utils.rviz_goal_listener import RvizGoalListener

            rviz_goal = RvizGoalListener()
            print("[SIM] RViz goal listener enabled (/clicked_point, /move_base_simple/goal)")
        except Exception as exc:
            print(f"[SIM] RViz goal listener unavailable: {exc}")
            rviz_goal = None
    
    # Monitor obstacles from /detected_spheres
    obs_monitor = ObstacleMonitor()
    
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
            goal_step_target = rviz_goal.get_goal_xyz().tolist() if rviz_goal is not None else None

            # 1. Get action from Oculus VR (includes IK and Safety Filter)
            # Pass the latest obstacles to the safety filter
            action, buttons = client.get_teleop_action(
                is_filter=True, 
                obstacles=obs_monitor.obstacles,
                goal_step_enabled=(goal_step_target is not None),
                goal_step_target=goal_step_target,
            )
            
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
