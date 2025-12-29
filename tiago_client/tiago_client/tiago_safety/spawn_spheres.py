#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnModel  # ROS1 service type (SpawnModel.srv)

class SphereSpawner:
    def __init__(self):
        rospy.init_node('sphere_spawner')

        # Create service proxy
        rospy.wait_for_service('/gazebo/spawn_sdf_model')  # SDF model service
        self.client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # Spawn spheres
        self.spawn_spheres()

    def spawn_spheres(self):
        # Define 10 sphere positions (relative to base_footprint), more dispersed to avoid collision
        spheres = [
            {'name': 'green_ball1', 'x': 0.5, 'y': -0.2, 'z': 1.0},
            {'name': 'green_ball2', 'x': 0.3, 'y': -0.4, 'z': 1.0},
            {'name': 'green_ball3', 'x': 0.7, 'y': -0.1, 'z': 1.0},
            {'name': 'green_ball4', 'x': 0.2, 'y': -0.5, 'z': 1.0},
            {'name': 'green_ball5', 'x': 0.8, 'y': 0.0, 'z': 1.0},
            {'name': 'green_ball6', 'x': 0.4, 'y': -0.3, 'z': 1.0},
            {'name': 'green_ball7', 'x': 0.6, 'y': -0.05, 'z': 1.0},
            {'name': 'green_ball8', 'x': 0.25, 'y': -0.35, 'z': 1.0},
            {'name': 'green_ball9', 'x': 0.75, 'y': 0.05, 'z': 1.0},
            {'name': 'green_ball10', 'x': 0.55, 'y': -0.45, 'z': 1.0}
        ]

        model_path = '/home/rhino/tiago_dual_public_ws/src/pal_gazebo_worlds/models/green_ball/model.sdf'
        try:
            with open(model_path, 'r') as f:
                sdf_content = f.read()
            rospy.loginfo(f'SDF file loaded successfully: {model_path}')
        except IOError as e:
            rospy.logerr(f'Failed to read SDF file {model_path}: {e}')
            return

        success_count = 0
        for sphere in spheres:
            pose = Pose()
            pose.position.x = sphere['x']
            pose.position.y = sphere['y']
            pose.position.z = sphere['z']

            try:
                # ROS1 SpawnModel call: model_name, model_xml, robot_namespace, initial_pose, reference_frame
                resp = self.client(sphere['name'], sdf_content, '', pose, 'world')  # reference_frame='world' for SDF
                if resp.success:
                    rospy.loginfo(f'Successfully spawned sphere {sphere["name"]} at ({sphere["x"]}, {sphere["y"]}, {sphere["z"]}) relative to base_footprint')
                    success_count += 1
                else:
                    rospy.logerr(f'Failed to spawn sphere {sphere["name"]}: {resp.status_message}')
            except rospy.ServiceException as e:
                rospy.logerr(f'Failed to call /gazebo/spawn_sdf_model service: {e}')

        rospy.loginfo(f'Total spawned {success_count}/10 spheres')

def main():
    try:
        spawner = SphereSpawner()
        # rospy.spin()  # Keep node running if needed
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()