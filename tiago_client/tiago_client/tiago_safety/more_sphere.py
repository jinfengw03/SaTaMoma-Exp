#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray
import tf
from geometry_msgs.msg import PointStamped

class SpherePublisher:
    def __init__(self):
        rospy.init_node('sphere_pub')

        self.pub = rospy.Publisher('/detected_spheres', Float64MultiArray, queue_size=10)
        self.tf_listener = tf.TransformListener()
        
        timer_period = 1.0  # Publish interval 1 second
        rospy.Timer(rospy.Duration(timer_period), self.publish_ten_spheres)  # ROS1 Timer

        rospy.loginfo('Sphere Publisher Node started (publishing in torso_lift_link frame)')
        rospy.spin()  # Keep node running

    def transform_to_torso_lift_link(self, x, y, z):
        """Transform point from base_footprint frame to torso_lift_link frame"""
        point = PointStamped()
        point.header.frame_id = 'base_footprint'
        point.header.stamp = rospy.Time(0)  # Use latest available transform
        point.point.x = x
        point.point.y = y
        point.point.z = z
        
        try:
            self.tf_listener.waitForTransform('torso_lift_link', 'base_footprint', 
                                             rospy.Time(0), rospy.Duration(1.0))
            transformed_point = self.tf_listener.transformPoint('torso_lift_link', point)
            return transformed_point.point.x, transformed_point.point.y, transformed_point.point.z
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f'TF transform failed: {e}')
            return x, y, z  # Return original coordinates on failure

    def publish_ten_spheres(self, event):
        # Original sphere positions (base_footprint frame)
        spheres_base = [
            (0.5, -0.2, 1.0, 0.05),   # Sphere 1
            (0.3, -0.4, 1.0, 0.05),   # Sphere 2
            (0.7, -0.1, 1.0, 0.05),   # Sphere 3
            (0.2, -0.5, 1.0, 0.05),   # Sphere 4
            (0.8, 0.0, 1.0, 0.05),    # Sphere 5
            (0.4, -0.3, 1.0, 0.05),   # Sphere 6
            (0.6, -0.05, 1.0, 0.05),  # Sphere 7
            (0.25, -0.35, 1.0, 0.05), # Sphere 8
            (0.75, 0.05, 1.0, 0.05),  # Sphere 9
            (0.55, -0.45, 1.0, 0.05)  # Sphere 10
        ]
        
        # Transform to torso_lift_link frame
        spheres_torso = []
        for x, y, z, r in spheres_base:
            x_t, y_t, z_t = self.transform_to_torso_lift_link(x, y, z)
            spheres_torso.extend([x_t, y_t, z_t, r])
        
        msg = Float64MultiArray()
        msg.data = spheres_torso
        self.pub.publish(msg)
        rospy.loginfo(f'Published ten spheres (torso_lift_link frame): {len(spheres_base)} spheres')

def main():
    try:
        publisher = SpherePublisher()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()