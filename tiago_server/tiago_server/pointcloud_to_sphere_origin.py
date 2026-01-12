#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from std_msgs.msg import Float64MultiArray
import tf
from geometry_msgs.msg import PointStamped
from sklearn.cluster import DBSCAN, KMeans
from scipy.optimize import leastsq
from visualization_msgs.msg import Marker, MarkerArray
import struct
from cv_bridge import CvBridge
import open3d as o3d

class PointCloudToSpheres:
    def __init__(self):
        rospy.init_node('pointcloud_to_spheres')
        self.bridge = CvBridge()
        self.camera_info = None
        self.rgb_image = None
        self.depth_image = None
        self.last_depth_time = None

        # Topics (private params)
        # Example:
        #   rosrun tiago_safety pointcloud_to_sphere.py \
        #     _rgb_image_topic:=/xtion/rgb/image_raw \
        #     _depth_image_topic:=/xtion/depth_registered/image_raw \
        #     _camera_info_topic:=/xtion/rgb/camera_info
        self.rgb_image_topic = rospy.get_param('~rgb_image_topic', '/head_front_camera/rgb/image_raw')
        self.depth_image_topic = rospy.get_param('~depth_image_topic', '/head_front_camera/depth/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/head_front_camera/rgb/camera_info')
        self.camera_frame = rospy.get_param('~camera_frame', 'head_front_camera_optical_frame')

        # Open3D voxel downsample
        self.use_open3d_voxel = rospy.get_param('~use_open3d_voxel', True)
        self.voxel_size = float(rospy.get_param('~voxel_size', 0.02))

        # TF (ROS1)
        self.tf_listener = tf.TransformListener()

        self.sphere_pub = rospy.Publisher('/detected_spheres', Float64MultiArray, queue_size=10)
        
        # RViz publishers
        self.marker_pub = rospy.Publisher('/sphere_markers', MarkerArray, queue_size=10)
        self.pointcloud_pub = rospy.Publisher('/camera_pointcloud', PointCloud2, queue_size=10)

        # Subscriptions
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self.camera_info_callback,
            queue_size=10)

        self.rgb_sub = rospy.Subscriber(
            self.rgb_image_topic,
            Image,
            self.rgb_callback,
            queue_size=10)

        self.depth_sub = rospy.Subscriber(
            self.depth_image_topic,
            Image,
            self.depth_callback,
            queue_size=10)

        # Timer: build pointcloud & spheres
        self.timer = rospy.Timer(rospy.Duration(0.5), self.process_pointcloud)

        rospy.loginfo('Subscriptions:')
        rospy.loginfo('  - rgb_image:    %s', self.rgb_image_topic)
        rospy.loginfo('  - depth_image:  %s', self.depth_image_topic)
        rospy.loginfo('  - camera_info:  %s', self.camera_info_topic)
        rospy.loginfo('  - camera_frame: %s', self.camera_frame)
        rospy.loginfo('  - voxel_size:   %s', self.voxel_size)
        rospy.loginfo('  - open3d_voxel: %s', self.use_open3d_voxel)
        rospy.loginfo('Waiting for camera data...')
        rospy.loginfo('RViz topics:')
        rospy.loginfo('  - markers: /sphere_markers (MarkerArray)')
        rospy.loginfo('  - cloud:   /camera_pointcloud (PointCloud2)')

        # Predefined arm spheres (for filtering in torso_lift_link)
        self.arm_right_link_names = [
            'arm_right_1_link',
            'arm_right_2_link',
            'arm_right_3_link',
            'arm_right_4_link',
            'arm_right_5_link',
            'arm_right_6_link',
            'arm_right_7_link',
        ]
        self.sphere_offsets = [
            [(0.0, 0.0, 0.0)],
            [(0.0, 0.0, 0.0)],
            [(0.0, 0.0, 0.0), (0.0, 0.0, -0.08), (0.0, 0.0, -0.16)],
            [(0.0, 0.01, 0.02), (-0.08, 0.02, 0.01)],
            [(0.0, 0.0, 0.02), (0.0, 0.0, 0.08)],
            [(0.09, 0.0, 0.0), (0.15, 0.0, 0.0)],
            [(0.0, 0.0, 0.0)],
        ]
        self.sphere_radii = [
            0.08,
            0.07,
            0.07, 0.07, 0.07,
            0.08, 0.07,
            0.07, 0.07,
            0.07, 0.07,
            0.07,
        ]

    @staticmethod
    def voxel_downsample(points, colors=None, voxel_size=0.02):
        """Voxel-grid downsample; keeps one point per voxel."""
        if points is None or len(points) == 0:
            return points, colors

        # Invalid voxel size
        if voxel_size is None or voxel_size <= 0:
            return points, colors

        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return points, colors

        voxel_idx = np.floor(pts / float(voxel_size)).astype(np.int32)
        _, unique_indices = np.unique(voxel_idx, axis=0, return_index=True)
        unique_indices.sort()

        pts_ds = pts[unique_indices]
        if colors is None:
            return pts_ds, None

        cols = np.asarray(colors)
        if len(cols) != len(pts):
            return pts_ds, None
        return pts_ds, cols[unique_indices]

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            rospy.loginfo('Received camera intrinsics: fx=%.3f fy=%.3f cx=%.3f cy=%.3f',
                          msg.K[0], msg.K[4], msg.K[2], msg.K[5])
            # Unsubscribe after first valid message
            self.camera_info_sub.unregister()
            rospy.loginfo('Unsubscribed from camera_info (latched).')

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rospy.loginfo_once('RGB image received.')
        except Exception as e:
            rospy.logerr('RGB conversion failed: %s', str(e))

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.last_depth_time = msg.header.stamp
            rospy.loginfo_once('Depth image received.')
            rospy.logdebug('Depth stamp: %s', str(self.last_depth_time))
        except Exception as e:
            rospy.logerr('Depth conversion failed: %s', str(e))

    def generate_pointcloud(self):
        if self.camera_info is None or self.depth_image is None:
            missing = []
            if self.camera_info is None:
                missing.append('camera_info')
            if self.depth_image is None:
                missing.append('depth_image')
            rospy.logwarn_throttle(5.0, 'Missing %s; skipping point cloud generation.', ', '.join(missing))
            return None, None

        height, width = self.depth_image.shape
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        # Pixel grid
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = self.depth_image

        # Keep valid, near-range points
        valid = (z > 0) & (z < 0.87) & (np.isfinite(z))
        z = z[valid]
        x =  (u[valid] - cx) * z / fx
        y =  (v[valid] - cy) * z / fy

        points = np.vstack((x, y, z)).T
        
        if len(points) == 0:
            rospy.logdebug_throttle(2.0, 'Valid points: 0')
            return None, None

        rospy.logdebug_throttle(2.0, 'Valid points: %d, depth range: %.2fm-%.2fm', len(points), float(z.min()), float(z.max()))

        # Optional color
        colors = None
        if self.rgb_image is not None:
            rgb_flat = self.rgb_image[valid] / 255.0
            colors = rgb_flat[:, [2, 1, 0]]  # BGR -> RGB

        return points, colors

    def transform_center(self, center, target_frame='torso_lift_link'):
        """Transform a 3D point from camera frame to target_frame."""
        point_stamped = PointStamped()
        point_stamped.header.frame_id = self.camera_frame
        # Use depth stamp for TF consistency
        stamp = self.last_depth_time if self.last_depth_time else rospy.Time(0)
        point_stamped.header.stamp = stamp
        point_stamped.point.x = center[0]
        point_stamped.point.y = center[1]
        point_stamped.point.z = center[2]
        try:
            self.tf_listener.waitForTransform(
                target_frame, self.camera_frame,
                stamp, rospy.Duration(1.0)
            )
            transformed_point = self.tf_listener.transformPoint(target_frame, point_stamped)
            rospy.logdebug('TF: %s -> [%.3f %.3f %.3f] (%s)',
                          str(center),
                          transformed_point.point.x, transformed_point.point.y, transformed_point.point.z,
                          target_frame)
            return [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn('TF transform failed: %s', str(e))
            return None

    def fit_sphere(self, points):
        """Least-squares sphere fit; returns (center_xyz, radius)."""
        def sphere_func(c, x, y, z):
            return np.sqrt((x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2) - c[3]

        center_init = np.mean(points, axis=0)
        radius_init = np.mean(np.linalg.norm(points - center_init, axis=1))
        params_init = np.append(center_init, radius_init)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        try:
            params, success = leastsq(sphere_func, params_init, args=(x, y, z), maxfev=5000)
            if success not in [1, 2, 3, 4]:
                rospy.logwarn('Sphere fit did not converge (status=%s).', str(success))
                return center_init, 999.0
        except Exception as e:
            rospy.logwarn('Sphere fit failed: %s', str(e))
            return center_init, 999.0

        radius = abs(params[3])
        
        max_extent = np.max(np.linalg.norm(points - params[:3], axis=1))
        if radius > max_extent * 2.0 or radius > 1.0:
            rospy.logdebug('Unreasonable radius (r=%.3fm, extent=%.3fm); using bounding radius.', float(radius), float(max_extent))
            radius = max_extent
        
        return params[:3], radius

    def fit_and_subdivide(self, cluster_points, sphere_data, depth=0, max_depth=5, num_sub_clusters=3):
        if depth > max_depth:
            rospy.logdebug('Reached max recursion depth; stop subdividing.')
            return
        if len(cluster_points) < 10:
            return
        center, radius = self.fit_sphere(cluster_points)
        
        if radius < 0.05:
            sphere_data.append((center[0], center[1], center[2], radius))
            rospy.logdebug('Sphere: center=%s r=%.3f', np.array2string(np.asarray(center), precision=3), float(radius))
        else:
            rospy.logdebug('Sphere too large (r=%.3f); subdividing...', float(radius))
            kmeans = KMeans(n_clusters=num_sub_clusters, n_init=10).fit(cluster_points)
            sub_labels = kmeans.labels_
            sub_unique_labels = np.unique(sub_labels)
            for sub_label in sub_unique_labels:
                sub_cluster_points = cluster_points[sub_labels == sub_label]
                self.fit_and_subdivide(sub_cluster_points, sphere_data, depth + 1, max_depth, num_sub_clusters + 1)

    # Predefined spheres in torso_lift_link (used to filter arm points)
    def get_predefined_spheres(self, stamp=None):
        if stamp is None:
            stamp = rospy.Time(0)
            
        positions = []
        radii = []
        radius_idx = 0
        target_frame = 'torso_lift_link'
        
        for link_name, offsets in zip(self.arm_right_link_names, self.sphere_offsets):
            for offset in offsets:
                pt = PointStamped()
                pt.header.frame_id = link_name
                pt.header.stamp = stamp
                pt.point.x, pt.point.y, pt.point.z = offset
                try:
                    self.tf_listener.waitForTransform(
                        target_frame, link_name, pt.header.stamp, rospy.Duration(0.1)
                    )
                    pt_transformed = self.tf_listener.transformPoint(target_frame, pt)
                    pos = [pt_transformed.point.x, pt_transformed.point.y, pt_transformed.point.z]
                    positions.append(pos)
                    radii.append(self.sphere_radii[radius_idx])
                    radius_idx += 1
                except Exception as e:
                    radius_idx += 1
        
        rospy.logdebug('Predefined arm spheres: %d/%d transformed into %s.',
                   len(positions), len(self.sphere_radii), target_frame)
        return np.array(positions), np.array(radii)

    def publish_pointcloud(self, points, colors, frame_id=None):
        """Publish a point cloud for RViz."""
        if points is None or len(points) == 0:
            return

        if frame_id is None:
            frame_id = self.camera_frame
        
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        
        cloud_data = []
        for i in range(len(points)):
            x, y, z = points[i]
            if colors is not None and i < len(colors):
                r, g, b = (colors[i] * 255).astype(np.uint8)
            else:
                r, g, b = 255, 255, 255
            
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
            cloud_data.append(struct.pack('fffI', x, y, z, rgb))
        
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True
        cloud_msg.data = b''.join(cloud_data)
        
        self.pointcloud_pub.publish(cloud_msg)
    
    def publish_sphere_markers(self, spheres, frame_id=None):
        """Publish sphere markers for RViz."""
        if frame_id is None:
            frame_id = self.camera_frame

        marker_array = MarkerArray()
        
        # Delete all previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Add current sphere centers
        for i, (center, radius) in enumerate(spheres):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "sphere_centers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = center[2]
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.05  # 5cm
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker.lifetime = rospy.Duration(2.0)
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
        rospy.logdebug_throttle(1.0, 'Published %d markers.', len(spheres))

    def process_pointcloud(self, event=None):
        # Generate point cloud from depth
        points, colors = self.generate_pointcloud()
        if points is None:
            return

        rospy.logdebug_throttle(1.0, 'Raw points: %d', len(points))

        # Publish a thin point cloud preview
        if len(points) > 1000:
            indices = np.random.choice(len(points), 1000, replace=False)
            self.publish_pointcloud(points[indices], colors[indices] if colors is not None else None)
        else:
            self.publish_pointcloud(points, colors)

        # 1) Voxel downsample
        pcd = None
        if self.use_open3d_voxel:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            points = np.asarray(pcd.points)
            if colors is not None and len(pcd.colors) == len(pcd.points):
                colors = np.asarray(pcd.colors)
            else:
                colors = None
        else:
            points, colors = self.voxel_downsample(points, colors, voxel_size=self.voxel_size)
        
        rospy.loginfo(f'Points after downsample : {len(points)}')

        if len(points) < 50:
            rospy.logwarn_throttle(2.0, 'Too few points after downsample (%d); skipping.', len(points))
            return

        # 2) DBSCAN clustering
        db = DBSCAN(eps=0.10, min_samples=50).fit(points)
        labels = db.labels_
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]

        rospy.loginfo_throttle(1.0, 'DBSCAN clusters: %d', len(unique_labels))

        if len(unique_labels) == 0:
            rospy.logwarn_throttle(2.0, 'No valid DBSCAN clusters; skipping.')
            return

        sphere_data = []  # spheres in camera frame
        for label in unique_labels:
            cluster_points = points[labels == label]
            rospy.logdebug('Cluster %s: %d points', str(label), len(cluster_points))
            
            self.fit_and_subdivide(cluster_points, sphere_data)

        # Transform centers into torso_lift_link (for safety_filter compatibility)
        target_frame = 'torso_lift_link'
        rospy.logdebug('Spheres in camera frame: %d; transforming into %s...', len(sphere_data), target_frame)
        
        transformed_sphere_data = []
        
        for center_x, center_y, center_z, radius in sphere_data:
            center_torso = self.transform_center([center_x, center_y, center_z], target_frame)
            if center_torso is not None:
                transformed_sphere_data.append((center_torso[0], center_torso[1], center_torso[2], radius))
            else:
                rospy.logdebug('Skipping sphere: TF transform failed.')
        
        rospy.logdebug('Transformed spheres: %d', len(transformed_sphere_data))

        # Predefined spheres at the same timestamp
        stamp = self.last_depth_time if self.last_depth_time else rospy.Time(0)
        predefined_positions, predefined_radii = self.get_predefined_spheres(stamp)
        
        if predefined_positions.size == 0:
            rospy.logwarn_throttle(2.0, 'No predefined arm spheres available; filtering disabled.')

        # Filter detections overlapping the predefined arm spheres
        filtered_sphere_data = []
        num_filtered = 0
        
        for detect_x, detect_y, detect_z, detect_r in transformed_sphere_data:
            is_overlapping = False
            detect_center = np.array([detect_x, detect_y, detect_z])
            
            if predefined_positions.size > 0:
                for i in range(len(predefined_positions)):
                    pre_center = predefined_positions[i]
                    pre_r = predefined_radii[i]
                    dist = np.linalg.norm(detect_center - pre_center)
                    
                    overlap_threshold = (pre_r + detect_r) * 1.2
                    
                    if dist < overlap_threshold:
                        is_overlapping = True
                        rospy.logdebug('Filtered overlap: det=%s r=%.3f vs arm=%s r=%.3f (d=%.3f < %.3f)',
                                      np.array2string(detect_center, precision=3), float(detect_r),
                                      np.array2string(pre_center, precision=3), float(pre_r),
                                      float(dist), float(overlap_threshold))
                        num_filtered += 1
                        break
                        
            if not is_overlapping:
                filtered_sphere_data.append((detect_x, detect_y, detect_z, detect_r))
        
        rospy.loginfo_throttle(1.0, 'Spheres: camera=%d torso=%d filtered=%d published=%d',
                       len(sphere_data), len(transformed_sphere_data), num_filtered, len(filtered_sphere_data))

        # Publish filtered spheres (no NMS)
        nms_spheres = filtered_sphere_data

        sphere_msg = Float64MultiArray()
        flat_data = []
        for center_x, center_y, center_z, radius in nms_spheres:
            flat_data.extend([center_x, center_y, center_z, radius])
        sphere_msg.data = flat_data
        self.sphere_pub.publish(sphere_msg)
        rospy.logdebug_throttle(1.0, 'Published %d spheres to /detected_spheres.', len(nms_spheres))
        
        # RViz markers (torso_lift_link)
        if len(nms_spheres) > 0:
            sphere_markers = [((x, y, z), r) for x, y, z, r in nms_spheres]
            self.publish_sphere_markers(sphere_markers, frame_id='torso_lift_link')

def main(args=None):
    try:
        node = PointCloudToSpheres()
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("PointCloudToSpheres stopped by user")
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()