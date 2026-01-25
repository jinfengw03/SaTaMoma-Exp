#!/usr/bin/env python

import sys
import json
import time
from pathlib import Path
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray
import tf
from geometry_msgs.msg import PointStamped
from sklearn.cluster import DBSCAN, KMeans
from scipy.optimize import leastsq
from visualization_msgs.msg import Marker, MarkerArray
import struct

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None

def sphere_nms(spheres, dist_thresh=0.02, radius_thresh=0.01):
    # spheres: [(x, y, z, r, score), ...] or [(x, y, z, r)]
    # If no score is provided, sort by radius desc.
    if len(spheres) == 0:
        return []
    if len(spheres[0]) == 5:
        spheres = sorted(spheres, key=lambda s: s[4], reverse=True)
    else:
        spheres = sorted(spheres, key=lambda s: s[3], reverse=True)
    keep = []
    for s in spheres:
        suppress = False
        for k in keep:
            dist = np.linalg.norm(np.array(s[:3]) - np.array(k[:3]))
            if dist < dist_thresh and abs(s[3] - k[3]) < radius_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(s)
    return keep

class PointCloudToSpheres:
    def __init__(self):
        rospy.init_node('pointcloud_to_spheres')
        self.bridge = CvBridge()
        self.pointcloud_data = None
        self.last_cloud_time = None

        # --- Offline / predefined environment mode ---
        # mode: live | record | playback
        self.mode = rospy.get_param('~mode', 'live').strip().lower()
        self.record_path = rospy.get_param('~record_path', str(Path.home() / 'tiago_predefined_spheres.json'))
        self.playback_rate_hz = float(rospy.get_param('~playback_rate_hz', 10.0))
        
        # Depth filtering params (camera frame)
        self.min_depth = float(rospy.get_param('~min_depth', 0.1))
        self.max_depth = float(rospy.get_param('~max_depth', 1.5))
        
        # Workspace height filtering (base frame)
        self.workspace_min_z = float(rospy.get_param('~workspace_min_z', 0.8))
        self.base_frame = rospy.get_param('~base_frame', 'base_footprint')

        self.xyz_offset = rospy.get_param('~xyz_offset', [0.0, 0.0, 0.0])
        self.radius_scale = float(rospy.get_param('~radius_scale', 1.0))
        self.radius_min = float(rospy.get_param('~radius_min', 0.0))
        self.radius_max = float(rospy.get_param('~radius_max', 10.0))
        self._playback_cache = None
        self._playback_mtime = None

        # Topics / frames (private params)
        self.pointcloud_topic = rospy.get_param('~pointcloud_topic', '/xtion/depth/points')
        self.camera_frame = rospy.get_param('~camera_frame', 'xtion_depth_optical_frame')

        # Open3D voxel downsample
        self.use_open3d_voxel = bool(rospy.get_param('~use_open3d_voxel', True))
        self.voxel_size = float(rospy.get_param('~voxel_size', 0.02))

        # TF (ROS1) - use lenient settings for real robot disruptions/latencies
        self.tf_timeout = float(rospy.get_param('~tf_timeout', 3.0))
        self.use_latest_tf = bool(rospy.get_param('~use_latest_tf', True))
        self.tf_listener = tf.TransformListener()

        self.sphere_pub = rospy.Publisher('/detected_spheres', Float64MultiArray, queue_size=10)
        
        # RViz publishers
        self.marker_pub = rospy.Publisher('/sphere_markers', MarkerArray, queue_size=10)
        self.pointcloud_pub = rospy.Publisher('/camera_pointcloud', PointCloud2, queue_size=10)

        if self.mode != 'playback':
            self.pointcloud_sub = rospy.Subscriber(
                self.pointcloud_topic,
                PointCloud2,
                self.pointcloud_callback,
                queue_size=10)

            self.timer = rospy.Timer(rospy.Duration(0.5), self.process_pointcloud)
        else:
            # Playback mode: no camera subscriptions
            self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.playback_rate_hz, 0.5)), self.publish_from_file)

        rospy.loginfo('Subscriptions:')
        if self.mode != 'playback':
            rospy.loginfo('  - pointcloud:   %s', self.pointcloud_topic)
        else:
            rospy.loginfo('  - playback_file: %s', self.record_path)
            rospy.loginfo('  - playback_rate: %.2f Hz', self.playback_rate_hz)
        rospy.loginfo('  - camera_frame: %s', self.camera_frame)
        rospy.loginfo('  - voxel_size:   %.3f', self.voxel_size)
        rospy.loginfo('  - open3d_voxel: %s', str(bool(self.use_open3d_voxel)))
        rospy.loginfo('  - mode:         %s', self.mode)
        rospy.loginfo('  - tf_timeout:   %.1fs', self.tf_timeout)
        rospy.loginfo('  - use_latest_tf: %s', str(self.use_latest_tf))
        rospy.loginfo('RViz topics:')
        rospy.loginfo('  - markers: /sphere_markers (MarkerArray)')
        rospy.loginfo('  - cloud:   /camera_pointcloud (PointCloud2)')

        # Predefined right-arm spheres (used to filter arm points in torso_lift_link)
        self.arm_right_link_names = [
            'arm_right_1_link',  # sphere_right_1
            'arm_right_2_link',  # sphere_right_2
            'arm_right_3_link',  # sphere_right_3, sphere_right_9, sphere_right_10
            'arm_right_4_link',  # sphere_right_4, sphere_right_8
            'arm_right_5_link',  # sphere_right_5, sphere_right_12
            'arm_right_6_link',  # sphere_right_6, sphere_right_11
            'arm_right_7_link',  # sphere_right_7
        ]
        self.sphere_offsets = [
            [(0.0, 0.0, 0.0)],                                 # arm_right_1_link
            [(0.0, 0.0, 0.0)],                                 # arm_right_2_link
            [(0.0, 0.0, 0.0), (0.0, 0.0, -0.08), (0.0, 0.0, -0.16)],  # arm_right_3_link
            [(0.0, 0.01, 0.02), (-0.08, 0.02, 0.01)],          # arm_right_4_link
            [(0.0, 0.0, 0.02), (0.0, 0.0, 0.08)],              # arm_right_5_link
            [(0.09, 0.0, 0.0), (0.15, 0.0, 0.0)],              # arm_right_6_link
            [(0.0, 0.0, 0.0)],                                 # arm_right_7_link
        ]
        self.sphere_radii = [
            0.08,      # sphere_right_1
            0.07,      # sphere_right_2
            0.07, 0.07, 0.07,  # sphere_right_3, sphere_right_9, sphere_right_10
            0.08, 0.07,        # sphere_right_4, sphere_right_8
            0.07, 0.07,        # sphere_right_5, sphere_right_12
            0.07, 0.07,        # sphere_right_6, sphere_right_11
            0.07               # sphere_right_7
        ]

    @staticmethod
    def voxel_downsample(points, colors=None, voxel_size=0.02):
        """Voxel-grid downsample; keeps one point per voxel."""
        if points is None or len(points) == 0:
            return points, colors

        if voxel_size is None or float(voxel_size) <= 0:
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

    def _apply_filters_and_adjustments(self, spheres_xyzr, frame_id='torso_lift_link'):
        """Apply user adjustments.
        (Height filtering is now done on points in base_frame, so z_min check is removed here)

        spheres_xyzr: list of (x, y, z, r)
        returns: filtered list of (x, y, z, r)
        """
        if spheres_xyzr is None:
            return []

        try:
            dx, dy, dz = [float(v) for v in self.xyz_offset]
        except Exception:
            dx, dy, dz = 0.0, 0.0, 0.0

        out = []
        for x, y, z, r in spheres_xyzr:
            x = float(x) + dx
            y = float(y) + dy
            z = float(z) + dz
            r = float(r) * float(self.radius_scale)
            r = max(self.radius_min, min(self.radius_max, r))
            
            # Note: We rely on workspace_min_z point filtering now.
            # If explicit z filtering in target frame is needed, add it here.
            
            out.append((x, y, z, r))
        return out

    def _save_spheres_json(self, spheres_xyzr, frame_id='torso_lift_link'):
        path = Path(self.record_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'frame_id': frame_id,
            'timestamp': time.time(),
            'xyz_offset': self.xyz_offset,
            'radius_scale': self.radius_scale,
            'radius_min': self.radius_min,
            'radius_max': self.radius_max,
            'spheres': [[float(x), float(y), float(z), float(r)] for x, y, z, r in spheres_xyzr],
        }
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        tmp.replace(path)

    def _load_spheres_json_cached(self):
        """Load spheres from record_path with mtime caching so edits take effect quickly."""
        path = Path(self.record_path)
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        if self._playback_cache is not None and self._playback_mtime == mtime:
            return self._playback_cache
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            rospy.logwarn(f'[PointCloudToSpheres] Failed to load spheres file: {path} err={e}')
            return None
        spheres = data.get('spheres', [])
        # Normalize to list of tuples
        xyzr = []
        for s in spheres:
            if isinstance(s, (list, tuple)) and len(s) >= 4:
                xyzr.append((s[0], s[1], s[2], s[3]))
        self._playback_cache = {
            'frame_id': data.get('frame_id', 'torso_lift_link'),
            'spheres_xyzr': xyzr,
        }
        self._playback_mtime = mtime
        return self._playback_cache

    def publish_from_file(self, event=None):
        loaded = self._load_spheres_json_cached()
        if loaded is None:
            rospy.logwarn_throttle(5.0, f'[PointCloudToSpheres] playback: no file {self.record_path}')
            return
        frame_id = loaded['frame_id']
        spheres_xyzr = self._apply_filters_and_adjustments(loaded['spheres_xyzr'], frame_id=frame_id)
        self._publish_spheres(spheres_xyzr, frame_id=frame_id)

    def _publish_spheres(self, spheres_xyzr, frame_id='torso_lift_link'):
        sphere_msg = Float64MultiArray()
        flat_data = []
        for x, y, z, r in spheres_xyzr:
            flat_data.extend([float(x), float(y), float(z), float(r)])
        sphere_msg.data = flat_data
        self.sphere_pub.publish(sphere_msg)
        if len(spheres_xyzr) > 0:
            sphere_markers = [((x, y, z), r) for x, y, z, r in spheres_xyzr]
            self.publish_sphere_markers(sphere_markers, frame_id=frame_id)

    def pointcloud_callback(self, msg):
        """Callback to receive PointCloud2 messages."""
        try:
            points_list = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = point
                if self.min_depth < z < self.max_depth and np.isfinite(x) and np.isfinite(y):
                    points_list.append([x, y, z])

            if len(points_list) > 0:
                self.pointcloud_data = {
                    'points': np.array(points_list),
                    'frame_id': msg.header.frame_id,
                    'stamp': msg.header.stamp
                }
                self.last_cloud_time = msg.header.stamp
                rospy.loginfo_once('PointCloud2 received: %d valid points', len(points_list))
                rospy.logdebug('PointCloud2: frame=%s points=%d', msg.header.frame_id, len(points_list))
            else:
                rospy.logwarn_throttle(5.0, 'No valid points in PointCloud2')
                self.pointcloud_data = None

        except Exception as e:
            rospy.logerr('PointCloud2 parsing failed: %s', str(e))

    def transform_points(self, points, target_frame, source_frame, stamp):
        """Transform numpy array of points (Nx3) from source_frame to target_frame."""
        if points is None or len(points) == 0:
            return points

        try:
            self.tf_listener.waitForTransform(target_frame, source_frame, stamp, rospy.Duration(0.5))
            trans, rot = self.tf_listener.lookupTransform(target_frame, source_frame, stamp)
            mat = self.tf_listener.fromTranslationRotation(trans, rot)
            
            # Homogeneous coordinates
            points_hom = np.hstack((points, np.ones((len(points), 1))))
            
            # Transform: P_new = mat * P_old
            # (4x4) * (4xN) = (4xN)
            points_new = np.dot(mat, points_hom.T).T
            
            return points_new[:, :3]
        except Exception as e:
            rospy.logwarn_throttle(2.0, '[transform_points] Failed to transform points: %s', str(e))
            return None

    def transform_center(self, center, target_frame='torso_lift_link', source_frame=None):
        """Transform a 3D point from camera frame (or source_frame) into target_frame."""
        if source_frame is None:
            source_frame = self.camera_frame
            
        point_stamped = PointStamped()
        point_stamped.header.frame_id = source_frame
        # Use latest transform if enabled, otherwise use exact timestamp
        if self.use_latest_tf:
            stamp = rospy.Time(0)  # Latest available transform
        else:
            stamp = self.last_cloud_time if self.last_cloud_time else rospy.Time(0)
        point_stamped.header.stamp = stamp
        point_stamped.point.x = center[0]
        point_stamped.point.y = center[1]
        point_stamped.point.z = center[2]
        try:
            self.tf_listener.waitForTransform(
                target_frame, source_frame,
                stamp, rospy.Duration(self.tf_timeout)
            )
            transformed_point = self.tf_listener.transformPoint(target_frame, point_stamped)
            rospy.logdebug('TF: %s -> [%.3f %.3f %.3f] (%s)',
                          str(center),
                          transformed_point.point.x, transformed_point.point.y, transformed_point.point.z,
                          target_frame)
            return [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, 'TF transform failed (%s->%s): %s', source_frame, target_frame, str(e))
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

    def fit_and_subdivide(self, cluster_points, sphere_data, new_spheres, depth=0, max_depth=5, num_sub_clusters=3):
        if depth > max_depth:
            rospy.logdebug('Reached max recursion depth; stop subdividing.')
            return
        if len(cluster_points) < 10:
            return
        center, radius = self.fit_sphere(cluster_points)
        
        if radius < 0.05:
            sphere_data.append((center[0], center[1], center[2], radius))
            rospy.logdebug('Sphere: center=%s r=%.3f', np.array2string(np.asarray(center), precision=3), float(radius))
            new_spheres.append((center, radius))
        else:
            rospy.logdebug('Sphere too large (r=%.3f); subdividing...', float(radius))
            kmeans = KMeans(n_clusters=num_sub_clusters, n_init=10).fit(cluster_points)
            sub_labels = kmeans.labels_
            sub_unique_labels = np.unique(sub_labels)
            for sub_label in sub_unique_labels:
                sub_cluster_points = cluster_points[sub_labels == sub_label]
                self.fit_and_subdivide(sub_cluster_points, sphere_data, new_spheres, depth + 1, max_depth, num_sub_clusters + 1)

    def get_predefined_spheres(self, stamp=None):
        # Always use latest transform for arm links (they move continuously)
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
                        target_frame, link_name, pt.header.stamp, rospy.Duration(self.tf_timeout)
                    )
                    pt_transformed = self.tf_listener.transformPoint(target_frame, pt)
                    pos = [pt_transformed.point.x, pt_transformed.point.y, pt_transformed.point.z]
                    positions.append(pos)
                    radii.append(self.sphere_radii[radius_idx])
                    radius_idx += 1
                except Exception as e:
                    rospy.logwarn_throttle(5.0, 'Failed to transform arm sphere %s: %s', link_name, str(e))
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
        
        rospy.loginfo('Publishing %d sphere markers in frame: %s', len(spheres), frame_id)
        
        marker_array = MarkerArray()
        
        # Delete all previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Add current sphere centers (using ACTUAL sphere radius for scale)
        for i, (center, radius) in enumerate(spheres):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time(0)  # Use latest transform
            marker.ns = "detected_spheres"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = center[2]
            marker.pose.orientation.w = 1.0
            
            # Use the actual sphere radius * 2 for diameter
            diameter = radius * 2.0
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = diameter
            
            # Semi-transparent red
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            
            marker.lifetime = rospy.Duration(5.0)  # Longer lifetime
            
            marker_array.markers.append(marker)
            
            rospy.loginfo('  Marker %d: pos=(%.3f, %.3f, %.3f) radius=%.3f frame=%s', 
                         i, center[0], center[1], center[2], radius, frame_id)
        
        self.marker_pub.publish(marker_array)
        rospy.loginfo('Published MarkerArray with %d markers to /sphere_markers', len(marker_array.markers))

    def process_pointcloud(self, event=None):  # ROS1 Timer callback requires the event arg
        if self.pointcloud_data is None:
            rospy.logwarn_throttle(5.0, 'No point cloud data available yet.')
            return

        points = self.pointcloud_data['points'].copy()
        colors = None

        rospy.logdebug_throttle(1.0, 'Raw points: %d', len(points))

        # Publish a thin point cloud preview for RViz
        if len(points) > 1000:
            indices = np.random.choice(len(points), 1000, replace=False)
            self.publish_pointcloud(points[indices], None, frame_id=self.camera_frame)
        else:
            self.publish_pointcloud(points, None, frame_id=self.camera_frame)

        # 1) Voxel downsample
        if self.use_open3d_voxel:
            if o3d is None:
                rospy.logwarn_throttle(5.0, 'open3d is not available; falling back to numpy voxel downsample.')
                points, colors = self.voxel_downsample(points, colors, voxel_size=self.voxel_size)
            else:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
                pcd = pcd.voxel_down_sample(voxel_size=float(self.voxel_size))
                points = np.asarray(pcd.points)
                if colors is not None and len(pcd.colors) == len(pcd.points):
                    colors = np.asarray(pcd.colors)
                else:
                    colors = None
        else:
            points, colors = self.voxel_downsample(points, colors, voxel_size=self.voxel_size)

        rospy.loginfo('Points after downsample: %d', len(points))

        if len(points) < 50:
            rospy.logwarn_throttle(2.0, 'Too few points after downsample (%d); skipping.', len(points))
            return

        # --- Base-Frame Height Filter (Workspace Restriction) ---
        processing_frame = self.camera_frame
        stamp = self.last_cloud_time if self.last_cloud_time else rospy.Time(0)

        if self.workspace_min_z is not None:
            transformed_points = self.transform_points(points, self.base_frame, self.camera_frame, stamp)

            if transformed_points is not None:
                mask = transformed_points[:, 2] > self.workspace_min_z
                filtered_points = transformed_points[mask]

                rospy.loginfo_throttle(1.0, 'Height filter (%s > %.2f): %d -> %d points',
                                       self.base_frame, self.workspace_min_z, len(points), len(filtered_points))

                points = filtered_points
                processing_frame = self.base_frame
            else:
                rospy.logwarn_throttle(2.0, 'Could not transform points to %s for height filtering.', self.base_frame)

        if len(points) < 50:
            rospy.logwarn_throttle(2.0, 'Too few points after height filter (%d); skipping.', len(points))
            return

        # 2) DBSCAN clustering
        db = DBSCAN(eps=0.10, min_samples=25).fit(points)
        labels = db.labels_
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]

        rospy.loginfo_throttle(1.0, 'DBSCAN clusters: %d', len(unique_labels))

        if len(unique_labels) == 0:
            rospy.logwarn_throttle(2.0, 'No valid DBSCAN clusters; skipping.')
            return

        sphere_data = []
        new_spheres = []
        for label in unique_labels:
            cluster_points = points[labels == label]
            rospy.logdebug('Cluster %s: %d points', str(label), len(cluster_points))
            self.fit_and_subdivide(cluster_points, sphere_data, new_spheres)

        # Transform centers into torso_lift_link (for safety_filter compatibility)
        target_frame = 'torso_lift_link'
        rospy.logdebug('Spheres in %s: %d; transforming into %s...', processing_frame, len(sphere_data), target_frame)

        transformed_sphere_data = []

        for center_x, center_y, center_z, radius in sphere_data:
            center_torso = self.transform_center([center_x, center_y, center_z], target_frame, source_frame=processing_frame)
            if center_torso is not None:
                transformed_sphere_data.append((center_torso[0], center_torso[1], center_torso[2], radius))
            else:
                rospy.logdebug('Skipping sphere: TF transform failed.')

        rospy.logdebug('Transformed spheres: %d', len(transformed_sphere_data))

        # Predefined spheres at the same timestamp
        stamp = self.last_cloud_time if self.last_cloud_time else rospy.Time(0)
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

        # Apply z-min and optional adjustments
        nms_spheres = self._apply_filters_and_adjustments(nms_spheres, frame_id='torso_lift_link')

        # Publish
        self._publish_spheres(nms_spheres, frame_id='torso_lift_link')
        rospy.logdebug_throttle(1.0, 'Published %d spheres to /detected_spheres.', len(nms_spheres))

        # Record offline spheres if enabled
        if self.mode == 'record':
            try:
                self._save_spheres_json(nms_spheres, frame_id='torso_lift_link')
                rospy.loginfo_throttle(2.0, '[PointCloudToSpheres] saved spheres -> %s', self.record_path)
            except Exception as e:
                rospy.logwarn_throttle(2.0, '[PointCloudToSpheres] save failed: %s', str(e))

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