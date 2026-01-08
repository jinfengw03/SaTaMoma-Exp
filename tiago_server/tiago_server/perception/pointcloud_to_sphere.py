#!/usr/bin/env python

import sys
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Float64MultiArray
import tf
from geometry_msgs.msg import PointStamped
from sklearn.cluster import DBSCAN, KMeans
from scipy.optimize import leastsq
from visualization_msgs.msg import Marker, MarkerArray
import struct

def sphere_nms(spheres, dist_thresh=0.02, radius_thresh=0.01):
    # spheres: [(x, y, z, r, score), ...] or [(x, y, z, r)]
    # 如果没有score，按半径降序
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
        self.camera_info = None
        self.rgb_image = None
        self.depth_image = None
        self.last_depth_time = None

        # TF 设置（ROS1）
        self.tf_listener = tf.TransformListener()

        self.sphere_pub = rospy.Publisher('/detected_spheres', Float64MultiArray, queue_size=10)
        
        # RViz 可视化发布器
        self.marker_pub = rospy.Publisher('/sphere_markers', MarkerArray, queue_size=10)
        self.pointcloud_pub = rospy.Publisher('/camera_pointcloud', PointCloud2, queue_size=10)

        # 订阅相机信息（使用 xtion 相机）
        self.camera_info_sub = rospy.Subscriber(
            '/xtion/rgb/camera_info',
            CameraInfo,
            self.camera_info_callback,
            queue_size=10)

        # 订阅 RGB 图像
        self.rgb_sub = rospy.Subscriber(
            '/xtion/rgb/image_raw',
            Image,
            self.rgb_callback,
            queue_size=10)

        # 订阅深度图像（使用 depth_registered）
        self.depth_sub = rospy.Subscriber(
            '/xtion/depth_registered/image_raw',
            Image,
            self.depth_callback,
            queue_size=10)

        # 定时器：每 0.5 秒生成点云和球体
        self.timer = rospy.Timer(rospy.Duration(0.5), self.process_pointcloud)

        rospy.loginfo('订阅话题：/xtion/rgb/image_raw, /xtion/depth_registered/image_raw, /xtion/rgb/camera_info')
        rospy.loginfo('等待相机数据...')
        rospy.loginfo('RViz 可视化已启用：')
        rospy.loginfo('  - 球体中心: /sphere_markers (MarkerArray)')
        rospy.loginfo('  - 点云: /camera_pointcloud (PointCloud2)')
        rospy.loginfo('  - 打开 RViz 并添加这些话题进行可视化')

        # 预定义机械臂球体数据
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

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            rospy.loginfo('已接收相机内参: fx={}, fy={}, cx={}, cy={}'.format(
                msg.K[0], msg.K[4], msg.K[2], msg.K[5]))
            # 取消订阅
            self.camera_info_sub.unregister()
            rospy.loginfo('相机内参已获取，取消 /xtion/rgb/camera_info 订阅')

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rospy.loginfo_once('RGB图像接收成功！')
        except Exception as e:
            rospy.logerr('RGB 图像转换错误: {}'.format(e))

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.last_depth_time = msg.header.stamp
            rospy.loginfo_once('深度图像接收成功！')
            rospy.logdebug('深度图像时间戳: {}'.format(self.last_depth_time))
        except Exception as e:
            rospy.logerr('深度图像转换错误: {}'.format(e))

    def generate_pointcloud(self):
        if self.camera_info is None or self.depth_image is None:
            missing = []
            if self.camera_info is None:
                missing.append('相机内参')
            if self.depth_image is None:
                missing.append('深度图像')
            rospy.logwarn_throttle(5.0, '缺少: {}，跳过点云生成'.format(', '.join(missing)))
            return None, None

        height, width = self.depth_image.shape
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        # 创建像素网格
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = self.depth_image

        # 过滤无效深度，只保留距离小于 0.87m 的点（机器人工作空间范围）
        # 参考代码使用 z < 0.87 来聚焦于近距离障碍物检测
        valid = (z > 0) & (z < 0.87) & (np.isfinite(z))
        z = z[valid]
        x = - (u[valid] - cx) * z / fx
        y = - (v[valid] - cy) * z / fy

        points = np.vstack((x, y, z)).T
        
        if len(points) == 0:
            rospy.loginfo_throttle(2.0, '有效点数: 0')
            return None, None

        rospy.loginfo_throttle(2.0, f'有效点数: {len(points)}, 深度范围: {z.min():.2f}m - {z.max():.2f}m')

        # 添加颜色
        colors = None
        if self.rgb_image is not None:
            rgb_flat = self.rgb_image[valid] / 255.0
            colors = rgb_flat[:, [2, 1, 0]]  # RGB 顺序

        return points, colors

    def transform_center(self, center, target_frame='torso_lift_link'):
        """
        将球体中心从相机坐标系变换到目标坐标系
        使用 torso_lift_link 以与 safety_filter_right.py 保持一致
        """
        point_stamped = PointStamped()
        point_stamped.header.frame_id = 'xtion_rgb_optical_frame'  # 相机坐标系
        # 使用深度图像的时间戳，保证空间一致性
        stamp = self.last_depth_time if self.last_depth_time else rospy.Time(0)
        point_stamped.header.stamp = stamp
        point_stamped.point.x = center[0]
        point_stamped.point.y = center[1]
        point_stamped.point.z = center[2]
        try:
            # ROS1 TF: 变换到 torso_lift_link（与 safety_filter 一致）
            self.tf_listener.waitForTransform(
                target_frame, 'xtion_rgb_optical_frame',
                stamp, rospy.Duration(1.0)
            )
            transformed_point = self.tf_listener.transformPoint(target_frame, point_stamped)
            rospy.logdebug(f'TF 变换: {center} -> [{transformed_point.point.x:.3f}, {transformed_point.point.y:.3f}, {transformed_point.point.z:.3f}] ({target_frame})')
            return [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f'TF 变换错误: {e}')
            return None

    def fit_sphere(self, points):
        """最小二乘拟合球体：输入点云，返回中心(x, y, z)和半径r"""
        def sphere_func(c, x, y, z):
            return np.sqrt((x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2) - c[3]

        center_init = np.mean(points, axis=0)
        radius_init = np.mean(np.linalg.norm(points - center_init, axis=1))
        params_init = np.append(center_init, radius_init)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # 增加最大迭代次数，添加完整性检查
        try:
            params, success = leastsq(sphere_func, params_init, args=(x, y, z), maxfev=5000)
            if success not in [1, 2, 3, 4]:  # 检查收敛状态
                rospy.logwarn(f'球体拟合未收敛，状态码: {success}')
                # 返回一个大半径作为标记，让后续逻辑进行细分
                return center_init, 999.0
        except Exception as e:
            rospy.logwarn(f'球体拟合异常: {e}')
            return center_init, 999.0

        radius = abs(params[3])
        
        # 合理性检查：半径不应超过点云范围的2倍
        max_extent = np.max(np.linalg.norm(points - params[:3], axis=1))
        if radius > max_extent * 2.0 or radius > 1.0:  # 半径不应超过1米
            rospy.logdebug(f'拟合半径不合理 (r={radius:.3f}m, extent={max_extent:.3f}m)，返回包围球')
            # 使用包围球半径
            radius = max_extent
        
        return params[:3], radius

    def fit_and_subdivide(self, cluster_points, sphere_data, new_spheres, depth=0, max_depth=5, num_sub_clusters=3):
        if depth > max_depth:
            rospy.logdebug('达到最大递归深度，停止细分')
            return
        if len(cluster_points) < 10:  # 参考代码使用 10 个点作为最小要求
            return
        center, radius = self.fit_sphere(cluster_points)
        
        # 参考代码使用 0.05m 作为球体半径阈值
        if radius < 0.05:
            sphere_data.append((center[0], center[1], center[2], radius))
            rospy.loginfo(f'添加球体: 中心 {center}, 半径 {radius:.3f}')
            # new_spheres 现在存储中心位置和半径，用于可视化
            new_spheres.append((center, radius))
        else:
            rospy.loginfo(f'球体过大 (r={radius:.3f})，进行细分...')
            kmeans = KMeans(n_clusters=num_sub_clusters, n_init=10).fit(cluster_points)
            sub_labels = kmeans.labels_
            sub_unique_labels = np.unique(sub_labels)
            for sub_label in sub_unique_labels:
                sub_cluster_points = cluster_points[sub_labels == sub_label]
                # 参考代码：每次递归增加子簇数量 (num_sub_clusters + 1)
                self.fit_and_subdivide(sub_cluster_points, sphere_data, new_spheres, depth + 1, max_depth, num_sub_clusters + 1)

    # 新增方法：获取预定义机械臂球体在 torso_lift_link 帧下的位置和半径
    # 注意：使用 torso_lift_link 而非 base_footprint，以与 safety_filter_right.py 中的 CBF 坐标系保持一致
    def get_predefined_spheres(self, stamp=None):
        if stamp is None:
            stamp = rospy.Time(0)
            
        positions = []
        radii = []
        radius_idx = 0  # 用于遍历 sphere_radii
        target_frame = 'torso_lift_link'  # 与 safety_filter 的 root_frame 一致
        
        for link_name, offsets in zip(self.arm_right_link_names, self.sphere_offsets):
            for offset in offsets:
                pt = PointStamped()
                pt.header.frame_id = link_name
                pt.header.stamp = stamp  # 使用指定的时间戳
                pt.point.x, pt.point.y, pt.point.z = offset
                try:
                    self.tf_listener.waitForTransform(
                        target_frame, link_name, pt.header.stamp, rospy.Duration(0.1) # 减少等待时间
                    )
                    pt_transformed = self.tf_listener.transformPoint(target_frame, pt)
                    pos = [pt_transformed.point.x, pt_transformed.point.y, pt_transformed.point.z]
                    positions.append(pos)
                    radii.append(self.sphere_radii[radius_idx])
                    # rospy.logdebug(f'预定义球体 {radius_idx}: {link_name} -> {target_frame}: {pos}, r={self.sphere_radii[radius_idx]:.3f}')
                    radius_idx += 1
                except Exception as e:
                    # rospy.logwarn(f'TF变换失败 for predefined sphere in {link_name}: {e}')
                    radius_idx += 1  # 仍然增加索引，保持与 sphere_radii 同步
        
        rospy.loginfo(f'成功获取 {len(positions)} 个预定义机械臂球体在 {target_frame} 坐标系下（共 {len(self.sphere_radii)} 个）')
        return np.array(positions), np.array(radii)

    def publish_pointcloud(self, points, colors, frame_id='xtion_rgb_optical_frame'):
        """发布点云到 RViz"""
        if points is None or len(points) == 0:
            return
        
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
    
    def publish_sphere_markers(self, spheres, frame_id='xtion_rgb_optical_frame'):
        """发布球体中心标记到 RViz"""
        marker_array = MarkerArray()
        
        # 删除旧的标记
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # 添加新的球体中心标记
        for i, (center, radius) in enumerate(spheres):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "sphere_centers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # 设置位置
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = center[2]
            marker.pose.orientation.w = 1.0
            
            # 设置大小（显示为小球）
            marker.scale.x = 0.05  # 5cm 直径的标记
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            
            # 设置颜色（红色）
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker.lifetime = rospy.Duration(2.0)  # 1秒后自动消失
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
        rospy.loginfo(f'发布 {len(spheres)} 个球体中心标记到 RViz')

    def process_pointcloud(self, event=None):  # ROS1 Timer callback 需要 event 参数
        # 生成点云
        points, colors = self.generate_pointcloud()
        if points is None:
            return

        rospy.loginfo(f'原始点云点数: {len(points)}')

        # 发布原始点云到 RViz（下采样前）
        if len(points) > 1000:
            # 随机采样以减少数据量
            indices = np.random.choice(len(points), 1000, replace=False)
            self.publish_pointcloud(points[indices], colors[indices] if colors is not None else None)
        else:
            self.publish_pointcloud(points, colors)

        # 1. 体素下采样 - 参考代码使用 0.02m 体素大小（更大的下采样）
        voxel_size = 0.02
        voxel_dict = {}
        for i, point in enumerate(points):
            voxel_key = tuple((point / voxel_size).astype(int))
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = i
        
        downsampled_indices = list(voxel_dict.values())
        points = points[downsampled_indices]
        if colors is not None:
            colors = colors[downsampled_indices]
        
        rospy.loginfo(f'下采样后点数: {len(points)}')

        if len(points) < 50:
            rospy.logwarn(f'下采样后点云太少 ({len(points)} 点)，跳过处理')
            return

        # 2. DBSCAN聚类 - 参考代码使用 (ε = 0.10m, minPts = 50)
        # 更宽松的参数可以更好地识别较大的障碍物
        db = DBSCAN(eps=0.10, min_samples=50).fit(points)
        labels = db.labels_
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # 忽略噪声簇

        rospy.loginfo(f'DBSCAN找到 {len(unique_labels)} 个簇')

        if len(unique_labels) == 0:
            rospy.logwarn('DBSCAN未找到有效簇，跳过球体生成')
            return

        sphere_data = []  # 相机坐标系下的球体
        new_spheres = []
        for label in unique_labels:
            cluster_points = points[labels == label]
            rospy.loginfo(f'簇 {label}: {len(cluster_points)} 个点')
            
            # 参考代码中移除了平面检测，直接进行球体拟合和细分
            self.fit_and_subdivide(cluster_points, sphere_data, new_spheres)

        # 球体中心变换到 torso_lift_link 坐标系（与 safety_filter 的 CBF 计算一致）
        target_frame = 'torso_lift_link'
        rospy.loginfo(f'相机坐标系下生成 {len(sphere_data)} 个球体，开始变换到 {target_frame}...')
        
        transformed_sphere_data = []  # torso_lift_link 坐标系（用于过滤和发布）
        
        for center_x, center_y, center_z, radius in sphere_data:
            # 变换到 torso_lift_link
            center_torso = self.transform_center([center_x, center_y, center_z], target_frame)
            if center_torso is not None:
                transformed_sphere_data.append((center_torso[0], center_torso[1], center_torso[2], radius))
            else:
                rospy.logwarn(f'球体中心变换到 {target_frame} 失败，跳过此球体')
        
        rospy.loginfo(f'成功变换 {len(transformed_sphere_data)} 个球体到 {target_frame} 坐标系')

        # 新增：获取预定义球体（torso_lift_link 坐标系）
        # 使用深度图时间戳，确保过滤时机械臂位置与点云时刻一致
        stamp = self.last_depth_time if self.last_depth_time else rospy.Time(0)
        predefined_positions, predefined_radii = self.get_predefined_spheres(stamp)
        
        if predefined_positions.size == 0:
            rospy.logwarn('未获取到预定义球体，跳过过滤（将包含机械臂球体）')
        else:
            rospy.loginfo(f'获取到 {len(predefined_positions)} 个预定义球体，开始过滤...')

        # 新增：在 torso_lift_link 坐标系下过滤掉与预定义球体重叠的检测球体
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
                    
                    # 使用更宽松的重叠判定：距离 < (半径1 + 半径2) * 1.2
                    overlap_threshold = (pre_r + detect_r) * 1.2
                    
                    if dist < overlap_threshold:
                        is_overlapping = True
                        rospy.loginfo(f'排除重叠球体: 检测={detect_center.round(3)}, r={detect_r:.3f}; '
                                    f'预定义={pre_center.round(3)}, r={pre_r:.3f}; 距离={dist:.3f} < 阈值={overlap_threshold:.3f}')
                        num_filtered += 1
                        break
                        
            if not is_overlapping:
                filtered_sphere_data.append((detect_x, detect_y, detect_z, detect_r))
        
        rospy.loginfo(f'过滤结果: 原始 {len(transformed_sphere_data)} 个球体，过滤掉 {num_filtered} 个，剩余 {len(filtered_sphere_data)} 个')
        rospy.loginfo(f'发布球体到 /detected_spheres(torso_lift_link 坐标系)')

        # NMS去重（使用过滤后的数据）- 暂时注释掉
        # nms_spheres = sphere_nms(filtered_sphere_data, dist_thresh=0.02, radius_thresh=0.01)

        # 直接使用过滤后的数据，不进行NMS去重
        nms_spheres = filtered_sphere_data

        sphere_msg = Float64MultiArray()
        flat_data = []
        for center_x, center_y, center_z, radius in nms_spheres:
            flat_data.extend([center_x, center_y, center_z, radius])
        sphere_msg.data = flat_data
        self.sphere_pub.publish(sphere_msg)
        rospy.loginfo(f'已发布 {len(nms_spheres)} 个球体到 /detected_spheres')
        
        # 发布球体中心标记到 RViz（使用 torso_lift_link 坐标系，与球体数据坐标系一致）
        if len(nms_spheres) > 0:
            # 将 nms_spheres 转换为 (center, radius) 格式
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