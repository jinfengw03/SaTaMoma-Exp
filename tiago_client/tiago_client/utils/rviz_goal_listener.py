# -*- coding: utf-8 -*-

"""RViz goal listener (ROS1).

Listens to RViz-published goal topics and exposes the latest goal point expressed in a
chosen target frame (default: `torso_lift_link`).

Topics supported:
- `/clicked_point` (geometry_msgs/PointStamped) via RViz "Publish Point" tool
- `/move_base_simple/goal` (geometry_msgs/PoseStamped) via RViz "2D Nav Goal" tool

This module is intentionally optional: if ROS is not available in the runtime
environment, import will fail and callers should disable RViz goal mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import rospy
import tf
from geometry_msgs.msg import PointStamped, PoseStamped


@dataclass
class RvizGoalListenerConfig:
    target_frame: str = "torso_lift_link"
    clicked_point_topic: str = "/clicked_point"
    nav_goal_topic: str = "/move_base_simple/goal"

    tf_wait_sec: float = 0.2
    max_age_sec: float = 10.0


class RvizGoalListener:
    def __init__(self, config: Optional[RvizGoalListenerConfig] = None):
        self.config = config or RvizGoalListenerConfig()

        # Make sure ROS is initialized (do not steal SIGINT).
        if not rospy.core.is_initialized():
            rospy.init_node("rviz_goal_listener", anonymous=True, disable_signals=True)

        self._tf = tf.TransformListener()
        self._last_goal_xyz = None  # type: Optional[np.ndarray]
        self._last_goal_time_wall = None  # type: Optional[float]
        self._last_source_frame = None  # type: Optional[str]

        self._sub_clicked = rospy.Subscriber(
            self.config.clicked_point_topic,
            PointStamped,
            self._on_clicked_point,
            queue_size=1,
        )
        self._sub_nav_goal = rospy.Subscriber(
            self.config.nav_goal_topic,
            PoseStamped,
            self._on_nav_goal,
            queue_size=1,
        )

    @property
    def last_source_frame(self) -> Optional[str]:
        return self._last_source_frame

    def _transform_point_to_target(self, pt: PointStamped) -> Optional[np.ndarray]:
        src = pt.header.frame_id or ""
        if src == "":
            return None

        try:
            self._tf.waitForTransform(self.config.target_frame, src, rospy.Time(0), rospy.Duration(self.config.tf_wait_sec))
            pt_t = self._tf.transformPoint(self.config.target_frame, pt)
            return np.array([pt_t.point.x, pt_t.point.y, pt_t.point.z], dtype=float)
        except Exception:
            return None

    def _record_goal(self, xyz: np.ndarray, source_frame: str) -> None:
        self._last_goal_xyz = np.asarray(xyz, dtype=float).reshape(3)
        self._last_goal_time_wall = time.time()
        self._last_source_frame = source_frame

    def _on_clicked_point(self, msg: PointStamped) -> None:
        xyz = self._transform_point_to_target(msg)
        if xyz is None:
            return
        self._record_goal(xyz, msg.header.frame_id)

    def _on_nav_goal(self, msg: PoseStamped) -> None:
        pt = PointStamped()
        pt.header = msg.header
        pt.point = msg.pose.position
        xyz = self._transform_point_to_target(pt)
        if xyz is None:
            return
        self._record_goal(xyz, msg.header.frame_id)

    def get_goal_xyz(self) -> Optional[np.ndarray]:
        """Returns the latest goal in target_frame, or None if missing/stale."""
        if self._last_goal_xyz is None or self._last_goal_time_wall is None:
            return None
        if (time.time() - self._last_goal_time_wall) > float(self.config.max_age_sec):
            return None
        return self._last_goal_xyz.copy()
