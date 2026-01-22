# -*- coding: utf-8 -*-

"""Goal-step assistance for teleoperation.

This module implements the *idea* from `hybrid_teleop_goal_step.py` in a way that fits
SaTaMoma-Exp's client-side teleop pipeline:

- User provides a manual arm command (via VR/keyboard -> Cartesian delta -> IK -> safety).
- Once the commanded joint target is *reached* (within tolerance for a hold time), the
  end-effector takes *one extra small step* toward a fixed goal point.

It is intentionally ROS-agnostic: callers provide current joint/EEF state and decide
how to run IK and safety filtering.

All positions are assumed to be expressed in the same frame as the EEF pose provided
by the caller (in this repo, it's typically `torso_lift_link`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class GoalStepAssistConfig:
    enabled: bool = False

    step_size: float = 0.03
    stop_distance: float = 0.02
    assist_disable_distance: float = 0.15

    reach_tolerance: float = 0.03
    reach_timeout: float = 2.5
    reach_hold_time: float = 0.15

    eps: float = 1e-6


class GoalStepAssist:
    def __init__(self, config: GoalStepAssistConfig, goal_xyz: Optional[np.ndarray] = None):
        self.config = config
        self.goal_xyz = None if goal_xyz is None else np.asarray(goal_xyz, dtype=float).reshape(3)

        self._pending_joint_target: Optional[np.ndarray] = None
        self._pending_gripper: float = 0.0
        self._armed_time: Optional[float] = None
        self._reached_since: Optional[float] = None
        self._stepped: bool = False

    def set_goal_xyz(self, goal_xyz: Optional[np.ndarray]) -> None:
        self.goal_xyz = None if goal_xyz is None else np.asarray(goal_xyz, dtype=float).reshape(3)

    def reset(self) -> None:
        self._pending_joint_target = None
        self._armed_time = None
        self._reached_since = None
        self._stepped = False

    def notify_manual_arm_command(self, joint_target: np.ndarray, gripper_val: float, now: Optional[float] = None) -> None:
        """Call this when you *send* a new manual arm joint command."""
        now = time.time() if now is None else float(now)
        self._pending_joint_target = np.asarray(joint_target, dtype=float).copy()
        self._pending_gripper = float(gripper_val)
        self._armed_time = now
        self._reached_since = None
        self._stepped = False

    def maybe_compute_goal_step(
        self,
        current_joints: np.ndarray,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        now: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """If conditions are met, returns (target_pos, target_quat, gripper_val) for a *single* goal step."""
        cfg = self.config
        if not cfg.enabled:
            return None
        if self.goal_xyz is None:
            return None
        if self._pending_joint_target is None or self._armed_time is None:
            return None
        if self._stepped:
            return None

        now = time.time() if now is None else float(now)
        if (now - self._armed_time) > cfg.reach_timeout:
            self.reset()
            return None

        current_joints = np.asarray(current_joints, dtype=float)
        joint_err = float(np.max(np.abs(current_joints - self._pending_joint_target)))

        if joint_err > cfg.reach_tolerance:
            self._reached_since = None
            return None

        if self._reached_since is None:
            self._reached_since = now
            return None

        if (now - self._reached_since) < cfg.reach_hold_time:
            return None

        current_pos = np.asarray(current_pos, dtype=float).reshape(3)
        current_quat = np.asarray(current_quat, dtype=float).reshape(4)
        delta = self.goal_xyz - current_pos
        dist = float(np.linalg.norm(delta))

        # Same behavior as original: don't assist very near the goal.
        if dist <= cfg.assist_disable_distance or dist <= cfg.stop_distance:
            self.reset()
            return None

        direction = delta / (dist + 1e-9)
        step = min(cfg.step_size, dist)
        target_pos = current_pos + direction * step

        self._stepped = True
        return target_pos, current_quat, float(self._pending_gripper)
