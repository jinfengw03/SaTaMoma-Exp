import numpy as np
import jax.numpy as jnp
from manipulator import Manipulator
from oscbf_configs import OSCBFVelocityConfig
from cbfpy import CBF

class CollisionsVelocityConfig(OSCBFVelocityConfig):
    def __init__(
        self,
        robot: Manipulator,
        collision_positions: np.ndarray = np.array([]),
        collision_radii: np.ndarray = np.array([]),
    ):
        self.collision_positions = jnp.atleast_2d(collision_positions)
        self.collision_radii = jnp.ravel(collision_radii)
        super(CollisionsVelocityConfig, self).__init__(robot)

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        h_collision = jnp.array([1.0])
        if self.collision_positions.size > 0:
            center_deltas = (
                robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
            ).reshape(-1, 3)
            radii_sums = (
                robot_collision_radii[:, None] + self.collision_radii[None, :]
            ).reshape(-1)
            h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums
        return h_collision

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2

class JointSafetyFilter:
    def __init__(self, urdf_path, side='right'):
        self.side = side
        self.dt = 0.1 # Default dt
        
        # Collision data for TIAGO arm (extracted from colleague's code)
        link_1_pos = ((0.0, 0.0, 0.0),)
        link_1_radii = (0.08,)
        link_2_pos = ((0.0, 0.0, 0.0),)
        link_2_radii = (0.07,)
        link_3_pos = ((0.0, 0.0, 0.0), (0.0, 0.0, -0.08), (0.0, 0.0, -0.16))
        link_3_radii = (0.07, 0.07, 0.07)
        link_4_pos = ((0.0, 0.01, 0.02), (-0.08, 0.02, 0.01))
        link_4_radii = (0.08, 0.07)
        link_5_pos = ((0.0, 0.0, 0.02), (0.0, 0.0, 0.08))
        link_5_radii = (0.07, 0.07)
        link_6_pos = ((0.09, 0.0, 0.0), (0.15, 0.0, 0.0))
        link_6_radii = (0.07, 0.07)
        link_7_pos = ((0.0, 0.0, 0.0),)
        link_7_radii = (0.07,)

        positions_list = (link_1_pos, link_2_pos, link_3_pos, link_4_pos, link_5_pos, link_6_pos, link_7_pos)
        radii_list = (link_1_radii, link_2_radii, link_3_radii, link_4_radii, link_5_radii, link_6_radii, link_7_radii)
        collision_data = {"positions": positions_list, "radii": radii_list}

        # Initialize Manipulator
        self.robot = Manipulator.from_urdf(
            urdf_filename=urdf_path,
            ee_offset=np.eye(4),
            collision_data=collision_data
        )
        
        self.cbf = None
        self.last_velocity = np.zeros(7)
        self.max_acceleration = 2.0

    def update_obstacles(self, obstacles):
        """
        Update the obstacles for the CBF.
        :param obstacles: List of [x, y, z, r] in torso_lift_link frame
        """
        obs_array = np.array(obstacles) if obstacles else np.array([])
        self.cbf = CBF.from_config(CollisionsVelocityConfig(
            robot=self.robot,
            collision_positions=obs_array[:, :3] if obs_array.size else np.array([]),
            collision_radii=obs_array[:, 3] if obs_array.size else np.array([])
        ))

    def filter(self, q_curr, q_target, dt=0.1):
        """
        Filters the target joint position using CBF.
        :param q_curr: Current joint positions
        :param q_target: Target joint positions
        :param dt: Time step
        :return: q_safe
        """
        if self.cbf is None:
            return q_target
            
        u_nom = (q_target - q_curr) / dt
        u_safe = self.cbf.safety_filter(q_curr, u_nom)
        
        # Acceleration limiting
        accel = (u_safe - self.last_velocity) / dt
        accel_limited = np.clip(accel, -self.max_acceleration, self.max_acceleration)
        u_safe_limited = self.last_velocity + accel_limited * dt
        self.last_velocity = u_safe_limited.copy()
        
        q_safe = q_curr + u_safe_limited * dt
        return q_safe
