from typing import Sequence

import numpy as np
from omni.isaac.sensor import BaseSensor
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core import SimulationContext
from rotations import quat_to_rot_matrix, rot_matrix_to_quat


class IMU(BaseSensor):
    """
    IMU class that runs at same rate as physics
    Calculates accel/gyro purely from prim pose 
    For simplicity, it has the same pose with its parent prim
    For now, use zero initial velocity. TODO: find initial velocity from parents..
    """

    def __init__(self,
                 prim_path: str,
                 name: str,
                 translation: Sequence[float] | None = None,
                 orientation: Sequence[float] | None = None,
                 ):
        BaseSensor.__init__(self,
                            prim_path=prim_path,
                            name=name,
                            translation=translation,
                            orientation=orientation
                            )
        self._parent_prim_path = '/'.join(prim_path.split('/')[:-1])
        self._parent_prim = get_prim_at_path(self._parent_prim_path)
        self._callback = None
        self._last_position = np.zeros(3)
        self._last_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_velocities = np.zeros(3)
        self._current_frame = {
            'time': 0.0,
            'physics_step': 0,
            'accel': np.zeros(3),
            'gyro': np.zeros(3),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0]),
        }
        self._gravity = np.zeros(3, dtype=np.float32)
        self._sim_ctx = None

    def initialize(self, physics_sim_view=None) -> None:
        BaseSensor.initialize(self, physics_sim_view)
        self._sim_ctx = SimulationContext.instance()
        self._sim_ctx.add_physics_callback(
            f"{self.prim_path}_physics_update", self.physics_update)
        self._last_position, self._last_orientation = self.get_world_pose()
        self._gravity[2] = self._sim_ctx.get_physics_context().get_gravity()[1]

    def set_callback(self, callback):
        self._callback = callback

    def physics_update(self, step_size):
        position, orientation = self.get_world_pose()

        # forward pass
        velocities = (position - self._last_position) / step_size
        accel = (velocities - self._last_velocities) / step_size
        accel += self._gravity
        rot = quat_to_rot_matrix(orientation)
        rot_last = quat_to_rot_matrix(self._last_orientation)
        accel_sensor = rot_last.T @ accel

        rot_diff = rot_last.T @ rot
        angular_velocities = _rotmat_to_rotvec(rot_diff) / step_size

        # backward pass, to compensate for stacking numerical error
        accel = rot_last @ accel_sensor
        accel -= self._gravity
        velocities = self._last_velocities + accel * step_size
        rotdiff = _rotvec_to_rotmat(angular_velocities * step_size)
        rot = rot_last @ rotdiff
        self._last_orientation = rot_matrix_to_quat(rot)
        self._last_position += velocities * step_size
        self._last_velocities = velocities

        # save result
        time = self._sim_ctx.current_time
        step = self._sim_ctx.current_time_step_index
        self._current_frame['accel'] = accel_sensor
        self._current_frame['gyro'] = angular_velocities
        self._current_frame['orientation'] = orientation
        self._current_frame['time'] = time
        self._current_frame['physics_step'] = step

        if self._callback is not None:
            self._callback(self._current_frame)

    def get_current_frame(self):
        return self._current_frame


def _rotmat_to_rotvec(r):
    costheta = np.clip((np.trace(r) - 1) / 2, -1, 1)
    theta = np.arccos(costheta)
    if theta < 1e-6:
        return np.zeros(3)
    return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) * theta / (2 * np.sin(theta))


def _rotvec_to_rotmat(v):
    theta = np.linalg.norm(v)  # Magnitude of the rotation vector
    if theta < 1e-6:  # Handle the case of small angles
        return np.eye(3)  # Return identity matrix for zero rotation

    k = v / theta  # Normalize the rotation vector
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])  # Skew-symmetric matrix

    # Rodrigues' rotation formula
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R
