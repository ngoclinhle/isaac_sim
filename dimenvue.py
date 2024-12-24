import json
from typing import Optional, Sequence
import copy
from pxr import Gf
from omni.isaac.sensor import IMUSensor
from omni.isaac.core.utils.prims import create_prim, is_prim_path_valid
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core import SimulationContext
from camera import CameraAsync
from rotations import euler_angles_to_quat
from lidar import RtxLidar
from imu import IMU


class Dimenvue(RigidPrim):
    """
    Create a Dimenvue object with the given configuration file.
    """

    def __init__(self,
                 config_filfe_name: str,
                 **kwargs):
        """
        Create an xform prim for chassis, then lidar, imu, and cameras as children.
        Finally wrap it up in a RigidPrim. Then convert it to kinematic body, so that it won't be affected by physics.
        Args:
            config_filfe_name: The path to the configuration file.
            prim_path: The path to the xform prim.
            name: The name of the xform prim.
            position: world frame position
            orientation: local frame rotation in quaternion
            translation: local frame translation (mutually exclusive with position)
            scale: local frame scale
            visible: visibility of the prim
            mass: mass of the prim
            density: density of the prim
            linear_velocity: initial linear velocity
            angular_velocity: initial angular velocity
        Exceptions:
            ValueError: If the camera FPS does not match rendering FPS.
            ValueError: If the IMU rate does not match physics rate.
            ValueError: If the rendering rate is not a multiple of Lidar rotating rate.
            ValueError: If the physics rate is not a multiple of rendering rate.
        """
        with open(config_filfe_name, 'r', encoding='utf-8') as file:
            self._config = json.load(file)
        self._prim_path = self._next_available_prim_path(kwargs["prim_path"])
        kwargs["prim_path"] = self._prim_path
        if "translation" not in kwargs and "position" not in kwargs:
            kwargs["translation"] = self._config['chassis']['world_chassis_translate']
        if "orientation" not in kwargs:
            rotation = euler_angles_to_quat(
                self._config['chassis']['world_chassis_rotate'], degrees=True)
            kwargs["orientation"] = rotation
        create_prim(self._prim_path, prim_type="Xform")
        RigidPrim.__init__(self, **kwargs)
        self._lidar = None
        self._imu = None
        self._cameras = []
        self._build()
        self._check_dt()
        self._reset_current_frame()
        self._setup_callbacks()
        self._defy_physics()

    @property
    def config(self):
        return self._config

    @property
    def lidar(self):
        return self._lidar

    @property
    def imu(self):
        return self._imu

    @property
    def cameras(self):
        return self._cameras

    def consume_frame(self):
        """
        return all current data and reset the current frame.
        """
        current_frame = self._current_frame
        self._reset_current_frame()
        return current_frame

    def _reset_current_frame(self):
        self._current_frame = {
            "chassis": {},
            "lidar": {},
            "imu": {},
            "cameras": {cam.name: {} for cam in self._cameras}
        }

    def _next_available_prim_path(self, prim_path: str):
        index = 0
        prim_path_ = prim_path
        while is_prim_path_valid(prim_path_):
            prim_path_ = f"{prim_path}_{index}"
            index += 1
        return prim_path_

    def _build(self):
        if self._config['lidar']['enable']:
            self._build_lidar()
        if self._config['imu']['enable']:
            self._build_imu()
        if self._config['cameras']['enable']:
            self._build_cameras()

    def _build_lidar(self):
        lidar_config = self._config['lidar']
        lidar_name = 'lidar0'
        lidar_prim_path = f'{self._prim_path}/{lidar_name}'
        t = lidar_config['chassis_lidar_translate']
        r = lidar_config['chassis_lidar_rotate']
        self._lidar = RtxLidar(
            prim_path=lidar_prim_path,
            name=lidar_name,
            translation=t,
            orientation=euler_angles_to_quat(r, degrees=True),
            config_file_name=lidar_config['lidar_config_file']
        )
        self._lidar.initialize()

    def _build_imu(self):
        imu_config = self._config['imu']
        imu_name = 'imu0'
        imu_prim_path = f'{self._prim_path}/{imu_name}'
        t = imu_config['chassis_imu_translate']
        r = imu_config['chassis_imu_rotate']
        self._imu = IMU(
            prim_path=imu_prim_path,
            name=imu_name,
            translation=t,
            orientation=euler_angles_to_quat(r, degrees=True),
        )
        self._imu.initialize()

    def _build_cameras(self):
        cameras_config = self._config['cameras']
        camera_rig_prim_path = f'{self._prim_path}/cameras'
        t = cameras_config['chassis_cameras_translate']
        r = cameras_config['chassis_cameras_rotate']
        create_prim(
            prim_path=camera_rig_prim_path,
            translation=t,
            orientation=euler_angles_to_quat(r, degrees=True),
        )
        resolution = cameras_config['calibrate_config']['camera_resolution']
        for i in range(cameras_config['num_cameras']):
            camera_name = f'camera{i}'
            camera_prim_path = f'{camera_rig_prim_path}/{camera_name}'
            cam = CameraAsync(
                prim_path=camera_prim_path,
                name=camera_name,
                frequency=-1,  # update at rendering rate
                resolution=resolution
            )
            cam.initialize()
            t = cameras_config['camera_translate'][i]
            r = cameras_config['camera_rotate'][i]
            cam.set_local_pose(
                translation=t,
                orientation=euler_angles_to_quat(r, degrees=True),
                camera_axes=cameras_config['camera_axes']
            )
            cam.calibrate(cameras_config['calibrate_config'])
            self._cameras.append(cam)

    def _setup_callbacks(self):
        sim_ctx = SimulationContext.instance()
        sim_ctx.add_render_callback("chassis_cb", self._chassis_callback)
        if self._lidar:
            self._lidar.set_callback(self._lidar_callback)
        if self._imu:
            self._imu.set_callback(self._imu_callback)
        for cam in self._cameras:
            cam.set_callback(self._camera_callback)

    def _chassis_callback(self, event):
        pos, orient = self.get_local_pose()
        t = SimulationContext.instance().current_time
        chassis_data = {'time': t, 'gt_position': pos,
                        'gt_orientation': orient}
        self._add_dict(self._current_frame['chassis'], chassis_data)

    def _lidar_callback(self, lidar_data):
        self._add_dict(self._current_frame['lidar'], lidar_data)

    def _imu_callback(self, frame):
        self._add_dict(self._current_frame['imu'], frame)

    def _camera_callback(self, camera_data, camera_name):
        self._add_dict(
            self._current_frame['cameras'][camera_name], camera_data)

    def _add_dict(self, current_frame: dict, new_frame: dict):
        for key in new_frame.keys():
            if key not in current_frame:
                current_frame[key] = []
            current_frame[key].append(new_frame[key])

    def _check_dt(self):
        render_fps = 1.0/SimulationContext.instance().get_rendering_dt()
        physx_fps = 1.0/SimulationContext.instance().get_physics_dt()
        if abs(round(physx_fps/render_fps) - physx_fps/render_fps) > 1e-6:
            raise ValueError(
                "Physics rate is not a multiple of rendering rate")
        if self._config['lidar']['enable']:
            lidar_fps = self._lidar.config['profile']['scanRateBaseHz']
            if abs(render_fps - lidar_fps) > 1e-6:
                raise ValueError(
                    "Rendering rate does not match Lidar rotating rate")
        if self._config['imu']['enable']:
            imu_fps = self._config['imu']['imu_rate']
            if abs(imu_fps - physx_fps) > 1e-6:
                raise ValueError("IMU rate does not match physics rate")
        if self._config['cameras']['enable']:
            camera_fps = self._config['cameras']['camera_fps']
            if abs(render_fps - camera_fps) > 1e-6:
                raise ValueError("Camera rate does not match rendering rate")

    def _defy_physics(self):
        """
        make the prim a kinematic body, meaning it won't be affected by any forces.
        Let user control the motion of the prim.
        """
        self.prim.GetAttribute('physics:kinematicEnabled').Set(True)

    def set_local_pose(self,
                       translation: Optional[Sequence[float]] = None,
                       orientation: Optional[Sequence[float]] = None):
        """
        Set the local pose of the prim. Because XformPrim set_local_pose does not update IMU velocity, so we have to set USD attribute directly.
        Args:
            translation: local frame translation
            orientation: local frame rotation in quaternion
        """
        if translation is not None:
            self.prim.GetAttribute('xformOp:translate').Set(
                Gf.Vec3d(*translation))
        if orientation is not None:
            self.prim.GetAttribute('xformOp:orient').Set(
                Gf.Quatd(*orientation))

    def get_local_pose(self):
        t = self.prim.GetAttribute('xformOp:translate').Get()
        q = self.prim.GetAttribute('xformOp:orient').Get()
        translate = [t[0], t[1], t[2]]
        rotate = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()
                  [1], q.GetImaginary()[2]]
        return translate, rotate
