import os
import json
from typing import Optional
import glob
import copy

import carb
import omni.graph.core as og
from omni.isaac.sensor import BaseSensor
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import *
from omni.isaac.IsaacSensorSchema import IsaacRtxLidarSensorAPI
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as _syntheticdata
from omni.isaac.core_nodes.bindings import _omni_isaac_core_nodes

from rotations import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class RtxLidar(BaseSensor):
    """
    Replacement for omni.isaac.sensor.LidarRtx class
    Improvements:
    - connect acquisition callback to stage rendering event instead of update event
    - add buffering option to buffer frames and return them in a single callback
    TODO: it only works if simulation rate is the same as rotation rate.
    For example if lidar is 10Hz and sim render at 30fps the point cloud will have error
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "lidar_rtx",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        config_file_name: Optional[str] = None,
        max_queue_size: int = 1
    ):
        self._max_queue_size = max_queue_size
        self._config = self._read_config(config_file_name)
        self.rotation_hz = self._config['profile']['scanRateBaseHz']
        self._buffering = False
        self._buffer_cb = None

        self._render_product_path = None
        self._render_event_sub = None
        self._physics_sub = None
        self._og_controller = None
        self._sdg_graph_pipeline = None
        self._sdg_interface = None

        self._frame_keys = [
            "frameId",
            "azimuths",
            "elevations",
            "distances",
            "intensities",
            "hitPointNormals",
            "materialIds",
            "objectIds",
            "transform",
            "transformStart",
        ]

        self._current_frames = {}
        self._clear_current_frames()
        self._poses = []
        self._last_frame_time = 0.0

        self._create_rtx_lidar_sensor(prim_path, config_file_name)
        BaseSensor.__init__(
            self,
            prim_path=prim_path,
            name=name,
            translation=translation,
            position=position,
            orientation=orientation
        )
        self._create_point_cloud_graph()

    def __del__(self):
        if self._render_event_sub is not None:
            self._render_event_sub.unsubscribe()
            del self._render_event_sub
        if self._physics_sub is not None:
            self._physics_sub.unsubscribe()
            del self._physics_sub

    def _read_config(self, config_file_name):
        config_path = os.path.abspath(
            os.path.join(
                get_extension_path_from_name("omni.isaac.sensor"),
                "data/lidar_configs/"
            )
        )
        config_files = glob.glob(
            config_path+f'/**/{config_file_name}.json', recursive=True)
        if len(config_files) == 0:
            raise ValueError(f"Config file {config_file_name} not found")

        file_path = config_files[0]
        with open(file_path, 'rt', encoding='utf-8') as f:
            config = json.load(f)
        return config

    def _clear_current_frames(self):
        self._current_frames = {}
        keys = self._frame_keys + ['rendering_time', 'xyz']
        for k in keys:
            self._current_frames[k] = None
        self._current_frames['poses'] = []

    @property
    def config(self):
        return self._config

    def _create_rtx_lidar_sensor(self, prim_path, config_file_name):
        if is_prim_path_valid(prim_path):
            if get_prim_type_name(prim_path) != "Camera" or not get_prim_at_path(prim_path).HasAPI(
                IsaacRtxLidarSensorAPI
            ):
                raise ValueError(
                    "prim path does not correspond to a Isaac Lidar prim.")
            carb.log_warn(
                f"Using existing RTX Lidar prim at path {prim_path}")
        else:
            _, self._sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar", path=prim_path, parent=None, config=config_file_name
            )

    def _create_point_cloud_graph(self):
        # create base graph by initializing annotator
        self._render_product = rep.create.render_product(
            self._sensor.GetPath(), resolution=(1, 1))
        self._render_product_path = self._render_product.path
        self._annotator = rep.AnnotatorRegistry.get_annotator(
            # "RtxSensorCpu" + "IsaacComputeRTXLidarPointCloud"
            "RtxSensorCpu" + "IsaacReadRTXLidarData"
        )
        self._annotator.attach([self._render_product])
        lidar_node = self._annotator.get_node().get_prim_path()
        og.Controller.attribute(
            "inputs:keepOnlyPositiveDistance", lidar_node).set(False)
        self._time_annotator = rep.AnnotatorRegistry.get_annotator(
            "IsaacReadTimes")
        self._time_annotator.attach([self._render_product_path])

        self._render_event_sub = (
            omni.usd.get_context()
            .get_rendering_event_stream()
            .create_subscription_to_pop_by_type(
                int(omni.usd.StageRenderingEventType.NEW_FRAME),
                self._data_acquisition_callback,
                name="my.rtx.lidar.data.acquisition",
                order=1000,
            )
        )

        self._physx_interface = omni.physx.acquire_physx_interface()
        self._physics_sub = self._physx_interface.subscribe_physics_step_events(
            self._add_poses_cb)
        self._core_nodes_interface = _omni_isaac_core_nodes.acquire_interface()

        self._og_controller = og.Controller()
        self._sdg_graph_pipeline = "/Render/PostProcess/SDGPipeline"
        self._sdg_interface = _syntheticdata.acquire_syntheticdata_interface()

    def _data_acquisition_callback(self, event):
        parsed_payload = self._sdg_interface.parse_rendered_simulation_event(
            event.payload["product_path_handle"], event.payload["results"]
        )

        if parsed_payload[0] != self._render_product_path:
            return

        self._og_controller.evaluate_sync(
            graph_id=self._sdg_graph_pipeline)

        lidar_data = self._annotator.get_data()
        if len(lidar_data["azimuths"]) == 0:
            print("skip empty frame")
            return

        time_data = self._time_annotator.get_data()
        rendering_time = time_data['simulationTime']
        self._current_frames['rendering_time'] = rendering_time
        for key in self._frame_keys:
            if key in ['transform', 'transformStart']:
                transform = lidar_data[key].reshape(4, 4).T
                self._current_frames[key] = transform
            else:
                self._current_frames[key] = lidar_data[key]

        point_cloud = _calculate_xyz(lidar_data, degrees=True)
        self._current_frames['xyz'] = point_cloud
        frame_poses, remaining_poses = self._get_poses_list(
            self._poses, self._last_frame_time, rendering_time)
        self._current_frames['poses'] = frame_poses
        self._poses = remaining_poses

        pcd = redistort(self._current_frames)
        transform_start = pose_to_transform(frame_poses[0][1:])
        transform = pose_to_transform(frame_poses[-1][1:])
        self._current_frames['transformStart'] = transform_start
        self._current_frames['transform'] = transform
        self._current_frames['xyz'] = pcd

        if self._buffer_cb:
            result = self._current_frames
            self._clear_current_frames()
            self._buffer_cb(result)

    def _add_poses_cb(self, stepsize):
        position, orientation = self.get_world_pose()
        time = self._core_nodes_interface.get_sim_time()
        self._poses.append((time, position, orientation))

    def _get_poses_list(self, poses, last_frame_time, current_time):
        """
        the lidar NEW_FRAME are not sync with rendering loop so its not guaranteed that the recorded poses 
        are in the frame boundary. so we manually slice the poses list using the frame time
        """
        frame_poses = [p for p in poses if p[0] >=
                       last_frame_time and p[0] <= current_time]
        remaining = [p for p in poses if p[0] > current_time]
        return frame_poses, remaining

    def consume_current_frames(self):
        """
        return current frames immediately and clear the buffer
        """
        frames = self._current_frames
        self._current_frames = {}
        return frames

    def set_callback(self, buffer_cb=None):
        """
        Args:
            buffer_cb: callback function when lidar finishes a full revolution
        """
        self._buffer_cb = buffer_cb


def _calculate_xyz(frame, degrees=False, filtered=False):
    """remove the second echo and calculate point xyz"""
    azimuths = frame['azimuths']
    elevations = frame['elevations']
    distances = frame['distances']
    if not filtered:
        azimuths = azimuths.reshape(-1, 2)[:, 0]
        elevations = elevations.reshape(-1, 2)[:, 0]
        distances = distances.reshape(-1, 2)[:, 0]
    if degrees:
        azimuths = np.deg2rad(azimuths)
        elevations = np.deg2rad(elevations)
    x = distances * np.cos(elevations) * np.cos(azimuths)
    y = distances * np.cos(elevations) * np.sin(azimuths)
    z = distances * np.sin(elevations)
    points = np.array([x, y, z]).T
    return points


def _to_sensor(pcd, transform):
    return np.matmul(transform[:3, :3].T, (pcd - transform[:3, 3]).T).T


def _to_world(pcd, transform):
    return np.matmul(transform[:3, :3], pcd.T).T + transform[:3, 3]


def undistort(pcd, poses):
    """
    undistort point cloud in sensor frame to world frame
    Args:
        pcd: point cloud in sensor frame
        poses: list of poses [time, translation, quaternion(wxyz)]
    Return:
        undistorted point cloud in world frame
    """

    times = np.array([p[0] for p in poses])
    trans = np.array([p[1] for p in poses])
    quats = np.array([p[2] for p in poses])
    quats = wxyz2xyzw(quats)

    point_times = np.linspace(times[0], times[-1], len(pcd))
    f = interp1d(times, trans, axis=0)
    trans = f(point_times)

    slerp = Slerp(times, R.from_quat(quats))
    rotations = slerp(point_times).as_matrix()

    pcd_world = np.matmul(rotations, pcd[..., None]).squeeze() + trans
    return pcd_world


def distort(pcd_world, poses):
    """
    distort a pointcloud in world frame to sensor frame
    Args:
        pcd_world: point cloud in world frame
        poses: list of poses [time, translation, quaternion(wxyz)]
    Return:
        distorted point cloud in sensor frame
    """
    times = np.array([p[0] for p in poses])
    trans = np.array([p[1] for p in poses])
    quats = np.array([p[2] for p in poses])
    quats = wxyz2xyzw(quats)

    point_times = np.linspace(times[0], times[-1], len(pcd_world))
    f = interp1d(times, trans, axis=0)
    trans = f(point_times)

    f = Slerp(times, R.from_quat(quats))
    rotations_inv = f(point_times).inv().as_matrix()

    pcd_sensor = np.matmul(
        rotations_inv, (pcd_world - trans)[..., None]).squeeze()
    return pcd_sensor


def transform_to_pose(transform):
    position = transform[:3, 3]
    rotation = transform[:3, :3]
    quat = rot_matrix_to_quat(rotation)
    return (position, quat)


def pose_to_transform(pose):
    position, quat = pose
    rotation = quat_to_rot_matrix(quat)
    transform = np.eye(4)
    transform[:3, 3] = position
    transform[:3, :3] = rotation
    return transform


def redistort(isaac_frame):
    p0, q0 = transform_to_pose(isaac_frame['transformStart'])
    p1, q1 = transform_to_pose(isaac_frame['transform'])
    poses = [[0, p0, q0], [1, p1, q1]]
    # only undistort half of the points
    pcd = isaac_frame['xyz']
    pcd1 = pcd[:len(pcd)//2]
    pcd2 = pcd[len(pcd)//2:]
    pcd1_world = undistort(pcd1, poses)
    pcd2_world = _to_world(pcd2, isaac_frame['transform'])
    pcd_world = np.concatenate([pcd1_world, pcd2_world])
    print(len(isaac_frame['poses']))
    poses = isaac_frame['poses']
    pcd_sensor = distort(pcd_world, poses)
    return pcd_sensor
