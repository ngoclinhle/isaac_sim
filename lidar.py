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

import numpy as np
from rotations import *


class RtxLidar(BaseSensor):
    """
    Replacement for omni.isaac.sensor.LidarRtx class
    Improvements:
    - connect acquisition callback to stage rendering event instead of update event
    - add buffering option to buffer frames and return them in a single callback
    TODO: it only works if simulation rate is the same as rotation rate.
    For example if lidar is 10Hz and sim render at 30fps the point cloud will fail
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
        keys = self._frame_keys + ['rendering_time', 'xyz']
        for k in keys:
            self._current_frames[k] = []

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

        # self._print_writer = rep.WriterRegistry.get(
        #     "WriterIsaacPrintRTXSensorInfo")
        # self._print_writer.attach([self._render_product_path])

        # self._render_event_sub = (
        #     omni.usd.get_context()
        #     .get_rendering_event_stream()
        #     .create_subscription_to_pop_by_type(
        #         int(omni.usd.StageRenderingEventType.NEW_FRAME),
        #         self._data_acquisition_callback,
        #         name="my.rtx.lidar.data.acquisition",
        #         order=1000,
        #     )
        # )
        self._render_event_sub = (
            omni.usd.get_context()
            .get_rendering_event_stream()
            .create_subscription_to_pop(
                self._data_acquisition_callback
            )
        )
        self._events = []

        self._og_controller = og.Controller()
        self._sdg_graph_pipeline = "/Render/PostProcess/SDGPipeline"
        self._sdg_interface = _syntheticdata.acquire_syntheticdata_interface()

    def _data_acquisition_callback(self, event):
        self._events.append(event)
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
        self._current_frames['rendering_time'].append(rendering_time)
        for key in self._frame_keys:
            if key in ['transform', 'transformStart']:
                transform = lidar_data[key].reshape(4, 4).T
                self._current_frames[key].append(transform)
            else:
                self._current_frames[key].append(lidar_data[key])

        point_cloud = _calculate_xyz(lidar_data, degrees=True)
        self._current_frames['xyz'].append(point_cloud)

        if self._buffering:
            self._check_buffer()

    def _check_buffer(self):
        if self._buffer_cb is None:
            return

        dt = SimulationContext.instance().get_rendering_dt()
        steps_per_second = int(1/dt)
        frames_per_second = int(self.rotation_hz)
        if steps_per_second % frames_per_second != 0:
            raise ValueError("frame time must be a multiple of render step")
        steps_per_frame = steps_per_second // frames_per_second
        if len(self._current_frames['rendering_time']) >= steps_per_frame:
            result = {}
            for key, _ in self._current_frames.items():
                result[key] = self._current_frames[key][:steps_per_frame]
                self._current_frames[key] = self._current_frames[key][steps_per_frame:]
            self._buffer_cb(result)

    def consume_current_frames(self):
        """
        return current frames immediately and clear the buffer
        """
        frames = self._current_frames
        self._current_frames = {}
        return frames

    def set_buffering(self, buffering=False, buffer_cb=None):
        """
        Args:
            buffering: whether to buffer frames
            buffer_cb: callback function when lidar finishes a full revolution
        """
        self._buffering = buffering
        self._buffer_cb = buffer_cb


def _linear_interpolate(xp, yp, x):
    if yp.shape[0] != xp.shape[0]:
        raise ValueError("xp and yp must have the same length")
    yp_shape = yp.shape
    yp = yp.reshape(yp_shape[0], -1).T
    y = np.array([np.interp(x, xp, ypi) for ypi in yp]).T
    y = y.reshape(-1, *yp_shape[1:])
    return y


def _to_world(pcd, transform):
    return np.matmul(transform[:3, :3], pcd.T).T + transform[:3, 3]


def _to_sensor(pcd, transform):
    return np.matmul(transform[:3, :3].T, (pcd - transform[:3, 3]).T).T


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


def undistort(pcd, transforms):
    """
    The Isaac RTX Lidar sensor seems to model the distortion for half of the frame
    So we undistort the first half and use the last transformation for the second half
    If the transformation at the beginning and end of the frame are the same, we skip 
    the interpolation and simply move it to world frame
    Args:
        pcd: point cloud in sensor frame
        transforms: [transformStart, transformEnd] pose of lidar when capture first and last point.
    Return:
        undistorted point cloud in sensor frame (transformEnd)
    """
    if np.allclose(transforms[0], transforms[1]):
        return pcd

    positions = np.array([t[:3, 3] for t in transforms])
    rotations = np.array([t[:3, :3] for t in transforms])
    angles = np.array([matrix_to_euler_angles(r) for r in rotations])
    pi = np.pi
    angles_changes = angles[1] - angles[0]
    # assuming during one scan, the frame rotate in the shorter direction
    angles_changes[angles_changes > pi] -= 2*pi
    angles_changes[angles_changes < -pi] += 2*pi
    angles[1] = angles[0] + angles_changes
    pcd1 = pcd[:len(pcd)//2]
    pcd2 = pcd[len(pcd)//2:]
    timestamps = np.array([0, 1])
    times = np.linspace(timestamps[0], timestamps[-1], len(pcd1))

    pos_interpolated = _linear_interpolate(
        timestamps, positions, times)
    angles_interpolated = _linear_interpolate(
        timestamps, angles, times)
    rot_interpolated = np.array([euler_to_rot_matrix(a)
                                for a in angles_interpolated])
    # rot_interpolated = _linear_interpolate(
    #     timestamps, rotations, times)
    pcd1 = np.expand_dims(pcd1, axis=-1)
    pcd1_undistorted = np.squeeze(
        np.matmul(rot_interpolated, pcd1)) + pos_interpolated
    pcd1_undistorted = _to_sensor(pcd1_undistorted, transforms[-1])
    pcd2_undistorted = pcd2
    return np.concatenate([pcd1_undistorted, pcd2_undistorted])
