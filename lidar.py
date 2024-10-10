import os
import json
from typing import Optional
import glob

import carb
import omni.graph.core as og
from omni.isaac.sensor import BaseSensor
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import *
from omni.isaac.IsaacSensorSchema import IsaacRtxLidarSensorAPI
import omni.replicator.core as rep

import numpy as np


class RtxLidar(BaseSensor):
    """
    Replacement for omni.isaac.sensor.LidarRtx class
    Improvements:
    - add event listener to get point cloud data at correct time
    - add option to buffer frames for a full lidar rotation
    - add option to remove motion distortion from the point cloud
    - add option to get the point cloud in world frame
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "lidar_rtx",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        config_file_name: Optional[str] = None,
        transform_to_world: bool = False,
        no_distortion: bool = False
    ) -> None:
        self._current_frames = {}
        self._config = self._read_config(config_file_name)
        self.rotation_hz = self._config['profile']['scanRateBaseHz']
        self._buffering = False
        self._buffer_cb = None
        self._transform_to_world = transform_to_world
        self._no_distortion = no_distortion
        self._event_sub = None
        self._message_bus = None

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
        # self._create_event_sender()
        return

    def __del__(self):
        if self._event_sub is not None:
            self._event_sub.unsubscribe()

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

    def _og_connect(self, n1, a1, n2, a2):
        og.Controller.connect(f'{n1}.outputs:{a1}', f'{n2}.inputs:{a2}')

    def _create_point_cloud_graph(self):
        # create base graph by initializing annotator
        self._render_product = rep.create.render_product(
            self._sensor.GetPath(), resolution=(1, 1))
        self._annotator = rep.AnnotatorRegistry.get_annotator(
            # "RtxSensorCpu" + "IsaacComputeRTXLidarPointCloud"
            "RtxSensorCpu" + "IsaacReadRTXLidarData"
        )
        self._annotator.attach([self._render_product])
        lidar_node = self._annotator.get_node().get_prim_path()
        og.Controller.attribute(
            "inputs:keepOnlyPositiveDistance", lidar_node).set(False)

    def _create_event_sender(self):
        lidar_node = self._annotator.get_node().get_prim_path()
        # add event node
        graph_path = '/'.join(lidar_node.split('/')[:-1])
        event_node = f'{graph_path}/event_node'
        event_name = 'rtxLidarReady'
        og.Controller.create_node(
            event_node, "omni.graph.action.SendMessageBusEvent")
        og.Controller.attribute("inputs:eventName", event_node).set(event_name)
        self._og_connect(lidar_node, "exec", event_node, "execIn")
        attrs = [("azimuths", "float[]"),
                 ("elevations", "float[]"),
                 ("distances", "float[]"),
                 ("transform", "matrixd[4]"),
                 ("transformStart", "matrixd[4]"),
                 ("frameId", "uint64")]
        for aname, atype in attrs:
            og.Controller.create_attribute(event_node, aname, atype)
            self._og_connect(lidar_node, aname, event_node, aname)

        self._time_annotator = rep.AnnotatorRegistry.get_annotator(
            "IsaacReadTimes")
        self._time_annotator.attach([self._render_product])
        og.Controller.create_attribute(
            event_node, "rendering_time", "double")
        time_node = self._time_annotator.get_node().get_prim_path()
        self._og_connect(time_node, "simulationTime",
                         event_node, "rendering_time")

        # set event listener
        # event_type = carb.events.type_from_string(event_name)
        # self._message_bus = omni.kit.app.get_app().get_message_bus_event_stream()
        # self._event_sub = self._message_bus.create_subscription_to_pop_by_type(
        #     event_type, self._on_lidar_event)

    def _on_lidar_event(self, e):
        frame = e.payload.get_dict()
        print('_on_lidar_event: ', self._message_bus.event_count)
        if _is_empty_frame(frame):
            return

        frame['azimuths'] = np.array(frame['azimuths'])
        frame['elevations'] = np.array(frame['elevations'])
        frame['distances'] = np.array(frame['distances'])
        frame['transform'] = np.array(frame['transform']).reshape(4, 4).T
        frame['transformStart'] = np.array(
            frame['transformStart']).reshape(4, 4).T
        frame['xyz'] = _calculate_xyz(frame, degrees=True)

        if self._no_distortion:
            transforms = np.array(
                [frame['transformStart'], frame['transform']])
            frame['xyz'] = undistort(frame['xyz'], transforms)
            if not self._transform_to_world:
                frame['xyz'] = _to_sensor(
                    frame['xyz'], frame['transform'])

        if not self._no_distortion and self._transform_to_world:
            frame['xyz'] = _to_world(frame['xyz'], frame['transform'])

        for key in frame.keys():
            if key not in self._current_frames:
                self._current_frames[key] = []
            self._current_frames[key].append(frame[key])

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
        if len(self._current_frames) >= steps_per_frame:
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

    def set_transform_to_world(self, transform_to_world):
        """
        get the point cloud in world frame
        """
        self._transform_to_world = transform_to_world

    def set_no_distortion(self, no_distortion):
        """
        remove motion distortion from the point cloud
        """
        self._no_distortion = no_distortion


def _is_empty_frame(frame):
    return len(frame['azimuths']) == 0


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
        transforms: transform at the beginning and end of the frame
    Return:
        undistorted point cloud in world frame.
    """
    if np.allclose(transforms[0], transforms[1]):
        return _to_world(pcd, transforms[0])

    positions = np.array([t[:3, 3] for t in transforms])
    rotations = np.array([t[:3, :3] for t in transforms])
    pcd1 = pcd[:len(pcd)//2]
    pcd2 = pcd[len(pcd)//2:]
    timestamps = np.array([0, 1])
    times = np.linspace(timestamps[0], timestamps[-1], len(pcd1))

    pos_interpolated = _linear_interpolate(
        timestamps, positions, times)
    rot_interpolated = _linear_interpolate(
        timestamps, rotations, times)
    pcd1 = np.expand_dims(pcd1, axis=-1)
    pcd1_undistorted = np.squeeze(
        np.matmul(rot_interpolated, pcd1)) + pos_interpolated
    pcd2_undistorted = _to_world(pcd2, transforms[-1])
    return np.concatenate([pcd1_undistorted, pcd2_undistorted])
