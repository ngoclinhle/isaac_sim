import os
import json
from typing import Optional
from glob import glob

import carb

import omni.graph.core as og
import omni.replicator.core as rep
from omni.isaac.sensor import LidarRtx
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.rotations import quat_to_rot_matrix

import numpy as np


class RtxLidarFullFrame(LidarRtx):
    """
    subclass the LidarRtx class. Changes:
    - replace the ""RtxSensorCpuIsaacComputeRTXLidarPointCloud" annotator with
        "RtxSensorCpuIsaacCreateRTXLidarScanBuffer" annotator
    - modify frame keys accordingly
    - initialize the lidar inside __init__ function
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "lidar_rtx",
        transform_to_world: bool = True,
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        config_file_name: Optional[str] = None,
    ) -> None:

        super().__init__(prim_path, name, position,
                         translation, orientation, config_file_name)

        self._attribute_map = {
            "lidar_info": "info",
            "point_cloud_data": "data",
            "intensities_data": "intensity",
            "index": "index"
        }

        for key in self._attribute_map:
            self._current_frame[key] = []

        self.initialize()

        self.config_copy = {}
        configs_path = os.path.abspath(
            os.path.join(
                get_extension_path_from_name("omni.isaac.sensor"),
                "data/lidar_configs/"
            )
        )
        configs = glob(
            configs_path+f'/**/{config_file_name}.json', recursive=True)
        assert len(configs) > 0, f"Config file {config_file_name} not found"
        file_path = configs[0]
        with open(file_path, 'rt', encoding='utf-8') as f:
            self.config_copy = json.load(f)

        scan_rate_base_Hz = self.get_rotation_frequency()
        self._update_dt = 1.0/scan_rate_base_Hz
        self._next_ready = self._update_dt

        self._transform_to_world = transform_to_world

        self._frame_ready_callback = None

        return

    def _create_point_cloud_graph_node(self):
        self._point_cloud_annotator = rep.AnnotatorRegistry.get_annotator(
            "RtxSensorCpu" + "IsaacCreateRTXLidarScanBuffer"
        )
        # always get the world points. Transform it back to sensor frame if needed
        self._point_cloud_annotator.initialize(
            transformPoints=True
        )
        self._point_cloud_annotator.attach([self._render_product_path])
        self._point_cloud_node_path = self._point_cloud_annotator.get_node().get_prim_path()

    def _data_acquisition_callback(self, event: carb.events.IEvent):
        self._current_frame["rendering_frame"] = (
            og.Controller()
            .node("/Render/PostProcess/SDGPipeline/PostProcessDispatcher")
            .get_attribute("outputs:referenceTimeNumerator")
            .get()
        )

        rendering_time = self._core_nodes_interface.get_sim_time_at_swh_frame(
            self._current_frame["rendering_frame"]
        )

        self._current_frame["rendering_time"] = rendering_time

        point_cloud_data = self._point_cloud_annotator.get_data()

        for key in self._current_frame:
            attribute_name = "".join(
                [word[0].upper() + word[1:] for word in key.split("_")])
            attribute_name = attribute_name[0].lower() + attribute_name[1:]
            if key not in ["rendering_time", "rendering_frame"]:
                if key in self._attribute_map:
                    self._current_frame[key] = point_cloud_data[self._attribute_map[key]]

        if rendering_time >= self._next_ready and self._frame_ready_callback is not None:
            frame = self._current_frame
            if not self._transform_to_world:
                frame = self._transform_to_sensor_frame(self._current_frame)
            self._frame_ready_callback(frame)
            self._next_ready += self._update_dt

    # lidar_publish_rate = lidar.get_rotation_frequency() # doesn't work until sim started
    # so we add a function to read it from json file =.=
    def get_rotation_frequency(self) -> float:
        return self.config_copy['profile']['scanRateBaseHz']

    # the transform in lidar_info does not corresponds to the current pose of the lidar
    # so we move the world points to current frame
    def _transform_to_sensor_frame(self, frame: dict) -> dict:
        points_world = frame['point_cloud_data']
        if points_world.shape[0] == 0:
            return frame
        position, orientation = self.get_world_pose()
        tf = np.eye(4)
        tf[:3, :3] = quat_to_rot_matrix(orientation)
        tf[:3, 3] = position
        tf_inv = np.linalg.inv(tf)
        points_sensor = np.dot(
            tf_inv[:3, :3], points_world.T).T + tf_inv[:3, 3]
        frame['point_cloud_data'] = points_sensor
        frame['lidar_info']['transform'] = tf.T.reshape(-1)

        return frame

    def set_frame_ready_callback(self, callback: callable):
        self._frame_ready_callback = callback

    def set_transform_to_world(self, transform_to_world: bool):
        self._transform_to_world = transform_to_world
