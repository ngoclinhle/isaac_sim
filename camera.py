from typing import Any
import math

import omni
import carb
from omni.isaac.sensor import Camera
from omni.isaac.core import SimulationContext
import omni.replicator.core as rep


class CameraAsync(Camera):
    """
    A wrapper to allow user to add their own callback to save frame
    """
    _instance_cnt = 0

    def __init__(self, *args: Any, **kwds: Any) -> Any:
        if "name" not in kwds:
            kwds["name"] = f"camera{CameraAsync._instance_cnt}"
            carb.log_warn(
                f"CameraAsync: name not provided, using default: {kwds['name']}")
            CameraAsync._instance_cnt += 1

        super().__init__(*args, **kwds)

        self._cb = None

    def set_callback(self, callback):
        self._cb = callback

    # def resume(self):
    #     # omni.usd.get_context().get_rendering_event_stream().pump()
    #     # super().resume()

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        self._read_time_annotator = rep.AnnotatorRegistry.get_annotator(
            "IsaacReadTimes")
        self._read_time_annotator.attach([self._render_product_path])

    def _data_acquisition_callback(self, event):

        parsed_payload = self._sdg_interface.parse_rendered_simulation_event(
            event.payload["product_path_handle"], event.payload["results"]
        )
        # return if the event is from other camera
        if parsed_payload[0] != self._render_product_path:
            return

        super()._data_acquisition_callback(event)

        ref_time = self._read_time_annotator.get_data()
        self._current_frame['rendering_time'] = ref_time['simulationTime']

        if self._cb:
            self._cb(self._current_frame, self.name)

    def calibrate(self, config: dict):
        # Calculate the focal length and aperture size from the camera matrix
        if config['use_intrinsics']:
            ((fx, _, _), (_, fy, _), (_, _, _)) = config['camera_intrinsics']
            width, _ = config['camera_resolution']
            pixel_size = config['pixel_size_mm']
            horizontal_aperture_mm = pixel_size * width
            focal_length_x = fx * pixel_size
            focal_length_y = fy * pixel_size
            # The focal length in mm
            focal_length_mm = (focal_length_x + focal_length_y) / 2
        else:
            focal_length_mm = config['focal_length_mm']
            h_fov = config['horizontal_fov_deg'] * math.pi / 180.0
            horizontal_aperture_mm = 2 * focal_length_mm * math.tan(h_fov / 2)

        # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
        # Convert from mm to cm (or 1/10th of a world unit)
        self.set_focal_length(focal_length_mm / 10.0)
        # Convert from mm to cm (or 1/10th of a world unit)
        self.set_horizontal_aperture(horizontal_aperture_mm / 10.0)
        if config['f_stop'] > 0.0:
            # The focus distance in meters
            self.set_focus_distance(config['focus_distance'])
            # Convert the f-stop to Isaac Sim units
            self.set_lens_aperture(config['f_stop'] * 100.0)
        self.set_clipping_range(0.05, 1.0e5)
