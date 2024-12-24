
from isaacsim import SimulationApp

simulation_app = SimulationApp()

import psutil
import carb
import omni
import omni.graph.core as og
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.utils.stage import *
from omni.isaac.core.utils.rotations import *
from omni.isaac.sensor import LidarRtx

def _og_connect(no, ao, ni, ai):
    og.Controller.connect(f'{no}.outputs:{ao}', f'{ni}.inputs:{ai}')


def get_mem():
    proc = psutil.Process()
    meminfo = proc.memory_info()
    return meminfo.rss / 1024 / 1024


def setup_scene(with_event=False):
    create_new_stage()
    world = World(physics_dt=0, rendering_dt=1.0/30)
    world.scene.add_default_ground_plane()
    lidar = LidarRtx(
        prim_path='/lidar',
        name='lidar',
        config_file_name='OS1_REV6_32ch10hz512res_noiseless',
        translation=[0, 0, 10],
        orientation=euler_angles_to_quat([0, 90, 0], degrees=True),
    )
    lidar.add_point_cloud_data_to_frame()
    lidar.add_azimuth_data_to_frame()
    lidar.add_elevation_data_to_frame()
    lidar.add_intensities_data_to_frame()
    lidar.add_range_data_to_frame()

    world.scene.add(lidar)
    world.reset()

    if with_event:
        lidar_node = lidar._point_cloud_annotator.get_node().get_prim_path()
        graph_path = '/'.join(lidar_node.split('/')[:-1])
        event_node = f'{graph_path}/event_node'
        event_name = 'rtxLidarReady'
        og.Controller.create_node(
            event_node, "omni.graph.action.SendMessageBusEvent")
        og.Controller.attribute("inputs:eventName", event_node).set(event_name)
        _og_connect(lidar_node, "exec", event_node, "execIn")
        attrs = [("azimuth", "float[]"),
                 ("elevation", "float[]"),
                 ("range", "float[]"),
                 ("transform", "matrixd[4]")]
        for aname, atype in attrs:
            og.Controller.create_attribute(event_node, aname, atype)
            _og_connect(lidar_node, aname, event_node, aname)

    return world, lidar


print("Testing without message bus event node")
mem1 = get_mem()
world, lidar = setup_scene(with_event=False)
for _ in range(1000):
    world.step()
mem2 = get_mem()
print(f"Memory usage increased: {mem2 - mem1:.2f} MB")

print("Testing with message bus event node")
mem1 = get_mem()
world, lidar = setup_scene(with_event=True)
for _ in range(1000):
    world.step()
mem2 = get_mem()
print(f"Memory usage increased: {mem2 - mem1:.2f} MB")

world.stop()
simulation_app.close()
