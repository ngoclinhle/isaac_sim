import json
import sys
import os

import numpy as np
import pandas as pd
import rerun as rr

from matplotlib import pyplot as plt

from rotations import *

rr.init('office_trajectory', spawn=True)


def Axes(length=1.0):
    return rr.Arrows3D(
        vectors=[[length, 0, 0], [0, length, 0], [0, 0, length]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    )


rr.log('/world', Axes(0.1), static=True)


def rr_transform(pos: np.ndarray, angles: np.ndarray):
    wxyz = euler_angles_to_quat(angles, degrees=True, extrinsic=True)
    xyzw = np.roll(wxyz, -1)
    return rr.Transform3D(
        translation=pos,
        quaternion=xyzw
    )


def rr_set_up_frame():
    # world, dimenvue = setup_sim()
    # dimenvue_config = dimenvue.config
    dimenvue_config = json.load(open('dimenvue_config.json'))
    rr.set_time_seconds('sim_time', 0.0)
    rr.log('/world', rr.Clear(recursive=True))
    rr.log('/world', Axes(1), static=True)
    c = dimenvue_config['chassis']
    p, r = c['world_chassis_translate'], c['world_chassis_rotate']
    rr.log('/world/chassis', rr_transform(p, r))
    rr.log('/world/chassis', Axes(1))
    if 'lidar' in dimenvue_config:
        c = dimenvue_config['lidar']
        p, r = c['chassis_lidar_translate'], c['chassis_lidar_rotate']
        rr.log('/world/chassis/lidar', rr_transform(p, r))
        rr.log('/world/chassis/lidar', Axes(1))
    if 'cameras' in dimenvue_config:
        c = dimenvue_config['cameras']
        p, r = c['chassis_cameras_translate'], c['chassis_cameras_rotate']
        rr.log('/world/chassis/cameras', rr_transform(p, r))
        for i in range(c['num_cameras']):
            p, r = c['camera_translate'][i], c['camera_rotate'][i]
            rr.log(f'/world/chassis/cameras/camera{i}', rr_transform(p, r))
            rr.log(f'/world/chassis/cameras/camera{i}', Axes(1))

            calib = c['calibrate_config']
            w, h = calib['camera_resolution']
            if calib['use_intrinsics']:
                ((fx, _, cx), (_, fy, cy), (_, _, _)
                 ) = calib['camera_intrinsics']
                f = (fx+fy)/2
            else:
                h_fov = calib['horizontal_fov_deg']*np.pi/180
                f = w/(2*np.tan(h_fov/2))
                cx, cy = w/2, h/2

            rr.log(f'/world/chassis/cameras/camera{i}/image', rr.Pinhole(
                resolution=(w, h),
                camera_xyz=rr.ViewCoordinates.RDF,  # same as ROS
                focal_length=(f, f),
                principal_point=(cx, cy),
                image_plane_distance=1.0
            ))


rr_set_up_frame()
