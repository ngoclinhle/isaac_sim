"""
Post process the trajectory:
- transform camera pose to chassis pose
- interpolate to generate more data
- split to multiple segments
"""
import os
import shutil
import pandas as pd
import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R
from scipy.interpolate import interp1d
import rerun as rr
import argparse


enable_debug = False
enable_transform = False
target_hz = [10, 30, 400]
datasets = ['office']
split_interval = 0
overlap = 0


def transform_cam_to_chassis(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms camera coordinates to chassis coordinates.
    Args:
        data_frame (pd.DataFrame): The input DataFrame containing camera data.
    Returns:
        pd.DataFrame: The transformed DataFrame containing chassis data.
    """
    quaternions = R.from_quat(
        data_frame[['qx', 'qy', 'qz', 'qw']])
    # camera form is usd convention +y up -z forward
    # rotate back to world frame so that +z up +x forward
    q_usd_world = R.from_euler('XYZ', [-90, 0, 90], True)  # intrinsic rotate

    # q_usd_world is applied on local axes, so reverse multiply order
    q_chassis = quaternions * q_usd_world

    data_copy = data_frame.copy()

    data_copy['qx'] = q_chassis.as_quat()[:, 0]
    data_copy['qy'] = q_chassis.as_quat()[:, 1]
    data_copy['qz'] = q_chassis.as_quat()[:, 2]
    data_copy['qw'] = q_chassis.as_quat()[:, 3]

    return data_copy


def interpolate(data_frame: pd.DataFrame, hz: int) -> pd.DataFrame:
    """
    interpolate position and orientation data to a target frequency.
    Args:
        data_frame (pd.DataFrame): The input DataFrame containing position and orientation data.
        hz (int): The target frequency.
    """
    original_timestamps = data_frame['timestamp'].values
    new_interval = 1 / hz
    new_timestamps = np.arange(
        original_timestamps.min(), original_timestamps.max(), new_interval)
    new_timestamps = new_timestamps[1:-1]

    positions = data_frame[['x', 'y', 'z']].to_numpy()
    f = interp1d(original_timestamps, positions, kind='cubic', axis=0)
    interpolated_positions = f(new_timestamps)

    quaternions = R.from_quat(data_frame[['qx', 'qy', 'qz', 'qw']])
    slerp = Slerp(original_timestamps, quaternions)
    interpolated_orientations = slerp(new_timestamps).as_quat()

    interpolated_data = pd.DataFrame({
        'timestamp': new_timestamps,
        'x': interpolated_positions[:, 0],
        'y': interpolated_positions[:, 1],
        'z': interpolated_positions[:, 2],
        'qw': interpolated_orientations[:, 3],
        'qx': interpolated_orientations[:, 0],
        'qy': interpolated_orientations[:, 1],
        'qz': interpolated_orientations[:, 2],
    })

    return interpolated_data


def split(data_frame: pd.DataFrame, split_interval: float, overlap: float):
    """
    Splits a given DataFrame into multiple smaller DataFrames
    Args:
        data_frame (pd.DataFrame): The DataFrame to be split.
        split_interval (float): The time interval at which to split the DataFrame.
        overlap (float): The percentage of overlap between consecutive splits.
    """
    duration = data_frame['timestamp'].iloc[-1]
    if split_interval == 0:
        yield data_frame
        return

    num_splits = int(duration / split_interval)
    if num_splits*split_interval < duration:
        num_splits += 1

    for i in range(num_splits):
        t_start = (i - overlap) * split_interval
        t_end = (i + 1 + overlap) * split_interval
        t_start = max(0, t_start)
        t_end = min(duration, t_end)
        split_data = data_frame[(data_frame['timestamp'] >= t_start) & (
            data_frame['timestamp'] < t_end)]
        split_data = split_data[['timestamp', 'x',
                                 'y', 'z', 'qw', 'qx', 'qy', 'qz']]
        yield split_data


def Axes(length=1.0):
    return rr.Arrows3D(
        vectors=[[length, 0, 0], [0, length, 0], [0, 0, length]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    )


def log_trajectory(data_frame: pd.DataFrame, name: str, with_axes=False, color=[255, 0, 0], downsample=0.1) -> None:
    """
    Log the trajectory data to rerun visualizer. Downsample=0.1 means only 10% of the data will be logged.
    """
    if not enable_debug:
        return
    data = data_frame
    points = data[['x', 'y', 'z']].to_numpy()
    rr.log(f'/world/{name}', rr.Points3D(points,  colors=color))
    quaternions = data[['qx', 'qy', 'qz', 'qw']].to_numpy()

    if with_axes:
        for j in range(int(len(data)*downsample)):
            i = int(j/downsample)
            rr.log(f'/world/{name}/{i}',
                   rr.Transform3D(translation=points[i], quaternion=quaternions[i]))
            rr.log(f'/world/{name}/{i}', Axes(1))


def process(dataset: str) -> None:
    """
    Process the dataset by performing the following steps:
    1. Read the low hz dataset from a CSV file.
    2. Transform the camera coordinates to chassis coordinates.
    3. Interpolate the data based on the target frequency.
    4. Split the data into smaller chunks.
    5. Save the splits as csv files
    Parameters:
    - dataset (str): The name of the dataset.
    Returns:
    - None
    """
    data = pd.read_csv(f'{dataset}/trajectory_{dataset}.csv')
    if all(isinstance(col, int) for col in data.columns):
        data.columns = ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
    if enable_transform:
        log_trajectory(data, 'OmniverseKit_Persp_trajectory', True)
        data = transform_cam_to_chassis(data)
    log_trajectory(data, 'chassis_trajectory', True)
    for hz in target_hz:
        interpolated_data = interpolate(data, hz)
        # if hz == 400:
        #     log_trajectory(interpolated_data,
        #                    'interpolated_trajectory', False, [0, 255, 0])
        splits_gen = split(interpolated_data, split_interval, overlap)
        splits_dir = f'{dataset}/{hz}'
        shutil.rmtree(splits_dir, ignore_errors=True)
        os.makedirs(splits_dir)
        for i, split_data in enumerate(splits_gen):
            # delete the folder and content inside
            split_data.to_csv(f'{splits_dir}/{i}.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_transform',
                        action='store_true', default=False)
    parser.add_argument('--datasets', nargs='+', default=['office'])
    parser.add_argument('--target_hz', nargs='+', default=[10, 400])
    parser.add_argument('--enable_debug', action='store_true', default=False)
    parser.add_argument('--split_interval', type=float, default=0)
    parser.add_argument('--overlap', type=float, default=0)
    args = parser.parse_args()
    enable_transform = args.enable_transform
    datasets = args.datasets
    target_hz = args.target_hz
    enable_debug = args.enable_debug
    split_interval = args.split_interval
    overlap = args.overlap

    if enable_debug:
        rr.init('post_process', spawn=True)

    for d in datasets:
        process(d)
