import numpy as np
import os
import sys
import pickle
import time
import threading


class DataLogger:
    """
    Basically convert list of dicts to dict of lists
    Then flush to disk when size exceeds chunk size to avoid memory overflow
    """

    def __init__(self, data_dir, chunk_size_mb=1000):
        self._chunk_size_mb = chunk_size_mb
        self._data_dir = data_dir
        if os.path.exists(self._data_dir):
            self._data_dir = f"{data_dir}_{time.strftime('%Y%m%d_%H%M%S')}"
            print(
                f"DataLogger: {data_dir} already exists, using {self._data_dir}")
        os.makedirs(self._data_dir)
        self._chunks_cnt = 0
        self._transform_func = None
        self._current_data = {}
        self._current_chunk_size = 0

    def __del__(self):
        self._flush()

    def _flush(self):
        file_name = f"{self._data_dir}/{self._chunks_cnt}.pkl"
        self._chunks_cnt += 1
        with open(file_name, 'wb') as f:
            pickle.dump(self._current_data, f)
        self._current_data = {}
        self._current_chunk_size = 0

    def set_transform(self, func):
        """
        set a transform function to be applied to each data frame
        """
        self._transform_func = func

    def log(self, data_frame):
        """
        accumulate and flush
        """
        if self._transform_func:
            transformed_frame = self._transform_func(data_frame)
        else:
            transformed_frame = data_frame
        _accumulate_dict(self._current_data, transformed_frame)
        self._current_chunk_size += _get_size(transformed_frame)
        if self._current_chunk_size > self._chunk_size_mb * 1024 * 1024:
            self._flush()


def _accumulate_dict(root, data_frame):
    for key, value in data_frame.items():
        if isinstance(value, dict):
            if key not in root:
                root[key] = {}
            _accumulate_dict(root[key], value)
        else:
            if key not in root:
                root[key] = []
            if isinstance(value, list):
                root[key].extend(value)
            else:
                root[key].append(value)


def _get_size(obj):
    if isinstance(obj, dict):
        return sum(_get_size(v) for v in obj.values()) + sys.getsizeof(obj)
    if isinstance(obj, list):
        return sum(_get_size(v) for v in obj) + sys.getsizeof(obj)
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    return sys.getsizeof(obj)


def test_accumulation():
    import json

    root = {}
    frame = {
        'key_obj_1': 1,
        'key_list_1': [1, 2, 3],
        'key_dict_1': {
            'key_obj_3': 1,
        }
    }

    _accumulate_dict(root, frame)
    print(json.dumps(root, indent=4))

    frame = {
        'key_obj_1': 2,
        'key_obj_2': 3,  # add new key
        'key_list_1': [4, 5],
        'key_dict_1': {
            'key_obj_3': 2,
            'key_dict_2': {
                'key_obj_4': 1,
                'key_list_1': [1, 2, 3]
            }
        }
    }
    _accumulate_dict(root, frame)
    print(json.dumps(root, indent=4))

    frame = {
        'key_dict_1': {
            'key_dict_2': {
                'key_list_1': [4]
            }
        }
    }
    _accumulate_dict(root, frame)
    print(json.dumps(root, indent=4))


def test_get_size():
    data = {
        'key_obj_1': 1,
        'key_list_1': [1, 2, 3],
        'key_dict_1': {
            'key_obj_3': 1,
        }
    }
    print(_get_size(data))
    data = {
        'key_obj_1': np.random.rand(1000, 1000).astype(np.float32),
        'key_list_1': [1, 2, 3],
        'key_dict_1': {
            'key_obj_3': np.random.rand(10000, 1000).astype(np.float64),
        },
        'key_obj_2': 2
    }
    print(_get_size(data))


def test_flush():
    dl = DataLogger('test_data', chunk_size_mb=10)
    n = 100
    frame_size = 1024 * 1024
    for _ in range(n):
        data = {
            'key_obj_1': np.random.rand(int(frame_size / 4)).astype(np.float32),
        }
        dl.log(data)
    del dl
    assert len(os.listdir('test_data')) >= 10


if __name__ == "__main__":
    test_accumulation()
    test_get_size()
    test_flush()
