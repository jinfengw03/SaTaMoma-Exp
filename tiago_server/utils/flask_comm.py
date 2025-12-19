import cv2
import gym
import base64
import numpy as np
from typing import Any
from collections import OrderedDict

_INF_REPLACEMENT = 1e10  # Replacement value for inf/-inf to make the data JSON-compliant

def encode_image(img: np.ndarray) -> str:
    if img.dtype != np.uint8:
        img = img.astype(np.uint8) # to uint8 - [0, 255]
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')

def decode_image(data_str):
    jpg_data = base64.b64decode(data_str)
    img_arr = np.frombuffer(jpg_data, dtype=np.uint8) # 1 byte = 8 bits
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

def encode_depth(depth: np.ndarray) -> str:
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16) # to uint16 - [0, 65535]
    _, buffer = cv2.imencode('.png', depth)
    return base64.b64encode(buffer).decode('utf-8')

def decode_depth(data_str):
    png_data = base64.b64decode(data_str)
    img_arr = np.frombuffer(png_data, dtype=np.uint8) # 1 byte = 8 bits
    return cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)

def encode2json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: encode2json(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.integer) and obj.ndim == 3 and obj.shape[-1] == 3:
            return {
                'type': 'image',
                'encoding': 'jpeg',
                'data': encode_image(obj)
            }
        elif np.issubdtype(obj.dtype, np.integer) and obj.ndim == 3 and obj.shape[-1] == 1:
            return {
                'type': 'depth',
                'encoding': 'png',
                'data': encode_depth(obj)
            }
        else:
            return {
                'type': 'array',
                'data': obj.tolist()
            }
    elif isinstance(obj, tuple):
        return {
            'type': 'tuple',
            'data': [encode2json(v) for v in obj]
        }
    elif isinstance(obj, list):
        return {
            'type': 'list',
            'data': [encode2json(v) for v in obj]
        }
    else:
        return obj

def decode4json(obj: Any) -> Any:
    if isinstance(obj, dict):
        obj_type = obj.get('type')
        if obj_type == 'image':
            return decode_image(obj['data'])
        elif obj_type == 'depth':
            return decode_depth(obj['data'])
        elif obj_type == 'array':
            return np.array(obj['data'], dtype=np.float32)
        elif obj_type == 'tuple':
            return tuple(decode4json(v) for v in obj['data'])
        elif obj_type == 'list':
            return [decode4json(v) for v in obj['data']]
        else:
            return {k: decode4json(v) for k, v in obj.items()}
    else:
        return obj

def sanitize_array(arr, dtype=np.float32):
    """
    @func: replace inf, -inf, and NaN in a NumPy array with large/safe values, so the result can be serialized to JSON.
    """
    arr = np.array(arr, dtype=dtype)
    arr = np.nan_to_num(arr, nan=0.0, posinf=_INF_REPLACEMENT, neginf=-_INF_REPLACEMENT)
    return arr.tolist()

def restore_inf(arr, dtype=np.float32):
    """
    @func: restore values close to ±_INF_REPLACEMENT back to ±inf after deserialization.
    """
    arr = np.array(arr, dtype=dtype)
    if np.issubdtype(arr.dtype, np.floating):
        arr[np.isclose(arr, _INF_REPLACEMENT)] = np.inf
        arr[np.isclose(arr, -_INF_REPLACEMENT)] = -np.inf
    return arr

def serialize_space_dict(space_dict):
    """
    @func: serialize a gym.spaces.Dict composed of Box spaces into a JSON-serializable dictionary.t
        this function handles inf, -inf, and NaN by replacing them with large/safe values.
    """
    serialized = OrderedDict()
    for key, box in space_dict.spaces.items():
        serialized[key] = {
            'type': 'Box',
            'low': sanitize_array(box.low, box.dtype),
            'high': sanitize_array(box.high, box.dtype),
            'shape': box.shape,
            'dtype': str(box.dtype)
        }
    return serialized

def restore_inf(arr, dtype=np.float32):
    """
    @func: restore values close to ±_INF_REPLACEMENT back to ±inf after deserialization.
    """
    arr = np.array(arr, dtype=dtype)
    if np.issubdtype(arr.dtype, np.floating):
        arr[np.isclose(arr, _INF_REPLACEMENT)] = np.inf
        arr[np.isclose(arr, -_INF_REPLACEMENT)] = -np.inf
    return arr

def reconstruct_space_dict(data):
    """
    @func: reconstruct a gym.spaces.Dict from a JSON-deserialized dictionary
        by restoring the original inf values and creating Box spaces.
    """
    space_dict = OrderedDict()
    for key, val in data.items():
        if val['type'] == 'Box':
            low = restore_inf(val['low'], dtype=val['dtype'])
            high = restore_inf(val['high'], dtype=val['dtype'])
            shape = tuple(val['shape'])
            dtype = np.dtype(val['dtype'])
            space_dict[key] = gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return gym.spaces.Dict(space_dict)
