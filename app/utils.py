import json
from typing import Union, Dict
from datetime import datetime
from decimal import Decimal

from aioredis import Redis
import struct
import numpy as np


def format_datetime(value: datetime):
    """Deserialize datetime object into string form for JSON processing."""
    if value is None:
        return None
    return value.isoformat(timespec="seconds")


def format_precision(value):
    if value is None:
        return None
    return f"{value:.18f}"


def get_attr(
    target: Dict,
    attr: str,
):
    if target is None or attr not in target:
        return None
    else:
        return target[attr]


def format_decimal(
    price: Union[str, None],
    decimals: int,
):
    if price is None:
        return None
    else:
        return Decimal(price) / (10 ** decimals)


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return format_precision(o)
        elif isinstance(o, datetime):
            return format_datetime(o)
        return super(CustomEncoder, self).default(o)


async def save_np_to_redis(r: Redis, a, key):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    h, w = a.shape
    shape = struct.pack('>II', h, w)
    encoded = shape + a.tobytes()

    # Store encoded data in Redis
    await r.set(key, encoded)
    return


async def load_np_from_redis(r: Redis, key) -> np.ndarray:
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = await r.get(key)
    h, w = struct.unpack('>II', encoded[:8])
    # Add slicing here, or else the array would differ from the original
    a = np.frombuffer(encoded[8:]).reshape(h, w)
    return a
