"""openpi-byte-compatible msgpack + numpy codec.

OmniGibson's BEHAVIOR-1K eval driver talks to a policy server with the wire
format from ``Physical-Intelligence/openpi``
(``packages/openpi-client/src/openpi_client/msgpack_numpy.py``): msgpack with a
*custom* numpy codec (a fork of the PyPI ``msgpack-numpy`` that drops the pickle
fallback for object arrays). This module reproduces that codec exactly so the
bridge server is byte-compatible with ``WebsocketClientPolicy`` while depending
only on ``msgpack`` (not openpi, not the ``msgpack_numpy`` PyPI package, whose
on-wire keys differ).

An ``ndarray`` packs to a map with **bytes** keys::

    {b"__ndarray__": True, b"data": <raw tobytes>, b"dtype": "<f4", b"shape": (..)}

and an ``np.generic`` scalar to ``{b"__npgeneric__": True, b"data": <py scalar>,
b"dtype": "<f4"}``. Dtypes of kind V/O/c (void/object/complex) are rejected.
"""

import functools

import msgpack
import numpy as np


def pack_array(obj):
    """msgpack ``default`` hook: encode numpy arrays / scalars (openpi layout)."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    """msgpack ``object_hook``: rebuild numpy arrays / scalars from the openpi layout."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
