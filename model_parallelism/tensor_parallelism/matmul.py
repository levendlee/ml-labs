"""Shared matrix multiplication."""

# This lab is inspired by
# https://irhum.github.io/blog/pjit/#intro-parallelism

import numpy as np

from device import VirtualDevice
from mesh import MeshIndex
from op import Op
from sharding import DimSharding, MatMulSharding


class UnsharedMatMul(Op):
    """Matmul without sharding."""
    def __call__(self, a: np.ndarray, b: np.ndarray, *,
                 device: VirtualDevice) -> np.ndarray:
        return np.matmul(a, b)


class InnerShardedMatMul(Op):
    """Matmul with inner sharding."""
    def __init__(self, sharding: MatMulSharding):
        super().__init__(sharding)

        if sharding.a_shards[1] != sharding.b_shards[0]:
            raise NotImplementedError(
                'Inconsistent inner sharding not supported!')
        self._inner_sharding: DimSharding = sharding.a_shards[1]

    def __call__(self, a: np.ndarray, b: np.ndarray, *,
                 device: VirtualDevice) -> np.ndarray:

        c = np.matmul(a, b)

        device.log('Virtual device (%s) ran matmul %s @ %s -> %s!', device,
                   a.shape, b.shape, c.shape)

        def group_id_fn(index: MeshIndex):
            return ((index.x // self._inner_sharding.x_shard),
                    (index.y // self._inner_sharding.y_shard))

        reduce_fn = lambda a, b: a + b
        return device.all_reduce(c, reduce_fn, group_id_fn)


def create_matmul_op(sharding: MatMulSharding) -> Op:
    if sharding.full:
        return UnsharedMatMul(sharding)
    if sharding.inner_sharding:
        return InnerShardedMatMul(sharding)
    raise NotImplementedError('Sharding policy not supported! '
                              f'Sharding: {sharding}.')
