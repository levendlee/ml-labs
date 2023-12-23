"""Sharded matrix multiplication."""

# This lab is inspired by
# https://irhum.github.io/blog/pjit/#intro-parallelism

import functools

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


def dim_sharding_group_id_fn(sharding: DimSharding, index: MeshIndex):
    return ((index.x // sharding.x_shard), (index.y // sharding.y_shard))


class MatchedInnerShardedMatMul(Op):
    """Matmul with matched inner sharding."""
    def __init__(self, sharding: MatMulSharding):
        super().__init__(sharding)

        if sharding.a_shards[1] != sharding.b_shards[0]:
            raise NotImplementedError(
                'Inconsistent inner sharding not supported!')
        self._inner_sharding: DimSharding = sharding.a_shards[1]

    def __call__(self, a: np.ndarray, b: np.ndarray, *,
                 device: VirtualDevice) -> np.ndarray:
        # 1. Run matmul on shared inputs.
        # 2. Get unsharded output.
        # 2. Reduce.

        c = np.matmul(a, b)

        device.log('Virtual device (%s) ran matmul %s @ %s -> %s!', device,
                   a.shape, b.shape, c.shape)

        reduce_fn = lambda a, b: a + b
        return device.all_reduce(
            c, reduce_fn,
            functools.partial(dim_sharding_group_id_fn, self._inner_sharding))


class UnmatchedInnerShardedMatMul(Op):
    """Matmul with unmatched inner sharding."""
    def __call__(self, a: np.ndarray, b: np.ndarray, *,
                 device: VirtualDevice) -> np.ndarray:
        # 1. Gather.
        # 2. Run matmul on unsharded inputs.
        # 2. Get unsharded output.

        a_shards = device.all_gather(
            a,
            functools.partial(dim_sharding_group_id_fn,
                              self._sharding.a_shards[1]))
        b_shards = device.all_gather(
            b,
            functools.partial(dim_sharding_group_id_fn,
                              self._sharding.b_shards[0]))

        full_a = np.concatenate(a_shards, axis=1)
        full_b = np.concatenate(b_shards, axis=0)
        device.log('Virtual device (%s) ran matmul %s @ %s -> ?!', device,
                   full_a.shape, full_b.shape)
        full_c = np.matmul(full_a, full_b)

        device.log('Virtual device (%s) ran matmul %s @ %s -> %s!', device,
                   full_a.shape, full_b.shape, full_c.shape)
        return full_c


def create_matmul_op(sharding: MatMulSharding) -> Op:
    if sharding.full:
        return UnsharedMatMul(sharding)
    if sharding.inner_sharding:
        if sharding.a_shards[1] == sharding.b_shards[0]:
            return MatchedInnerShardedMatMul(sharding)
        else:
            return UnmatchedInnerShardedMatMul(sharding)
    raise NotImplementedError('Sharding policy not supported! '
                              f'Sharding: {sharding}.')
