"""Sharded matrix multiplication."""

# This lab is inspired by
# https://irhum.github.io/blog/pjit/#intro-parallelism

import functools
from typing import Any, Sequence

import numpy as np

from device import VirtualDevice
from mesh import MeshIndex
from op import Op
from sharding import DimSharding, MatMulSharding


def update_flops(a: np.ndarray, b: np.ndarray, device: VirtualDevice):
    m, k = a.shape
    _, n = b.shape
    device.stats.flops += 2 * m * k * n


class UnsharedMatMul(Op):
    """Matmul without sharding."""
    def __call__(self, a: np.ndarray, b: np.ndarray, *,
                 device: VirtualDevice) -> np.ndarray:
        update_flops(a, b, device)
        return np.matmul(a, b)


def add_fn(a: Any, b: Any) -> Any:
    return a + b


def dim_sharding_group_id_fn(sharding: DimSharding,
                             index: MeshIndex) -> tuple[int, int]:
    return (index.x // sharding.x_shard), (index.y // sharding.y_shard)


def ndim_sharding_group_id_fn(shardings: Sequence[DimSharding],
                              index: MeshIndex) -> Sequence[tuple[int, int]]:
    return tuple(dim_sharding_group_id_fn(s, index) for s in shardings)


class MatchedInnerShardingMatMul(Op):
    """Matmul with matched inner sharding.
    A: (full, sharded)
    B: (sharded, full)
    """
    def __init__(self, sharding: MatMulSharding):
        super().__init__(sharding)

        if sharding.a_shards[1] != sharding.b_shards[0]:
            raise NotImplementedError(
                'Inconsistent inner sharding not supported!')
        self._inner_sharding: DimSharding = sharding.a_shards[1]

    def __call__(self, a_shard: np.ndarray, b_shard: np.ndarray, *,
                 device: VirtualDevice) -> np.ndarray:
        # 1. Run matmul on shared inputs.
        # 2. Get unsharded output.
        # 2. Reduce.

        c_shard = np.matmul(a_shard, b_shard)

        update_flops(a_shard, b_shard, device)
        device.log('Virtual device (%s) ran matmul %s @ %s -> %s!', device,
                   a_shard.shape, b_shard.shape, c_shard.shape)

        return device.all_reduce(c_shard,
                                 reduce_fn=add_fn,
                                 group_id_fn=functools.partial(
                                     dim_sharding_group_id_fn,
                                     self._inner_sharding))


class UnmatchedInnerShardingMatMul(Op):
    """Matmul with unmatched inner sharding.
    A: (full, sharded_Y)
    B: (sharded_X, full)
    """
    def __call__(self, a_shard: np.ndarray, b_shard: np.ndarray, *,
                 device: VirtualDevice) -> np.ndarray:
        # 1. Gather.
        # 2. Run matmul on un-sharded inputs.
        # 3. Get un-sharded output.

        a_shards = device.all_gather(a_shard,
                                     group_id_fn=functools.partial(
                                         dim_sharding_group_id_fn,
                                         self._sharding.a_shards[1]))
        b_shards = device.all_gather(b_shard,
                                     group_id_fn=functools.partial(
                                         dim_sharding_group_id_fn,
                                         self._sharding.b_shards[0]))

        a = np.concatenate(a_shards, axis=1)
        b = np.concatenate(b_shards, axis=0)
        c = np.matmul(a, b)

        update_flops(a, b, device)
        device.log('Virtual device (%s) ran matmul %s @ %s -> %s!', device,
                   a.shape, b.shape, c.shape)
        return c


class OuterShardingMatMul(Op):
    """Matmul with outer sharding.
    A: (sharded_X, sharded_Y)
    B: (sharded_X, full)
    """
    def __call__(self, a_shard: np.ndarray, b_shard: np.ndarray, *,
                 device: VirtualDevice) -> np.ndarray:
        # 1. Gather along inner dimension.
        # 2. Run matmul on sharded inputs.
        # 3. Get sharded output.
        # 5. Concatenate along outer dimension.

        a_m_shards = device.all_gather(a_shard,
                                       group_id_fn=functools.partial(
                                           dim_sharding_group_id_fn,
                                           self._sharding.a_shards[1]))
        b_shards = device.all_gather(b_shard,
                                     group_id_fn=functools.partial(
                                         dim_sharding_group_id_fn,
                                         self._sharding.b_shards[0]))

        a_m_shard = np.concatenate(a_m_shards, axis=1)
        b = np.concatenate(b_shards, axis=0)
        c_m_shard = np.matmul(a_m_shard, b)

        update_flops(a_m_shard, b, device)
        device.log('Virtual device (%s) ran matmul %s @ %s -> %s!', device,
                   a_m_shard.shape, b.shape, c_m_shard.shape)

        c_m_shards = device.all_gather(c_m_shard,
                                       group_id_fn=functools.partial(
                                           dim_sharding_group_id_fn,
                                           self._sharding.a_shards[0]))
        return np.concatenate(c_m_shards, axis=0)


class FullyShardingMatMul(Op):
    """Matmul with outer sharding.
    A: (sharded_X, sharded_Y)
    B: (sharded_X, sharded_Y)
    """
    def __call__(self, a_shard: np.ndarray, b_shard: np.ndarray, *,
                 device: VirtualDevice) -> np.ndarray:
        # 1. Gather along inner dimension.
        # 2. Run matmul on sharded inputs.
        # 3. Get sharded output.
        # 5. Concatenate along outer dimension.

        a_m_shards = device.all_gather(a_shard,
                                       group_id_fn=functools.partial(
                                           dim_sharding_group_id_fn,
                                           self._sharding.a_shards[1]))
        b_n_shards = device.all_gather(b_shard,
                                       group_id_fn=functools.partial(
                                           dim_sharding_group_id_fn,
                                           self._sharding.b_shards[0]))

        a_m_shard = np.concatenate(a_m_shards, axis=1)
        b_n_shard = np.concatenate(b_n_shards, axis=0)
        c_mn_shard = np.matmul(a_m_shard, b_n_shard)

        update_flops(a_m_shard, b_n_shard, device)
        device.log('Virtual device (%s) ran matmul %s @ %s -> %s!', device,
                   a_m_shard.shape, b_n_shard.shape, c_mn_shard.shape)

        c_mn_shards = device.all_gather(
            c_mn_shard,
            group_id_fn=lambda index: 0)

        num_m_shards = self._sharding.a_shards[0].num_shards
        num_n_shards = self._sharding.b_shards[1].num_shards
        if len(c_mn_shards) != num_m_shards * num_n_shards:
            raise RuntimeError(
                f'Expect to gather {num_m_shards} * {num_n_shards} = '
                f'{num_m_shards * num_n_shards}. '
                f'Actually gathered {len(c_mn_shards)} shards!'
            )

        c_m_shards = [
            np.concatenate(c_mn_shards[start * num_n_shards:start * num_n_shards + num_n_shards], axis=1)
            for start in range(num_m_shards)
        ]
        return np.concatenate(c_m_shards, axis=0)


def create_matmul_op(sharding: MatMulSharding) -> Op:
    if sharding.full:
        return UnsharedMatMul(sharding)
    if sharding.inner_sharding:
        if sharding.a_shards[1] == sharding.b_shards[0]:
            return MatchedInnerShardingMatMul(sharding)
        return UnmatchedInnerShardingMatMul(sharding)
    if sharding.b_shards[1].full:
        return OuterShardingMatMul(sharding)
    return FullyShardingMatMul(sharding)
