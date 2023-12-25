"""Sharded matrix multiplication."""

# This lab is inspired by
# https://irhum.github.io/blog/pjit/#intro-parallelism

import dataclasses
import enum
import functools
from typing import Callable, Mapping

import numpy as np

from model_parallelism.cluster import VirtualDevice
from model_parallelism.op import Op
from model_parallelism.sharding import DimSharding, Sharding, TensorSharding
from model_parallelism.utils import *

Tensor = np.ndarray


class MatMulShardingPolicy(enum.Enum):
    Unshared = 1
    MatchedInnerSharding = 2
    UnmatchedInnerSharding = 3
    OuterSharding = 4
    FullSharding = 5


@dataclasses.dataclass
class MatMulSharding(Sharding):
    """Sharding configuration of matrix multiplication."""
    tensor_shards: tuple[TensorSharding, TensorSharding, TensorSharding]

    @property
    def a_shards(self):
        return self.tensor_shards[0]

    @property
    def b_shards(self):
        return self.tensor_shards[1]

    @property
    def c_shards(self):
        return self.tensor_shards[2]

    @property
    def full(self) -> bool:
        return all(map(lambda s: s.full, self.tensor_shards))

    @property
    def inner_sharding(self):
        return (self.a_shards.sharded_at(1) and self.b_shards.sharded_at(0)
                and self.c_shards.full)

    @property
    def policy(self) -> MatMulShardingPolicy:
        if self.full:
            return MatMulShardingPolicy.Unshared
        elif self.inner_sharding:
            if self.a_shards[1] == self.b_shards[0]:
                return MatMulShardingPolicy.MatchedInnerSharding
            else:
                return MatMulShardingPolicy.UnmatchedInnerSharding
        else:
            if self.b_shards[1].full:
                return MatMulShardingPolicy.OuterSharding
            else:
                return MatMulShardingPolicy.FullSharding


# Statistics functions.
def update_flops(a: Tensor, b: Tensor, device: VirtualDevice):
    m, k = a.shape
    _, n = b.shape
    device.stats.flops += 2 * m * k * n


# Matmul functions.
def unshared_matmul(a: Tensor, b: Tensor, *, device: VirtualDevice,
                    sharding: MatMulSharding) -> Tensor:
    update_flops(a, b, device)
    return np.matmul(a, b)


def matched_inner_sharding_matmul(a_shard: Tensor, b_shard: Tensor, *,
                                  device: VirtualDevice,
                                  sharding: MatMulSharding) -> Tensor:
    """Matmul with matched inner sharding.
    A: (full, sharded)
    B: (sharded, full)
    """
    if sharding.a_shards[1] != sharding.b_shards[0]:
        raise NotImplementedError('Inconsistent inner sharding not supported!')
    inner_sharding: DimSharding = sharding.a_shards[1]

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
                                 get_dim_sharding_group, inner_sharding))


def unmatched_inner_sharding_matmul(a_shard: Tensor, b_shard: Tensor, *,
                                    device: VirtualDevice,
                                    sharding: MatMulSharding) -> Tensor:
    """Matmul with unmatched inner sharding.
    A: (full, sharded_Y)
    B: (sharded_X, full)
    """
    # 1. Gather.
    # 2. Run matmul on un-sharded inputs.
    # 3. Get un-sharded output.

    a_shards = device.all_gather(a_shard,
                                 group_id_fn=functools.partial(
                                     get_dim_sharding_group,
                                     sharding.a_shards[1]))
    b_shards = device.all_gather(b_shard,
                                 group_id_fn=functools.partial(
                                     get_dim_sharding_group,
                                     sharding.b_shards[0]))

    a = np.concatenate(a_shards, axis=1)
    b = np.concatenate(b_shards, axis=0)
    c = np.matmul(a, b)

    update_flops(a, b, device)
    device.log('Virtual device (%s) ran matmul %s @ %s -> %s!', device,
               a.shape, b.shape, c.shape)
    return c


def outer_sharding_matmul(a_shard: Tensor, b_shard: Tensor, *,
                          device: VirtualDevice,
                          sharding: MatMulSharding) -> Tensor:
    """Matmul with outer sharding.
    A: (sharded_X, sharded_Y)
    B: (sharded_X, full)
    """
    # 1. Gather along inner dimension.
    # 2. Run matmul on sharded inputs.
    # 3. Get sharded output.
    # 5. Concatenate along outer dimension.

    a_m_shards = device.all_gather(a_shard,
                                   group_id_fn=functools.partial(
                                       get_dim_sharding_group,
                                       sharding.a_shards[1]))
    b_shards = device.all_gather(b_shard,
                                 group_id_fn=functools.partial(
                                     get_dim_sharding_group,
                                     sharding.b_shards[0]))

    a_m_shard = np.concatenate(a_m_shards, axis=1)
    b = np.concatenate(b_shards, axis=0)
    c_m_shard = np.matmul(a_m_shard, b)

    update_flops(a_m_shard, b, device)
    device.log('Virtual device (%s) ran matmul %s @ %s -> %s!', device,
               a_m_shard.shape, b.shape, c_m_shard.shape)

    c_m_shards = device.all_gather(c_m_shard,
                                   group_id_fn=functools.partial(
                                       get_dim_sharding_group,
                                       sharding.a_shards[0]))
    return np.concatenate(c_m_shards, axis=0)


def fully_sharding_matmul(a_shard: Tensor, b_shard: Tensor, *,
                          device: VirtualDevice,
                          sharding: MatMulSharding) -> Tensor:
    """Matmul with outer sharding.
    A: (sharded_X, sharded_Y)
    B: (sharded_X, sharded_Y)
    """
    # 1. Gather along inner dimension.
    # 2. Run matmul on sharded inputs.
    # 3. Get sharded output.
    # 5. Concatenate along outer dimension.

    a_m_shards = device.all_gather(a_shard,
                                   group_id_fn=functools.partial(
                                       get_dim_sharding_group,
                                       sharding.a_shards[1]))
    b_n_shards = device.all_gather(b_shard,
                                   group_id_fn=functools.partial(
                                       get_dim_sharding_group,
                                       sharding.b_shards[0]))

    a_m_shard = np.concatenate(a_m_shards, axis=1)
    b_n_shard = np.concatenate(b_n_shards, axis=0)
    c_mn_shard = np.matmul(a_m_shard, b_n_shard)

    update_flops(a_m_shard, b_n_shard, device)
    device.log('Virtual device (%s) ran matmul %s @ %s -> %s!', device,
               a_m_shard.shape, b_n_shard.shape, c_mn_shard.shape)

    c_mn_shards = device.all_gather(c_mn_shard, group_id_fn=lambda index: 0)

    num_m_shards = sharding.a_shards[0].num_shards
    num_n_shards = sharding.b_shards[1].num_shards
    if len(c_mn_shards) != num_m_shards * num_n_shards:
        raise RuntimeError(
            f'Expect to gather {num_m_shards} * {num_n_shards} = '
            f'{num_m_shards * num_n_shards}. '
            f'Actually gathered {len(c_mn_shards)} shards!')

    c_m_shards = [
        np.concatenate(c_mn_shards[start * num_n_shards:start * num_n_shards +
                                   num_n_shards],
                       axis=1) for start in range(num_m_shards)
    ]
    return np.concatenate(c_m_shards, axis=0)


class MatMul(Op):
    """Matrix multiplication."""

    dispatch: Mapping[MatMulShardingPolicy,
                      Callable[[Tensor, Tensor, VirtualDevice, MatMulSharding],
                               Tensor]] = {
                                   MatMulShardingPolicy.Unshared:
                                   unshared_matmul,
                                   MatMulShardingPolicy.MatchedInnerSharding:
                                   matched_inner_sharding_matmul,
                                   MatMulShardingPolicy.UnmatchedInnerSharding:
                                   unmatched_inner_sharding_matmul,
                                   MatMulShardingPolicy.OuterSharding:
                                   outer_sharding_matmul,
                                   MatMulShardingPolicy.FullSharding:
                                   fully_sharding_matmul
                               }

    def __str__(self):
        return f'{self.__class__.__name__}({self._sharding.policy})'

    def __call__(self, a_shard: Tensor, b_shard: Tensor, *,
                 device: VirtualDevice) -> Tensor:
        return self.dispatch[self._sharding.policy](a_shard,
                                                    b_shard,
                                                    device=device,
                                                    sharding=self._sharding)
