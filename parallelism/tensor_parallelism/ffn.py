"""Sharded feed-forward layer in Transformer."""

# GSPMD: General and scalable parallelization for ML computation graphs
# https://arxiv.org/abs/2105.04663

import dataclasses
import enum
import functools
from typing import Callable, Mapping, Sequence

import numpy as np

from parallelism.cluster import VirtualDevice
from parallelism.operation import ShardedOperation
from parallelism.sharding import DimSharding, Sharding, TensorSharding
from parallelism.utils import *

Tensor = np.ndarray


class FeedforwardShardingPolicy(enum.Enum):
    Unsharded = 1
    GSPMD = 2
    Unsupported = 3


@dataclasses.dataclass
class FeedforwardSharding(Sharding):
    """Sharding configuration of matrix multiplication."""
    tensor_shards: tuple[TensorSharding, TensorSharding, TensorSharding,
                         TensorSharding, TensorSharding, TensorSharding,
                         TensorSharding]

    @property
    def x1_shards(self):
        return self.tensor_shards[0]

    @property
    def w1_shards(self):
        return self.tensor_shards[1]

    @property
    def b1_shards(self):
        return self.tensor_shards[2]

    @property
    def x2_shards(self):
        return self.tensor_shards[3]

    @property
    def w2_shards(self):
        return self.tensor_shards[4]

    @property
    def b2_shards(self):
        return self.tensor_shards[5]

    @property
    def o_shards(self):
        return self.tensor_shards[6]

    @property
    def full(self) -> bool:
        return all(map(lambda s: s.full, self.tensor_shards))

    @property
    def gspmd_sharding(self):
        # X1: X,_,Y
        # W1: X,Y
        # X2: X,_,Y
        # W2: Y,X
        # O" X,_,Y
        # Shouldn't need to shard bias in practice?
        # Shard for consistency.
        return (self.x1_shards.xy_sharded_at(0, 2)
                and self.w1_shards.xy_sharded_at(0, 1)
                and self.b1_shards.sharded_at(0)
                and self.x2_shards.xy_sharded_at(0, 2)
                and self.w2_shards.xy_sharded_at(1, 0)
                and self.b2_shards.sharded_at(0)
                and self.o_shards.xy_sharded_at(0, 2))

    @property
    def policy(self) -> FeedforwardShardingPolicy:
        if self.full:
            return FeedforwardShardingPolicy.Unsharded
        if self.gspmd_sharding:
            return FeedforwardShardingPolicy.GSPMD
        return FeedforwardShardingPolicy.Unsupported


def update_einsum_flops(x: Tensor, w: Tensor, device: VirtualDevice):
    if x.shape[-1] != w.shape[0]:
        raise ValueError('Contraction dimension mismatch! '
                         f'x: {x.shape}, w: {w.shape}.')
    b, s, m = x.shape
    _, h = w.shape
    device.stats.flops += 2 * b * s * m * h


def unsharded_ffn(x1_shard: Tensor, w1_shard: Tensor, b1_shard: Tensor,
                  w2_shard: Tensor, b2_shard: Tensor, *, device: VirtualDevice,
                  sharding: FeedforwardSharding) -> Tensor:
    update_einsum_flops(x1_shard, w1_shard, device)
    o = np.einsum('abc,cd->abd', x1_shard, w1_shard)
    # Ignore bias add flops.
    o += b1_shard
    # Should be a configuration activation. Use ReLU for simplicity.
    o = np.maximum(o, 0.0)

    update_einsum_flops(o, w2_shard, device)
    o = np.einsum('abc,cd->abd', o, w2_shard)
    o += b2_shard
    return o


def gspmd_ffn(x1_shard: Tensor, w1_shard: Tensor, b1_shard: Tensor,
              w2_shard: Tensor, b2_shard: Tensor, *, device: VirtualDevice,
              sharding: FeedforwardSharding) -> Tensor:
    # X: (shard_X, full, shard_Y)  # (batch, seq_len, embed)
    # W: (shard_X, shard_Y)  # (embed, hidden)

    x1_s_shards = device.all_gather(shard=x1_shard,
                                    group_id_fn=functools.partial(
                                        get_dim_sharding_group,
                                        sharding.x1_shards[2]))
    x1_s_shard = np.concatenate(x1_s_shards, axis=2)
    w1_h_shards = device.all_gather(shard=w1_shard,
                                    group_id_fn=functools.partial(
                                        get_dim_sharding_group,
                                        sharding.w1_shards[0]))
    w1_h_shard = np.concatenate(w1_h_shards, axis=0)

    update_einsum_flops(x1_s_shard, w1_h_shard, device)
    o_sh_shard = np.einsum('abc,cd->abd', x1_s_shard, w1_h_shard)
    o_sh_shard += b1_shard
    o_sh_shard = np.maximum(o_sh_shard, 0.0)

    # X: (shard_X, full, shard_Y)  # (batch, seq_len, hidden)
    # W: (shard_Y, shard_X)  # (hidden, embed)
    w2_h_shards = device.all_gather(shard=w2_shard,
                                    group_id_fn=functools.partial(
                                        get_dim_sharding_group,
                                        sharding.w2_shards[1]))
    w2_h_shard = np.concatenate(w2_h_shards, axis=1)
    update_einsum_flops(o_sh_shard, w2_h_shard, device)
    o_s_shard = np.einsum('abc,cd->abd', o_sh_shard, w2_h_shard)

    # x2 has sharing at `b` dimension, which we already sharded, and
    # don't need further sharding at `reduce_scatter`.
    o_sharding = TensorSharding([DimSharding(1, 1), *sharding.x2_shards[1:]])
    o_sh_shard = device.reduce_scatter(
        o_s_shard,
        shard_fn=functools.partial(shard_tensor, o_sharding),
        reduce_fn=add_fn,
        group_id_fn=functools.partial(get_dim_sharding_group,
                                      sharding.x2_shards[2]))
    o_sh_shard += b2_shard

    return o_sh_shard


def unsupported_ffn(*args, sharding: FeedforwardShardingPolicy,
                    **kwargs) -> Tensor:
    raise NotImplementedError(f'Unsupported policy: {sharding}!')


class Feedforward(ShardedOperation):
    """Multihead Feedforward."""

    dispatch_dict: Mapping[FeedforwardShardingPolicy, Callable[
        [np.ndarray, np.ndarray, VirtualDevice, FeedforwardSharding],
        np.ndarray]] = {
            FeedforwardShardingPolicy.Unsharded: unsharded_ffn,
            FeedforwardShardingPolicy.GSPMD: gspmd_ffn,
            FeedforwardShardingPolicy.Unsupported: unsupported_ffn
        }

    @classmethod
    def dispatch(cls, sharding: FeedforwardSharding) -> Callable[..., Tensor]:
        return cls.dispatch_dict[sharding.policy]

    def __str__(self):
        return f'{self.__class__.__name__}({self._sharding.policy})'

    @property
    def sharding(self) -> FeedforwardSharding:
        return self._sharding

    @property
    def activation_shardings(self) -> Sequence[TensorSharding]:
        return (self.sharding.x1_shards, )

    @property
    def parameter_shardings(self) -> Sequence[TensorSharding]:
        return (self.sharding.w1_shards, self.sharding.b1_shards,
                self.sharding.w2_shards, self.sharding.b2_shards)
