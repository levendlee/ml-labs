""""Random utils."""

from typing import Any

import numpy as np

from model_parallelism.mesh import MeshIndex
from model_parallelism.sharding import DimSharding, TensorSharding

__all__ = [
    'add_fn',
    'get_dim_sharding_group',
    'get_tensor_sharding_slices',
    'shard_tensor',
]

Tensor = np.ndarray


# Group id functions.
def get_dim_sharding_group(sharding: DimSharding,
                           index: MeshIndex) -> tuple[int, int]:
    return (index.x // sharding.x_shard), (index.y // sharding.y_shard)


# Reduce functions.
def add_fn(a: Any, b: Any) -> Any:
    return a + b


# Shard function.
def get_tensor_sharding_slices(tensor_sharding: TensorSharding, tensor: Tensor,
                               index: MeshIndex) -> tuple[slice, ...]:
    assert tensor.ndim == len(tensor_sharding.dim_shards)
    slices = []
    for dim, dim_sharding in zip(tensor.shape, tensor_sharding.dim_shards):
        assert dim % dim_sharding.num_shards == 0
        shard_size = dim // dim_sharding.num_shards
        shard_index = (
            (index.x % dim_sharding.x_shard) * dim_sharding.y_shard +
            (index.y % dim_sharding.y_shard))
        # When the mesh is not sharded along a dimension, duplicates
        # happens.
        start = shard_index * shard_size
        stop = start + shard_size
        slices.append(slice(start, stop))
    return tuple(slices)


def shard_tensor(tensor_sharding: TensorSharding, tensor: Tensor,
                 index: MeshIndex) -> Tensor:
    if tensor_sharding.full:
        return tensor
    slices = get_tensor_sharding_slices(tensor_sharding, tensor, index)
    return tensor.__getitem__(tuple(slices))
