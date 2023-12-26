""""Random utils."""

import logging
from typing import Any, Sequence

import numpy as np

from parallelism.mesh import MeshIndex
from parallelism.sharding import DimSharding, TensorSharding

__all__ = [
    'add_fn',
    'batching',
    'format_arrays',
    'get_dim_sharding_group',
    'shard_tensor',
    'shard_input_tensors',
    'shard_output_tensor',
]

Tensor = np.ndarray


# String format functions.
def format_slices(slices: Sequence[slice]) -> str:
    assert all(s.step is None for s in slices)
    return ', '.join(f'{s.start}:{s.stop}' for s in slices)


def format_arrays(arrays: Sequence[Tensor] | Tensor) -> str:
    if isinstance(arrays, Tensor):
        arrays = [arrays]
    return ', '.join(map(lambda a: str(a.shape), arrays))


# Group functions.
def get_dim_sharding_group(sharding: DimSharding,
                           index: MeshIndex) -> tuple[int, int]:
    return (index.x // sharding.x_shard), (index.y // sharding.y_shard)


# Reduce functions.
def add_fn(a: Any, b: Any) -> Any:
    return a + b


# Shard function.
def shard_tensor(tensor_sharding: TensorSharding, tensor: Tensor,
                 index: MeshIndex) -> Tensor:
    if tensor_sharding.full:
        logging.info('No sharding.')
        return tensor
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
    slices = tuple(slices)
    logging.info(f'Sharding: [{format_slices(slices)}]')
    # YAPF crashed if using tensor[*slices]
    return tensor.__getitem__(tuple(slices))


def shard_input_tensors(shardings: Sequence[TensorSharding],
                        tensors: Sequence[Tensor],
                        index: MeshIndex) -> Sequence[Tensor]:
    assert len(tensors) == len(shardings)
    shards = []
    for i, (tensor, tensor_sharding) in enumerate(zip(tensors, shardings)):
        logging.info(f'Device={index}, Input={i}:')
        shards.append(
            shard_tensor(tensor_sharding=tensor_sharding,
                         tensor=tensor,
                         index=index))
    return shards


def shard_output_tensor(sharding: TensorSharding, tensor: Tensor,
                        indices: Sequence[MeshIndex]) -> Sequence[Tensor]:
    shards = []
    for index in indices:
        logging.info(f'Device={index}:')
        shards.append(
            shard_tensor(tensor_sharding=sharding, tensor=tensor, index=index))
    return shards


def batching(num_batches: int,
             tensors: Sequence[Tensor]) -> Sequence[Sequence[Tensor]]:
    batched = []
    for i in range(num_batches):
        batch = []
        for t in tensors:
            assert t.shape[0] % num_batches == 0
            batch_size = t.shape[0] // num_batches
            slices = [slice(i * batch_size, i * batch_size + batch_size)
                      ] + [slice(d) for d in t.shape[1:]]
            batch.append(t.__getitem__(tuple(slices)))
        batched.append(batch)
    return batched
