"""Model sharding at various levels.

Assumes 2D mesh.
"""

import dataclasses
from typing import Sequence, Tuple


class Sharding:
    pass


@dataclasses.dataclass
class DimSharding(Sharding):
    """Sharding configuration of dimension.

    Attributes:
      x_shard: Number of shards along X dimension.
      y_shard: Number of shards along Y dimension.
    """
    x_shard: int = 1
    y_shard: int = 1

    @property
    def full(self) -> bool:
        return self.x_shard == 1 and self.y_shard == 1

    @property
    def num_shards(self) -> int:
        return self.x_shard * self.y_shard


@dataclasses.dataclass
class TensorSharding(Sharding):
    """Sharding configuration of tensor."""
    dim_shards: Sequence[DimSharding]

    def __getitem__(self, index: int) -> DimSharding:
        return self.dim_shards[index]

    @property
    def full(self) -> bool:
        return all(map(lambda s: s.full, self.dim_shards))

    def sharded_at(self, dim: int) -> bool:
        return not self.dim_shards[dim].full and all(
            map(lambda s: s.full,
                self.dim_shards[:dim] + self.dim_shards[dim + 1:]))


@dataclasses.dataclass
class MatMulSharding(Sharding):
    """Sharding configuration of matrix multiplication."""
    tensor_shards: Tuple[TensorSharding, TensorSharding, TensorSharding]

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
