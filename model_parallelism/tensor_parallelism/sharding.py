"""Model sharding at various levels.

Assumes 2D mesh.
"""

import dataclasses
from typing import Sequence


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

    @property
    def x_sharding(self) -> bool:
        return self.x_shard != 1 and self.y_shard == 1

    @property
    def y_sharding(self) -> bool:
        return self.x_shard == 1 and self.y_shard != 1


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

    def xy_sharded_at(self, x_shard_dim: int, y_shard_dim: int) -> bool:
        return (self.dim_shards[x_shard_dim].x_sharding
                and self.dim_shards[y_shard_dim].y_sharding and all(
                    map(lambda dim: self.dim_shards[dim].full, [
                        dim for dim in range(len(self.dim_shards))
                        if dim not in (x_shard_dim, y_shard_dim)
                    ])))
