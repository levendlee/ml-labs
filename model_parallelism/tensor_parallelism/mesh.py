"""Hardware mesh."""

import dataclasses

# 2D mesh setting.


@dataclasses.dataclass
class MeshIndex:
    """2D mesh index."""
    x: int
    y: int


@dataclasses.dataclass
class Mesh:
    """2D mesh setting.
    
    Attributes:
      x_dim:  Horizontal shards.
      y_dim:  Vertical shards.
  """
    x_dim: int
    y_dim: int

    def __iter__(self):
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                yield MeshIndex(x, y)
