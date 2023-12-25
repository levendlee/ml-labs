"""Pipelined MLP run."""

from typing import Callable

import numpy as np

from parallelism.operation import PipelinedOperation
from parallelism.pipelining import PipelineStage

Tensor = np.ndarray


class MLP(PipelinedOperation):
    """MLP."""

    # This example is really simple, as all layers are the same. In practice,
    # we will have different layers.

    @classmethod
    def dispatch(
        cls, stage: PipelineStage
    ) -> tuple[Callable[..., Tensor], Callable[..., Tensor]]:
        pass
