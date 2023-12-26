"""Pipelined MLP run."""

from typing import Sequence

import numpy as np

from parallelism.operation import PipelinedOperation, PipelineFunctions
from parallelism.pipelining import Pipeline, PipelineStage

Tensor = np.ndarray


class MLP(PipelinedOperation):
    """MLP."""

    # This example is really simple, as all layers are the same. In practice,
    # we will have different layers.

    # Here, the optimzer state is not considered. For pipeline parallelism
    # itself, as each device has complete and exclusive ownership to a specific
    # layer, so there is no sysnchronization issues on weights/optimizers. They
    # can be handled in the same way, pinned on device. So optimizers are not an
    # interesting topic here.
    def __init__(self, pipeline: Pipeline, learning_rate: float) -> None:
        super().__init__(pipeline=pipeline)
        self._learning_rate = learning_rate

    def __str__(self):
        return (f'{self.__class__.__name__}'
                f'({self._pipeline}, learning_rate={self._learning_rate})')

    @classmethod
    def dispatch(cls, stage: PipelineStage) -> PipelineFunctions:
        return PipelineFunctions(fwd_fn=dense_forward,
                                 bwd_fn=dense_backward,
                                 loss_fn=loss_gradients)


def dense_forward(*, op: PipelinedOperation, activations: Sequence[Tensor],
                  parameters: Sequence[Tensor], **kwargs) -> Tensor:
    x = activations[0]
    w, b = parameters
    return [np.maximum(np.matmul(x, w) + b, 0.0)]


def loss_gradients(*, op: PipelinedOperation, activations: Sequence[Tensor],
                   targets: Sequence[Tensor], **kwargs) -> Tensor:
    y = activations[0]
    targets = targets[0]
    dy = 2 * (y - targets) / y.size
    return [np.where(y >= 0.0, dy, 0.0)]


# The backward path includes next layer relu.
def dense_backward(*, op: PipelinedOperation, gradients: Sequence[Tensor],
                   activations: Sequence[Tensor], parameters: Sequence[Tensor],
                   stage: PipelineStage,
                   **kwargs) -> tuple[Sequence[Tensor], Sequence[Tensor]]:
    assert len(gradients) == 1
    dy = gradients[0]
    assert len(activations) == 1
    x = activations[0]
    w, b = parameters
    db = np.sum(dy, axis=0)
    dw = np.matmul(x.T, dy)
    dx = np.matmul(dy, w.T)
    if not stage.first_stage:
        dx = np.where(x >= 0.0, dx, 0.0)
    return [dx], [w - op._learning_rate * dw, b - op._learning_rate * db]
