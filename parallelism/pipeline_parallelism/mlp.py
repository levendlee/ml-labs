"""Pipelined MLP run."""

from typing import Sequence

import numpy as np

from parallelism.cluster import VirtualDevice
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


def update_flops(a: Tensor, b: Tensor, device: VirtualDevice):
    m, k = a.shape
    _, n = b.shape
    device.stats.flops += 2 * m * k * n


def dense_forward(*, op: PipelinedOperation, activations: Sequence[Tensor],
                  parameters: Sequence[Tensor], device: VirtualDevice,
                  **kwargs) -> Tensor:
    x = activations[0]
    w, b = parameters
    update_flops(x, w, device=device)
    y = np.matmul(x, w) + b
    return [np.maximum(y, 0.0)], [x, y]


def loss_gradients(*, op: PipelinedOperation, activations: Sequence[Tensor],
                   targets: Sequence[Tensor],
                   **kwargs) -> tuple[Tensor, float]:
    y = activations[0]
    targets = targets[0]
    loss = np.sum((y - targets)**2) / y.size
    dy = 2 * (y - targets) / y.size
    return [dy], loss


# The backward path includes next layer relu.
def dense_backward(*, op: PipelinedOperation, gradients: Sequence[Tensor],
                   states: Sequence[Tensor], parameters: Sequence[Tensor],
                   stage: PipelineStage, device: VirtualDevice,
                   **kwargs) -> tuple[Sequence[Tensor], Sequence[Tensor]]:
    assert len(gradients) == 1
    dz = gradients[0]
    x, y = states
    w, b = parameters

    dy = np.where(y >= 0.0, dz, 0.0)
    db = np.sum(dy, axis=0)
    device.log('Stage: %s. db max: %f, min: %f', stage, np.max(db), np.min(db))
    update_flops(x.T, dy, device=device)
    dw = np.matmul(x.T, dy)
    device.log('Stage: %s. dw max: %f, min: %f', stage, np.max(dw), np.min(dw))
    update_flops(dy, w.T, device=device)
    dx = np.matmul(dy, w.T)
    return [dx], [w - op._learning_rate * dw, b - op._learning_rate * db]
