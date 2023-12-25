"""Distributed operations."""

import abc
import dataclasses
from typing import Callable, Sequence

import numpy as np

from parallelism.mesh import MeshIndex
from parallelism.pipelining import PipelineStage
from parallelism.sharding import Sharding, TensorSharding
from parallelism.utils import *

Tensor = np.ndarray


class Operation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, device):
        pass

    @abc.abstractmethod
    def prepare_h2d(self, activations: Sequence[Tensor],
                    parameters: Sequence[Tensor],
                    index: MeshIndex) -> Sequence[Sequence[Tensor]]:
        pass


class ShardedOperation(Operation):
    def __init__(self, sharding: Sharding) -> None:
        self._fn = self.dispatch(sharding)
        self._sharding = sharding

    def __call__(self, device):
        device.log('Getting inputs!')
        activations = device.memcpy_h2d()
        parameters = device.memcpy_h2d()

        outputs = self._fn(*(activations + parameters),
                           device=device,
                           sharding=self._sharding)

        device.log('Sending outputs %s!', format_arrays(outputs))
        device.memcpy_d2h([outputs, device.stats, device.channels.stats])

    def prepare_h2d(self, activations: Sequence[Tensor],
                    parameters: Sequence[Tensor],
                    index: MeshIndex) -> Sequence[Sequence[Tensor]]:
        return [
            shard_input_tensors(shardings=self.activation_shardings,
                                tensors=activations,
                                index=index),
            shard_input_tensors(shardings=self.parameter_shardings,
                                tensors=parameters,
                                index=index)
        ]

    @abc.abstractclassmethod
    def activation_shardings(self) -> Sequence[TensorSharding]:
        pass

    @abc.abstractclassmethod
    def parameter_shardings(self) -> Sequence[TensorSharding]:
        pass

    @abc.abstractclassmethod
    def dispatch(cls, sharding: Sharding) -> Callable[..., Tensor]:
        pass


@dataclasses.dataclass
class PipelineFunctions:
    fwd_fn: Callable[..., Tensor]
    bwd_fn: Callable[..., Tensor]


class PipelinedOperation(Operation):
    @abc.abstractclassmethod
    def dispatch(cls, stage: PipelineStage) -> PipelineFunctions:
        """Dispatch initialization, forward and backward functions."""
        pass

    def __init__(self, stage: PipelineStage) -> None:
        self._fns = self.dispatch(stage)
        self._stage = stage

    def forward(self, device):
        if self._stage.first_stage:
            activations = device.memcpy_h2d()
        else:
            activations = device.channels.receive(src=(self._stage -
                                                       1).mesh_index,
                                                  dst=self._stage.mesh_index)
        ret = self._fns.fwd_fn(self, *activations, device=device)
        if self._stage.last_stage:
            self._activations = activations
        else:
            device.channels.send(src=self._stage.mesh_index,
                                 dst=(self._stage + 1).mesh_index,
                                 data=ret)

    def backward(self, device):
        if self._stage.last_stage:
            activations = self._activations
        else:
            activations = device.channels.receive(src=(self._stage +
                                                       1).mesh_index,
                                                  dst=self._stage.mesh_index)
        ret = self._fns.bwd_fn(self, *activations, device=device)
        if self._stage.first_stage:
            device.memcpy_d2h(ret)
        else:
            device.channels.send(src=self._stage.mesh_index,
                                 dst=(self._stage - 1).mesh_index,
                                 data=ret)

    def __call__(self, device):
        with self._stage:
            if self._stage.first_run:
                device.log('Stage: %s i getting parameters from host!')
                self._params = device.memcpy_h2d()
            if self._stage.forward:
                device.log('Stage: %s is running forward path!')
                self.forward(device)
            elif self._stage.backward:
                device.log('Stage: %s is running backward path!')
                self.backward(device)
            else:
                assert self._stage.idle
                device.log('Stage: %s is idle!')
                ret = None
            return ret
