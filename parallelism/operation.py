"""Distributed operations."""

import abc
import dataclasses
import itertools
from typing import Callable, Sequence

import numpy as np

from parallelism.mesh import MeshIndex
from parallelism.pipelining import Pipeline, PipelineStage
from parallelism.sharding import Sharding, TensorSharding
from parallelism.utils import *

Tensor = np.ndarray


class Operation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, device):
        pass

    @abc.abstractmethod
    def prepare_h2d(self, activations: Sequence[Tensor],
                    parameters: Sequence[Tensor], index: MeshIndex,
                    **kwargs) -> Sequence[Sequence[Tensor]]:
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
    loss_fn: Callable[..., Tensor]


class PipelinedOperation(Operation):
    @abc.abstractclassmethod
    def dispatch(cls, stage: PipelineStage) -> PipelineFunctions:
        """Dispatch initialization, forward and backward functions."""
        pass

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline
        self._activations = [None for _ in range(pipeline.num_runs)]
        self._parameters = None
        self._gradients = [None for _ in range(pipeline.num_runs)]

    def get_stage(self, index: MeshIndex) -> PipelineStage:
        return self._pipeline[index.x * self._pipeline.mesh.y_dim + index.y]

    def prepare_h2d(self, activations: Sequence[Tensor],
                    parameters: Sequence[Sequence[Tensor]],
                    targets: Sequence[Tensor],
                    index: MeshIndex) -> Sequence[Sequence[Tensor]]:
        stage = self.get_stage(index)
        params = parameters[stage.pipeline_index]
        if stage.first_stage:
            return [params, *batching(self._pipeline.num_runs, activations)]
        if stage.last_stage:
            return [params, *batching(self._pipeline.num_runs, targets)]
        return [params]

    def __call__(self, device):
        stage = self.get_stage(device.index)
        self._fns = self.dispatch(stage)
        with stage:
            if stage.first_run:
                device.log('Stage: %s is getting parameters from host!', stage)
                self._parameters = device.memcpy_h2d()
            if stage.forward:
                device.log('Stage: %s is running forward path!', stage)
                self.forward(device)
            elif stage.backward:
                device.log('Stage: %s is running backward path!', stage)
                self.backward(device)
            else:
                assert stage.idle
                device.log('Stage: %s is idle!', stage)
                ret = None
            self.report_memory(device)
            return ret

    def report_memory(self, device):
        stage = self.get_stage(device.index)

        arrays = itertools.chain(itertools.chain(self._activations), self._parameters,
                                 itertools.chain(self._gradients))
        device.log(
            'Stage: %s consumes memory: %s MB', stage,
            sum(arr.size * arr.itemsize
                for arr in arrays if arr is not None) / (1024 * 1024))

    def forward(self, device):
        stage = self.get_stage(device.index)
        # Forward pass: Passing activations.
        if stage.first_stage:
            # First stage -> Get activations from host.
            device.log('Stage: %s is getting activations from host!', stage)
            activations = device.memcpy_h2d()
        else:
            activations = device.channels.receive(src=(stage - 1).mesh_index,
                                                  dst=stage.mesh_index)
        # Cache activations.
        self._activations[stage.current_run] = activations
        ret = self._fns.fwd_fn(self,
                               activations,
                               self._parameters,
                               device=device)

        if stage.last_stage:
            # Last stage -> Return gradients instead of activations.
            device.log('Stage: %s is getting targets from host!', stage)
            targets = device.memcpy_h2d()
            gradients = self._fns.loss_fn(ret, targets)
            self._gradients[stage.current_run] = gradients
        else:
            device.channels.send(src=stage.mesh_index,
                                 dst=(stage + 1).mesh_index,
                                 data=ret)

    def backward(self, device):
        stage = self.get_stage(device.index)
        # Backward pass: Passing gradients.
        if stage.last_stage:
            # Last stage -> Get gradients local cache
            gradients = self._gradients[stage.current_run]
            self._gradients[stage.current_run] = None
        else:
            gradients = device.channels.receive(src=(stage + 1).mesh_index,
                                                dst=stage.mesh_index)
        activations = self._activations[stage.current_run]
        self._activations[stage.current_run] = None
        ret = self._fns.bwd_fn(self,
                               gradients,
                               activations,
                               self._parameters,
                               device=device)
        if stage.first_stage:
            device.log('Stage: %s is sending gradients to host!', stage)
            device.memcpy_d2h(ret)
        else:
            device.channels.send(src=stage.mesh_index,
                                 dst=(stage - 1).mesh_index,
                                 data=ret)
