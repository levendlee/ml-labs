"""Model pipelining at various levels."""

import dataclasses
from typing import Iterator

from parallelism.mesh import Mesh


@dataclasses.dataclass(frozen=True)
class Pipeline:
    mesh: Mesh
    num_runs: int
    num_stages: int

    def __pos_init__(self, *args, **kwargs):
        assert self.num_stages == self.mesh.x_dim * self.mesh.y_dim
        self.stages = tuple(
            PipelineStage(mesh_index=mesh_index,
                          pipeline=self,
                          pipeline_index=pipeline_index)
            for (pipeline_index,
                 mesh_index) in zip(range(self.num_stages), self.mesh))

    def __iter__(self) -> Iterator['PipelineStage']:
        return iter(self.get_stage)


@dataclasses.dataclass
class PipelineStage:
    mesh_index: Mesh
    pipeline: Pipeline
    pipeline_index: int
    # Each stage keeps its own copy of cycle to avoid synchronization.
    pipeline_cycle: int = 0

    def __enter__(self, *args, **kwargs):
        self.in_context = True
        return self

    def __exit__(self, *exc):
        self.in_context = False
        self.pipeline_cycle += 1
        return False

    def __add__(self, offset: int) -> 'PipelineStage':
        return self.pipeline.stages[self.pipeline_index + offset]

    def __sub__(self, offset: int) -> 'PipelineStage':
        return self.pipeline.stages[self.pipeline_index - offset]

    def check_status(self):
        if not getattr(self, 'in_context', False):
            raise RuntimeError(
                'Cannot query the status of `PipelineStage` outside its '
                'context. Please use `with PipelineStage():` to wrap each '
                'pipeline cycle.')

    @property
    def forward(self) -> bool:
        self.check_status()
        # Start: the first run hits the stage.
        # End: the first run hits the stage.
        return (self.pipeline_cycle >= self.pipeline_index) and (
            self.pipeline_cycle < self.pipeline_index + self.pipeline.num_runs)

    @property
    def backward(self) -> bool:
        self.check_status()
        start_cycle = 2 * self.pipeline.num_runs - self.pipeline_index - 1
        return (self.pipeline_cycle >= start_cycle) and (
            self.pipeline_cycle < start_cycle + self.pipeline.num_runs)

    @property
    def idle(self) -> bool:
        self.check_status()
        return not self.forward and not self.backward

    @property
    def first_run(self) -> bool:
        self.check_status()
        return self.pipeline_cycle == 0

    @property
    def first_stage(self) -> bool:
        self.check_status()
        return self.pipeline_index == 0

    @property
    def last_stage(self) -> bool:
        self.check_status()
        return self.pipeline_index == self.pipeline.num_stages - 1