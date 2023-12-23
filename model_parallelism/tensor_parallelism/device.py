"""Shared matrix multiplication."""

# This lab is inspired by
# https://irhum.github.io/blog/pjit/#intro-parallelism

from multiprocessing import Process, Queue
import time
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from mesh import Mesh, MeshIndex
from op import Op
from sharding import TensorSharding

# To determine the group id to run the collective ops on.
GroupIDFnType = Callable[[MeshIndex], Any]
ReduceFnType = Callable[[np.ndarray, np.ndarray], np.ndarray]


# Virtual intra-device communication channels.
class VirtualChannels:
    def __init__(self, mesh: Mesh):
        num_devices = mesh.x_dim * mesh.y_dim
        self._mesh = mesh
        self._h2d_channels = [Queue() for _ in range(num_devices)]
        self._d2h_channels = [Queue() for _ in range(num_devices)]
        self._d2d_channels = [[Queue() for _ in range(num_devices)]
                          for _ in range(num_devices)]

    def to_1d(self, index: MeshIndex) -> int:
        return index.x * self._mesh.y_dim + index.y

    def get_channel(self, src: Optional[MeshIndex], dst: Optional[MeshIndex]) -> Queue:
        if src is None:
          return self._h2d_channels[self.to_1d(dst)]
        elif dst is None:
          return self._d2h_channels[self.to_1d(src)]
        else:
          return self._d2d_channels[self.to_1d(src)][self.to_1d(dst)]

    def send(self, src: Optional[MeshIndex], dst: Optional[MeshIndex], data: Any) -> None:
        self.get_channel(src, dst).put(data)

    def receive(self, src: Optional[MeshIndex], dst: Optional[MeshIndex]) -> Any:
        return self.get_channel(src, dst).get()


# Virtual devices.
class VirtualDevice:
    def __init__(self, mesh: Mesh, index: MeshIndex,
                 channels: VirtualChannels):
        self._mesh = mesh
        self._index = index
        self._channels = channels

    @property
    def index(self) -> MeshIndex:
        return self._index

    def __str__(self) -> str:
        return f'VirtualDevice({self._index})'

    def run(self, op: Op, *args, **kwargs):
        return op(*args, **kwargs, device=self)

    def all_scatter(self, shard: np.ndarray,
                    group_id_fn: GroupIDFnType) -> None:
        print(f'AllScatter request at {self._index}')
        
        # In practice, should do in parallel instead of using loops.
        src = self._index
        group_id = group_id_fn(self._index)
        for dst in self._mesh:
            if src == dst:
                continue
            if group_id != group_id_fn(dst):
                continue
            print(f'Scatter {src} to {dst}')
            self._channels.send(src=src, dst=dst, data=shard)

    # Scatter is used to implement gather, as the Queue works in a push mode
    # instead of a pull mode.
    def all_gather(
            self, shard: np.ndarray,
            group_id_fn: GroupIDFnType) -> Sequence[Sequence[np.ndarray]]:
        # 1. Send out owned shard.
        self.all_scatter(shard, group_id_fn)

        # 2. Get other shards.
        # In practice, should do in parallel instead of using loops.
        gathered = []
        dst = self._index
        group_id = group_id_fn(self._index)
        for dst in self._mesh:
            if group_id != group_id_fn(dst):
                continue
            if src == dst:
                gathered.append(shard)
            else:
                gathered.append(self._channels.receive(src=src, dst=dst))
        return gathered

    # Scatter is used to implement reduce, as the Queue works in a push mode
    # instead of a pull mode.
    def all_reduce(self, shard: np.ndarray, reduce_fn: ReduceFnType,
                   group_id_fn: GroupIDFnType) -> np.ndarray:
        print(f'AllReduce request at {self._index}')

        # 1. Send out owned shard.
        self.all_scatter(shard, group_id_fn)

        # 2. Get and reduce other shards.
        reduced = shard
        dst = self._index
        group_id = group_id_fn(self._index)
        for dst in self._mesh:
            if group_id != group_id_fn(dst):
                continue
            if src == dst:
                continue
            print(f'Reduce {src} with {dst}')
            reduced = reduce_fn(reduced,
                                self._channels.receive(src=src, dst=dst))

        return reduced


def run_op_with_shared_inputs(op, device_, channels):
    index = device_.index
    print(f'Getting inputs on {device_.index}')
    # Use the special channel as client->worker communication
    inputs = channels.receive(None, index)
    print(f'Running op on {device_.index}')
    outputs = device_.run(op, *inputs)
    print(f'Sending output on {device_.index}')
    channels.send(index, None, outputs)


# Virtual device clusters.
class VirtualCluster:
    def __init__(self, x_dim: int, y_dim: int):
        self._mesh = Mesh(x_dim=x_dim, y_dim=y_dim)
        self._channels = VirtualChannels(self._mesh)
        self._devices = [
            VirtualDevice(index=index,
                          mesh=self._mesh,
                          channels=self._channels) for index in self._mesh
        ]

    def run(self, op: Op,
            tensor_and_sharding: Sequence[Tuple[np.ndarray, TensorSharding]]):
        def format_slices(slices: Sequence[slice]) -> str:
          assert all(s.step is None for s in slices)
          return ', '.join(f'{s.start}:{s.stop}' for s in slices)
        def shard_inputs(index: MeshIndex) -> Sequence[np.ndarray]:
            shards = []
            for i, (tensor, tensor_sharding) in enumerate(tensor_and_sharding):
                assert tensor.ndim == len(tensor_sharding.dim_shards)
                slices = []
                for dim, dim_sharding in zip(tensor.shape,
                                             tensor_sharding.dim_shards):
                    assert dim % dim_sharding.num_shards == 0
                    shard_size_0 = dim // dim_sharding.num_shards
                    shard_size_1 = shard_size_0 // dim_sharding.y_shard
                    # When the mesh can hold more than shards, duplicates happens.
                    start = (index.x * shard_size_0 +
                             index.y * shard_size_1) % dim
                    stop = start + shard_size_1
                    slices.append(slice(start, stop))
                # YAPF crashed if using tensor[*slices]
                print(f'Device: {index} get shared input {i}: '
                      f'[{format_slices(slices)}]')
                shards.append(tensor.__getitem__(tuple(slices)))
                # shards = tensor[*slices]
            return shards

        processes = [
            Process(target=run_op_with_shared_inputs,
                    args=(op, d, self._channels))
            for d in self._devices
        ]
        
        for p in processes:
            p.start()

        start = time.time()
        # Send inputs
        for index in self._mesh:
            self._channels.send(None, index, shard_inputs(index)) 

        # Receive outputs
        outputs = []
        for index in self._mesh:
            outputs.append(self._channels.receive(index, None))
        end = time.time()

        print(f'End to end time: {end - start}')

        for p in processes:
            p.join()

        return outputs
