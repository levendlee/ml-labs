"""Virtual compute cluster."""

import logging
import time
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Process, Queue
from typing import Any, Callable, Optional, Sequence

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

    def get_channel(self, src: Optional[MeshIndex],
                    dst: Optional[MeshIndex]) -> Queue:
        if src is None:
            return self._h2d_channels[self.to_1d(dst)]
        elif dst is None:
            return self._d2h_channels[self.to_1d(src)]
        else:
            return self._d2d_channels[self.to_1d(src)][self.to_1d(dst)]

    def send(self, src: Optional[MeshIndex], dst: Optional[MeshIndex],
             data: Any) -> None:
        self.get_channel(src, dst).put(data)

    def receive(self, src: Optional[MeshIndex],
                dst: Optional[MeshIndex]) -> Any:
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

    def log(self, fmt, *args):
        prefix = f'{self}: '
        logging.info(prefix + fmt, *args)

    def __str__(self) -> str:
        return f'VirtualDevice({self._index})'

    def run(self, op: Op, *args, **kwargs):
        return op(*args, **kwargs, device=self)

    def all_scatter(self, shard: np.ndarray,
                    group_id_fn: GroupIDFnType) -> None:
        self.log(f'AllScatter request start.')

        # In practice, should do in parallel instead of using loops.
        src = self._index
        group_id = group_id_fn(self._index)
        for dst in self._mesh:
            if src == dst:
                continue
            if group_id != group_id_fn(dst):
                continue
            self.log('Scatter %s to %s.', src, dst)
            self._channels.send(src=src, dst=dst, data=shard)

        self.log(f'AllScatter request finish.')

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
        for src in self._mesh:
            if group_id != group_id_fn(src):
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
        self.log(f'AllReduce request start.')

        # 1. Send out owned shard.
        self.all_scatter(shard, group_id_fn)

        # 2. Get and reduce other shards.
        reduced = shard
        dst = self._index
        group_id = group_id_fn(self._index)
        for src in self._mesh:
            if group_id != group_id_fn(src):
                continue
            if src == dst:
                continue
            self.log(f'Reduce %s with %s', src, dst)
            reduced = reduce_fn(reduced,
                                self._channels.receive(src=src, dst=dst))

        self.log(f'AllReduce request finish.')
        return reduced


def init_queue_listener(q: Queue):
    # Main process handler.
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(levelname)s: %(asctime)s - %(process)s - %(message)s"))

    # Main process logger.
    logger = logging.getLogger('VirtualCluster')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    # ql gets records from the queue and sends them to the handler
    ql = QueueListener(q, handler)
    ql.start()

    return ql


def init_queue_handler(q: Queue):
    # Sub-process handler.
    queue_handler = QueueHandler(q)

    # Sub-process logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(queue_handler)


def format_slices(slices: Sequence[slice]) -> str:
    assert all(s.step is None for s in slices)
    return ', '.join(f'{s.start}:{s.stop}' for s in slices)


def format_arrays(arrays: Sequence[np.ndarray] | np.ndarray) -> str:
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    return ', '.join(map(lambda a: str(a.shape), arrays))


def run_op_with_shared_inputs(op, device, channels, logger_queue):
    init_queue_handler(logger_queue)

    index = device.index
    # Use the special channel as client->worker communication
    device.log('Getting inputs!')
    inputs = channels.receive(None, index)
    device.log('Running op %s!', op)
    outputs = device.run(op, *inputs)
    device.log('Sending outputs %s!', format_arrays(outputs))
    channels.send(index, None, outputs)


def shard_inputs(index: MeshIndex, tensors: Sequence[np.ndarray],
                 shardings: Sequence[TensorSharding]) -> Sequence[np.ndarray]:
    shards = []
    for i, (tensor, tensor_sharding) in enumerate(zip(tensors, shardings)):
        assert tensor.ndim == len(tensor_sharding.dim_shards)
        slices = []
        for dim, dim_sharding in zip(tensor.shape, tensor_sharding.dim_shards):
            assert dim % dim_sharding.num_shards == 0
            shard_size = dim // dim_sharding.num_shards
            shard_index = (index.x %
                           dim_sharding.x_shard) * dim_sharding.y_shard + (
                               index.y % dim_sharding.y_shard)
            # When the mesh is not sharded along a dimension, duplicates
            # happens.
            start = shard_index * shard_size
            stop = start + shard_size
            slices.append(slice(start, stop))
        # YAPF crashed if using tensor[*slices]
        logging.info(f'Device: {index} get shared input {i}: '
                     f'[{format_slices(slices)}]')
        shards.append(tensor.__getitem__(tuple(slices)))
        # shards = tensor[*slices]
    return shards


# Virtual device clusters.
class VirtualCluster:
    def __init__(self, x_dim: int, y_dim: int):
        self._mesh = Mesh(x_dim=x_dim, y_dim=y_dim)
        self._channels = VirtualChannels(self._mesh)
        self._logger_queue = Queue()
        self._logger_queue_listener = init_queue_listener(self._logger_queue)
        self._devices = [
            VirtualDevice(index=index,
                          mesh=self._mesh,
                          channels=self._channels) for index in self._mesh
        ]

    def __del__(self):
        self._logger_queue_listener.stop()

    def run(self, op: Op, tensors: Sequence[np.ndarray],
            shardings: Sequence[TensorSharding]):

        processes = [
            Process(target=run_op_with_shared_inputs,
                    args=(op, d, self._channels, self._logger_queue))
            for d in self._devices
        ]

        for p in processes:
            p.start()

        start = time.time()
        # Send inputs
        for index in self._mesh:
            self._channels.send(None, index,
                                shard_inputs(index, tensors, shardings))

        # Receive outputs
        outputs = []
        for index in self._mesh:
            outputs.append(self._channels.receive(index, None))
        end = time.time()

        logging.info(f'End to end time: {end - start}')

        for p in processes:
            p.join()

        return outputs
