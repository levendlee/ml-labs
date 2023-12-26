"""Virtual compute cluster."""

import dataclasses
import logging
import time
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Process, Queue
from typing import Any, Callable, Optional, Sequence

import numpy as np

from parallelism.mesh import Mesh, MeshIndex
from parallelism.operation import Operation

Tensor = np.ndarray

# To determine the group id to run the collective ops on.
GroupIDFnType = Callable[[MeshIndex], Any]
ReduceFnType = Callable[[Tensor, Tensor], Tensor]
ShardFnType = Callable[[Tensor, MeshIndex], Tensor]


class Statistics:
    def __add__(self, other):
        cls = type(self)
        assert type(self) == type(other)
        lhs = dataclasses.asdict(self)
        rhs = dataclasses.asdict(other)
        return cls(**{k: lhs[k] + rhs[k] for k in lhs})


@dataclasses.dataclass
class ChannelStatistics(Statistics):
    h2d_times: int = 0
    h2d_bytes: int = 0
    d2d_times: int = 0
    d2d_bytes: int = 0


@dataclasses.dataclass
class DeviceStatistics(Statistics):
    flops: int = 0


# Virtual intra-device communication channels.
class VirtualChannels:
    def __init__(self, mesh: Mesh):
        num_devices = mesh.x_dim * mesh.y_dim
        self._mesh = mesh
        self._h2d_channels = [Queue() for _ in range(num_devices)]
        self._d2h_channels = [Queue() for _ in range(num_devices)]
        self._d2d_channels = [[Queue() for _ in range(num_devices)]
                              for _ in range(num_devices)]
        self._stats = ChannelStatistics()

    @property
    def stats(self) -> ChannelStatistics:
        return self._stats

    def to_1d(self, index: MeshIndex) -> int:
        return index.x * self._mesh.y_dim + index.y

    def get_channel(self, src: Optional[MeshIndex],
                    dst: Optional[MeshIndex]) -> Queue:
        if src is None:
            assert dst is not None
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
        data = self.get_channel(src, dst).get()
        data_bytes = np.sum([
            arr.size * arr.itemsize for arr in data if isinstance(arr, Tensor)
        ])
        if data_bytes:  # Excludes metadata communication
            if src is None and dst is not None:
                self._stats.h2d_times += 1
                self._stats.h2d_bytes += data_bytes
            elif src is not None and dst is not None:
                self._stats.d2d_times += 1
                self._stats.d2d_bytes += data_bytes
        return data


# Virtual devices.
class VirtualDevice:
    def __init__(self, mesh: Mesh, index: MeshIndex,
                 channels: VirtualChannels):
        self._mesh = mesh
        self._index = index
        self._channels = channels
        self._stats = DeviceStatistics()

    @property
    def index(self) -> MeshIndex:
        return self._index

    @property
    def channels(self) -> VirtualChannels:
        return self._channels

    @property
    def stats(self) -> DeviceStatistics:
        return self._stats

    def log(self, fmt, *args):
        prefix = f'{self}: '
        logging.info(prefix + fmt, *args)

    def __str__(self) -> str:
        return f'VirtualDevice({self._index})'

    def run(self, op: Operation):
        return op(device=self)

    def memcpy_h2d(self) -> Tensor:
        return self._channels.receive(src=None, dst=self._index)

    def memcpy_d2h(self, data: Tensor) -> None:
        self._channels.send(src=self._index, dst=None, data=data)

    def all_scatter(self, shard: Tensor, group_id_fn: GroupIDFnType) -> None:
        self.log(f'AllScatter request start.')

        # In practice, should do in parallel instead of using loops.
        src = self._index
        group_id = group_id_fn(self._index)
        for dst in self._mesh:
            if src == dst:
                continue
            if group_id != group_id_fn(dst):
                continue
            self.log('Scatter from %s to %s.', src, dst)
            self._channels.send(src=src, dst=dst, data=shard)

        self.log(f'AllScatter request finish.')

    # Scatter is used to implement gather, as the Queue works in a push mode
    # instead of a pull mode.
    def all_gather(self, shard: Tensor,
                   group_id_fn: GroupIDFnType) -> Sequence[Sequence[Tensor]]:
        self.log(f'AllGather request start.')

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
                self.log('Gather from %s to %s', src, dst)
                gathered.append(self._channels.receive(src=src, dst=dst))
        return gathered

    # Scatter is used to implement reduce, as the Queue works in a push mode
    # instead of a pull mode.
    def all_reduce(self, shard: Tensor, reduce_fn: ReduceFnType,
                   group_id_fn: GroupIDFnType) -> Tensor:
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

    def reduce_scatter(self, shard: Tensor, shard_fn: ShardFnType,
                       reduce_fn: ReduceFnType,
                       group_id_fn: GroupIDFnType) -> Tensor:
        self.log(f'ReduceScatter request start.')

        group_id = group_id_fn(self._index)

        src = self._index
        for dst in self._mesh:
            if src == dst:
                continue
            if group_id != group_id_fn(dst):
                continue
            self.log('Scatter from %s to %s.', src, dst)
            self._channels.send(src=src, dst=dst, data=shard_fn(shard, dst))

        dst = self._index
        reduced = shard_fn(shard, self._index)
        for src in self._mesh:
            if src == dst:
                continue
            if group_id != group_id_fn(src):
                continue
            self.log('Reduce from %s to %s.', src, dst)
            reduced = reduce_fn(reduced,
                                self._channels.receive(src=src, dst=dst))

        self.log(f'ReduceScatter request finish.')
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


def run_op(op, device, channels, logger_queue):
    init_queue_handler(logger_queue)
    device.run(op)


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

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self._mesh})'

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def run(
        self,
        op: Operation,
        *args,
        **kwargs,
    ):

        processes = [
            Process(target=run_op,
                    args=(op, d, self._channels, self._logger_queue))
            for d in self._devices
        ]

        for p in processes:
            p.start()

        start = time.time()
        # Send inputs
        for index in self._mesh:
            packets = op.preprocess_h2d(*args, **kwargs, index=index)
            logging.info(f'Sending %s packets to %s', len(packets), index)
            for p in packets:
                self._channels.send(None, index, p)

        # Receive outputs
        outputs = []
        device_stats = []
        channel_stats = []
        for index in self._mesh:
            num_packets = op.num_d2d_requests(index)
            logging.info(f'Getting %s packets from %s', num_packets, index)
            for i in range(op.num_d2d_requests(index)):
                logging.info(f'Got %s st packets from %s', i, index)
                outputs.append(self._channels.receive(index, None)[0])
            logging.info(f'Getting DeviceStatistics packets from %s', index)
            device_stats.append(self._channels.receive(index, None)[0])
            logging.info(f'Getting ChannelStatistics packets from %s', index)
            channel_stats.append(self._channels.receive(index, None)[0])
        end = time.time()

        logging.info(f'End to end time: {end - start}')

        for p in processes:
            p.join()

        logging.info(f'Compute: %s', sum(device_stats, DeviceStatistics()))
        logging.info(f'Network: %s', sum(channel_stats, ChannelStatistics()))

        return outputs
