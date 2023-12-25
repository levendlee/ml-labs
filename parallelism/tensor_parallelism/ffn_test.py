"""Tests on sharded feed-forward layer."""

from typing import Sequence

import numpy as np
import pytest

from model_parallelism.cluster import VirtualCluster
from model_parallelism.sharding import DimSharding, TensorSharding
from model_parallelism.tensor_parallelism.ffn import (Feedforward,
                                                      FeedforwardSharding)
from model_parallelism.utils import *

np.random.seed(2023)


def _run_test(tensor_shardings: Sequence[TensorSharding],
              atol=1e-6,
              rtol=1e-6):
    (x1_sharding, w1_sharding, b1_sharding, x2_sharding, w2_sharding,
     b2_sharding, o_sharding) = tensor_shardings
    ffn_sharding = FeedforwardSharding(tensor_shardings)

    b, s, m, h = 8, 128, 1024, 4096
    # x1, w1, b1, w2, b2
    input_tensors = [
        np.random.normal(size=shape).astype(np.float32)
        for shape in ((b, s, m), (m, h), (h, ), (h, m), (m, ))
    ]
    x1, w1, b1, w2, b2 = input_tensors

    o = np.einsum('abc,cd->abd', x1, w1)
    o += b1
    o = np.maximum(o, 0.0)
    o = np.einsum('abc,cd->abd', o, w2)
    o += b2

    output_tensor = o

    # 2x4 mesh. 8 devices.
    cluster = VirtualCluster(2, 4)

    outputs = cluster.run(op=Feedforward(ffn_sharding),
                          tensors=input_tensors,
                          shardings=[
                              x1_sharding, w1_sharding, b1_sharding,
                              w2_sharding, b2_sharding
                          ])

    assert len(outputs) == 8
    for index, tensor in zip(cluster.mesh, outputs):
        expected_tensor = shard_tensor(tensor_sharding=o_sharding,
                                       tensor=output_tensor,
                                       index=index)
        np.testing.assert_allclose(tensor,
                                   expected_tensor,
                                   atol=atol,
                                   rtol=rtol)


def test_no_sharding():
    # 2023-12-23 22:58:24 [    INFO] End to end time: 10.123413801193237 (device.py:334)
    # 2023-12-23 22:58:24 [    INFO] Compute: DeviceStatistics(flops=137,438,953,472) (device.py:339)
    # 2023-12-23 22:58:24 [    INFO] Network: ChannelStatistics(h2d_times=8, h2d_bytes=302,153,728, d2d_times=0, d2d_bytes=0) (device.py:340)
    ds = DimSharding(1, 1)
    ts_1d = TensorSharding([ds])
    ts_2d = TensorSharding([ds, ds])
    ts_3d = TensorSharding([ds, ds, ds])
    tensor_shardings = [
        # x1, w1, b1, x2, w2, b2, o
        ts_3d,
        ts_2d,
        ts_1d,
        ts_3d,
        ts_2d,
        ts_1d,
        ts_3d
    ]
    _run_test(tensor_shardings)


@pytest.mark.parametrize(['x_shard', 'y_shard'], [(2, 4)])
def test_gspmd_sharding(x_shard, y_shard):
    # 2023-12-23 23:17:50 [    INFO] End to end time: 1.7179160118103027 (device.py:335)
    # 2023-12-23 23:17:50 [    INFO] Compute: DeviceStatistics(flops=17,179,869,184) (device.py:340)
    # 2023-12-23 23:17:50 [    INFO] Network: ChannelStatistics(h2d_times=8, h2d_bytes=37,789,696, d2d_times=64, d2d_bytes=58,720,256) (device.py:341)
    dfull = DimSharding(1, 1)
    dx = DimSharding(x_shard, 1)
    dy = DimSharding(1, y_shard)

    x1_sharding = TensorSharding([dx, dfull, dy])
    w1_sharding = TensorSharding([dx, dy])
    b1_sharding = TensorSharding([dy])
    x2_sharding = TensorSharding([dx, dfull, dy])
    w2_sharding = TensorSharding([dy, dx])
    b2_sharding = TensorSharding([dy])
    o_sharding = x1_sharding

    tensor_shardings = [
        x1_sharding, w1_sharding, b1_sharding, x2_sharding, w2_sharding,
        b2_sharding, o_sharding
    ]
    # Such a huge numerics difference with large reduction.
    _run_test(tensor_shardings, atol=1e-2, rtol=1e-4)
