"""Tests on shared matrix multiplication."""

from typing import Sequence

import numpy as np
import pytest

from device import VirtualCluster
from matmul import create_matmul_op
from sharding import DimSharding, MatMulSharding, TensorSharding


def _run_test(tensor_shardings: Sequence[TensorSharding]):
    matmul_sharding = MatMulSharding(tensor_shardings)

    m, k, n = 4096, 2048, 4096
    input_tensors = [
        np.random.normal(size=shape).astype(np.float32)
        for shape in ((m, k), (k, n))
    ]
    output_tensor = np.matmul(input_tensors[0], input_tensors[1])

    # 2x4 mesh. 8 devices.
    cluster = VirtualCluster(2, 4)

    outputs = cluster.run(op=create_matmul_op(matmul_sharding),
                          tensors=input_tensors,
                          shardings=tensor_shardings[:2])

    assert len(outputs) == 8
    for tensor in outputs:
        np.testing.assert_allclose(tensor, output_tensor, atol=1e-4, rtol=1e-4)


def test_no_sharding():
    # Runs 3.77s on MacbookPro 2018 13inch i7
    # No sharding. 8x duplicated memory and compute.
    # 2023-12-23 16:23:07 [    INFO] Compute: DeviceStatistics(flops=549,755,813,888) (device.py:310)
    # 2023-12-23 16:23:07 [    INFO] Network: ChannelStatistics(h2d_times=8, h2d_bytes=536,870,912, d2d_times=0, d2d_bytes=0) (device.py:311)
    tensor_shardings = [
        TensorSharding([DimSharding(1, 1) for _ in range(2)]) for _ in range(3)
    ]
    _run_test(tensor_shardings)


@pytest.mark.parametrize(['x_shard', 'y_shard'], [(1, 4), (2, 1), (2, 4)])
def test_matched_inner_sharding(x_shard, y_shard):
    # 1x4: Runs 4.19s on MacbookPro 2018 13inch i7
    # 2023-12-23 16:23:15 [    INFO] Compute: DeviceStatistics(flops=137,438,953,472) (device.py:310)
    # 2023-12-23 16:23:15 [    INFO] Network: ChannelStatistics(h2d_times=8, h2d_bytes=134,217,728, d2d_times=24, d2d_bytes=1,610,612,736) (device.py:311)

    # 2x1: Runs 3.59s on MacbookPro 2018 13inch i7
    # 2023-12-23 16:23:22 [    INFO] Compute: DeviceStatistics(flops=274,877,906,944) (device.py:310)
    # 2023-12-23 16:23:22 [    INFO] Network: ChannelStatistics(h2d_times=8, h2d_bytes=268,435,456, d2d_times=8, d2d_bytes=536,870,912) (device.py:311)

    # 2x4: Runs 7.10s on MacbookPro 2018 13inch i7
    # 2023-12-23 16:23:36 [    INFO] Compute: DeviceStatistics(flops=68,719,476,736) (device.py:310)
    # 2023-12-23 16:23:36 [    INFO] Network: ChannelStatistics(h2d_times=8, h2d_bytes=67,108,864, d2d_times=56, d2d_bytes=3,758,096,384) (device.py:311)

    # Runs `AllReduce`.
    tensor_shardings = [
        TensorSharding([DimSharding(1, 1),
                        DimSharding(x_shard, y_shard)]),
        TensorSharding([DimSharding(x_shard, y_shard),
                        DimSharding(1, 1)]),
        TensorSharding([DimSharding(1, 1),
                        DimSharding(1, 1)]),
    ]
    _run_test(tensor_shardings)


@pytest.mark.parametrize(['x_shard', 'y_shard'], [(2, 4)])
def test_unmatched_inner_sharding(x_shard, y_shard):
    # 2x4: Runs 4.39s on MacbookPro 2018 13inch i7
    # 2023-12-23 16:23:43 [    INFO] Compute: DeviceStatistics(flops=549,755,813,888) (device.py:310)
    # 2023-12-23 16:23:43 [    INFO] Network: ChannelStatistics(h2d_times=8, h2d_bytes=201,326,592, d2d_times=32, d2d_bytes=335,544,320) (device.py:311)

    # Runs `AllGather`.
    tensor_shardings = [
        TensorSharding([DimSharding(1, 1),
                        DimSharding(1, y_shard)]),
        TensorSharding([DimSharding(x_shard, 1),
                        DimSharding(1, 1)]),
        TensorSharding([DimSharding(1, 1),
                        DimSharding(1, 1)]),
    ]
    _run_test(tensor_shardings)

@pytest.mark.parametrize(['x_shard', 'y_shard'], [(2, 4)])
def test_outer_sharding(x_shard, y_shard):
    # A: (sharded_X, sharded_Y)
    # B: (sharded_X, full)
    # 2x4: Runs 4.39s on MacbookPro 2018 13inch i7

    # Runs `AllGather`.
    tensor_shardings = [
        TensorSharding([DimSharding(x_shard, 1),
                        DimSharding(1, y_shard)]),
        TensorSharding([DimSharding(x_shard, 1),
                        DimSharding(1, 1)]),
        TensorSharding([DimSharding(1, 1),
                        DimSharding(1, 1)]),
    ]
    _run_test(tensor_shardings)

@pytest.mark.parametrize(['x_shard', 'y_shard'], [(2, 4)])
def test_full_sharding(x_shard, y_shard):
    # A: (sharded_X, sharded_Y)
    # B: (sharded_X, sharded_Y)
    # 2x4: Runs 4.39s on MacbookPro 2018 13inch i7

    # Runs `AllGather`.
    tensor_shardings = [
        TensorSharding([DimSharding(x_shard, 1),
                        DimSharding(1, y_shard)]),
        TensorSharding([DimSharding(x_shard, 1),
                        DimSharding(1, y_shard)]),
        TensorSharding([DimSharding(1, 1),
                        DimSharding(1, 1)]),
    ]
    _run_test(tensor_shardings)