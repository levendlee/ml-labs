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
    tensor_shardings = [
        TensorSharding([DimSharding(1, 1) for _ in range(2)]) for _ in range(3)
    ]
    _run_test(tensor_shardings)


@pytest.mark.parametrize(['x_shard', 'y_shard'], [(1, 4), (2, 1), (2, 4)])
def test_matched_inner_sharding(x_shard, y_shard):
    # 1x4: Runs 4.19s on MacbookPro 2018 13inch i7
    # 2x1: Runs 3.59s on MacbookPro 2018 13inch i7
    # 2x4: Runs 7.10s on MacbookPro 2018 13inch i7

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
