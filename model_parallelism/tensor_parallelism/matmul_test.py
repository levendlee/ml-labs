"""Tests on shared matrix multiplication."""

import unittest
from typing import Sequence

from device import VirtualCluster
from matmul import create_matmul_op
from sharding import DimSharding, TensorSharding, MatMulSharding

import numpy as np


class SharedMatMulTest(unittest.TestCase):

    def _run_test(self, tensor_shardings: Sequence[TensorSharding]):
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
                              tensor_and_sharding=tuple(
                                  zip(input_tensors, tensor_shardings[:2])))

        self.assertEqual(len(outputs), 8)
        for tensor in outputs:
            np.testing.assert_allclose(tensor, output_tensor)

    def test_no_sharing(self):
        # Runs 4.61s on MacbookPro 2018 13inch i7
        # No sharding. 8x duplicated memory and compute.
        tensor_shardings = [
            TensorSharding([DimSharding(1, 1) for _ in range(2)])
            for _ in range(3)
        ]
        self._run_test(tensor_shardings)

    def test_matched_inner_sharing(self):
        # Runs ?s on MacbookPro 2018 13inch i7

        # Matched inner sharding on y dimension.
        # 2x duplicated memory and compute.
        # Runs `AllReduce`.
        tensor_shardings = [
            TensorSharding([DimSharding(1, 1), DimSharding(1, 4)]),
            TensorSharding([DimSharding(1, 4), DimSharding(1, 1)]),
            TensorSharding([DimSharding(1, 1), DimSharding(1, 1)]),
        ]
        self._run_test(tensor_shardings)
