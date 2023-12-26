"""Tests on pipelined MLP layer."""

import logging
import os
from typing import Sequence

os.environ['JAX_PLATFORMS'] = 'cpu'

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from parallelism.cluster import VirtualCluster
from parallelism.pipeline_parallelism.mlp import MLP
from parallelism.pipelining import Pipeline

Tensor = np.ndarray


class Model(nn.Module):
    units: Sequence[int]

    def setup(self) -> None:
        self.layers = [
            nn.Dense(features=f, use_bias=True, dtype=jnp.float32)
            for f in self.units
        ]

    def __call__(self, x: jax.Array, target: jax.Array) -> float:
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
        return np.sum((x - target)**2) / x.size


@pytest.mark.parametrize(['x_dim', 'y_dim'], [(2, 2)])
def test_pipelined_mlp(x_dim, y_dim):
    cluster = VirtualCluster(x_dim, y_dim)
    logging.info(f'{cluster}')

    pipeline = Pipeline(mesh=cluster.mesh, num_runs=8, num_stages=4)
    logging.info(f'{pipeline}')

    operation = MLP(pipeline=pipeline, learning_rate=0.001)
    logging.info(f'{operation}')

    batch = 4096
    units = [256, 512, 1024, 512, 256]
    num_layers = len(units) - 1

    np.random.seed(2023)
    inputs = np.random.normal(size=[batch, units[0]]).astype(np.float32)
    targets = np.random.normal(size=[batch, units[0]]).astype(np.float32)

    rng = jax.random.PRNGKey(2023)
    flax_model = Model(units[1:])
    variables = flax_model.init(rng, inputs, targets)
    # print(jax.tree_map(lambda x: x.shape, variables))

    parameters = [[
        variables['params'][f'layers_{i}']['kernel'],
        variables['params'][f'layers_{i}']['bias']
    ] for i in range(num_layers)]
    outputs = cluster.run(op=operation,
                          activations=(inputs, ),
                          parameters=parameters,
                          targets=(targets, ))
