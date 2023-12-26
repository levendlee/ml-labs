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
            nn.Dense(
                features=f,
                use_bias=True,
                dtype=jnp.float32,
                #kernel_init=nn.initializers.truncated_normal(1.0),
                #bias_init=nn.initializers.truncated_normal(1.0)
            ) for f in self.units
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

    learning_rate = 0.001
    operation = MLP(pipeline=pipeline, learning_rate=learning_rate)
    logging.info(f'{operation}')

    num_datasets = 4096
    units = [256, 512, 1024, 512, 256]
    num_layers = len(units) - 1

    rng = jax.random.PRNGKey(2023)
    inputs = jax.random.truncated_normal(rng,
                                         lower=-2.0,
                                         upper=2.0,
                                         shape=[num_datasets, units[0]],
                                         dtype=np.float32)
    targets = jax.random.truncated_normal(rng,
                                          lower=-2.0,
                                          upper=2.0,
                                          shape=[num_datasets, units[0]],
                                          dtype=np.float32)
    flax_model = Model(units[1:])
    variables = flax_model.init(rng, inputs, targets)
    bounded_flax_model = flax_model.bind(variables)
    assert len(bounded_flax_model.layers) == num_layers

    # print(jax.tree_map(lambda x: x.shape, variables))
    # logging.info('%s', jax.tree_map(lambda x: (np.max(x), np.min(x)),
    #                                 variables))

    def read_params(variables):
        return [[
            variables['params'][f'layers_{i}']['kernel'],
            variables['params'][f'layers_{i}']['bias']
        ] for i in range(num_layers)]

    parameters = read_params(variables)
    outputs = cluster.run(op=operation,
                          activations=(inputs, ),
                          parameters=parameters,
                          targets=(targets, ))

    @jax.jit
    def forward(variables, inputs, targets):
        return flax_model.apply(variables, inputs, targets)

    backward = jax.jit(jax.grad(forward, argnums=(0, 1)))

    batch_size = num_datasets // pipeline.num_runs
    for i in range(pipeline.num_runs):
        batch_inputs = inputs[i * batch_size:i * batch_size + batch_size]
        batch_targets = targets[i * batch_size:i * batch_size + batch_size]
        loss = forward(variables, batch_inputs, batch_targets)
        logging.info('Batch %d. Report loss=%f.', i, loss)
        grad_var, grad_x = backward(variables, batch_inputs, batch_targets)
        # logging.info('%s',
        #              jax.tree_map(lambda x: (np.max(x), np.min(x)), grad_var))
        np.testing.assert_allclose(outputs[i], grad_x, atol=1e-5, rtol=1e-5)
        variables = jax.tree_map(lambda var, grad: var - learning_rate * grad,
                                 variables, grad_var)
