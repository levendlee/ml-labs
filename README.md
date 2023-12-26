# Implements some high level ideas from papers.

Not really functional. Can only emulate solutions proposed for large models due
to resource limitation.

## `parallelism/`

Virtual implementation of distributed training/serving parallelism.
Virtual hardware are implemented as:
- `cluster.py`
  - `VirtualDevice`: Emulates a compute device/machine using a
    `multiprocessing.Process`. Have all collective ops (e.g. `AllScatter`,
    `AllGather`, `AllReduce`, `ReduceScatter`, etc.) as builtin.
  - `VirtualChannel`: Emulates intra-device and host2device, device2host
    communication through `multiprocessing.Queue`.
  - `VirtualCluster`: Emulates a compute cluster through a pool of
    `multiprocessing.Process`s and `multiprocessing.Queue`s.
- `mesh.py`
  - `Mesh`: Hardware topology configuration.
  - `MeshIndex`: Hardware topology index.
- `operation.py`
  - `Operation`: An operation to be executed by the compute cluster.
  - `ShardedOperation`: An operation to run w/ tensor parallelism.
  - `PipelinedOperation`: An operation to run w/ pipeline parallelism.
- `sharding.py`
  - `DimSharding`: Sharding configuration along a dimension.
  - `TensorSharding`: Sharding configuration along a tensor.
- `pipelining.py`
  - `Pipeline`: A pipeline configuration. Mostly including total number of
    concurrent batches, stages, and hardware mesh mapping.
  - `PipelineStage`: A pipeline stage configuration and status. Mostly including
    the status of which cycle this stage is running, and the associated helper 
    functions to decide if it is in `forward`, `backward`, or `idle` status.

### `tensor_pallelsim/`

Intra-op model parallelism through sharding tensors and computation.
- `matmul.py`
  - `MatMulSharding`: Tensor parallelism sharding configuration.
  - `MatMulShardingPolicy`: Tensor parallelism sharding policies for function
    dispatch.
    - `Unsharded`: No sharding. Duplicated runs.
    - `MatchedInnerSharding`: Matched inner dimension sharding. Run with
      `AllReduce`.
    - `UnmatchedInnerSharding`: Unmatched inner dimension sharding. Run with
      `AllGather`.
    - `OuterSharding`: Outer (`m`) and inner (`k`) dimension sharding. Run with
      `AllGather`->`AllGather`.
    - `FullSharding`:  All uter (`m`, `n`) and inner (`k`) dimension sharding.
      Run with `AllGather`->`AllGather`.
  - `MatMul`: Tensor parallelism based operation. Implementation is dispatched
    based on `MatMulShardingPolicy`.
- `ffn.py`
  - `FeedForwardSharding`: Tensor parallelism sharding configuration.
  - `FeedForwardShardingPolicy`: Tensor parallelism sharding policies for
    function dispatch.
    - `Unsharded`: No sharding. Duplicated runs.
    - `GSPMD`: https://arxiv.org/abs/2105.04663. Run with `AllGather`->
      `AllGather`->`ReduceScatter`.
  - `FeedForward`: Tensor parallelism based operation. Implementation is
    dispatched  based on `FeedForwardShardingPolicy`.
  
### `pipeline_paralleism/`

Inter-op model parallelism through run different stages of the model
asynchronously.

- `mlp.py`
  - `MLP`: A simple MLP model with 4 dense layers running as different stages
    of pipeline. Both forward and backward paths are managed. Activations,
    parameters, gradients, and states are managed but optimizers are not.
