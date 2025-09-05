# [The Ultra-Scale Playbook: Training LLMs on GPU Clusters, HuggingFace](https://huggingface.co/spaces/nanotron/ultrascale-playbook)

- **Created**: 2025-02-08
- **Last Updated**: 2025-02-08
- **Status**: `Paused`

---

## Training on 1 GPU

- Assumes model params, grad-buffer, optimizer states can fit in single GPU.
- Peak memory usage due to activations memory.
- Reduce peak activations memory via
  - (1) activation checkpointing/recomputation
  - (2) gradient accumulation. global_batch_size = micro_batch_size x grad_acc_steps

## Collective Ops

- Broadcast
- Reduce, AllReduce
- Gather, AllGather
- Scatter, ReduceScatter
- Ring AllReduce: ReduceScatter (N-1 ring steps) followed by AllGather (N-1 ring steps)

## Data Parallelism (DP)

- global_batch_size = micro_batch_size x grad_acc_steps x num_gpus
- Standard DP: entire model params, gradients and optimizer states replicated on all GPUs
  - Forward -> Backward -> AllReduce(grads) -> Update
- DeepSpeed ZeRO (Zero Redundancy Optimizer): further memory optimization by sharding redundant memory
  - (1) ZeRO-1: optimizer states sharded across DP ranks
    - Forward -> Backward -> ReduceScatter(grads) -> Update -> AllGather(params)
    - The final AllGather over params can be overlapped with the next forward pass.
  - (2) ZeRO-2: optimizer states and gradients sharded across DP ranks
    - Same as ZeRO-1, except that grads are also sharded right after ReduceScatter.
    - Kinda weird why anyone would use ZeRO-1. This is a natural optimization, just cleaning up unwated grads corresponding to local batch after ReduceScatter.
  - (3) ZeRO-3 / FSDP: optimzier states, gradients and params sharded across DP ranks
    - In addition to ZeRO-2, also shard params across DP ranks.
    - Also called FSDP (Fully Sharded Data Parallel).
    - Params of each layer is sharded over DP ranks and materialized on the fly as needed via AllGather.
    - Reduces memory over ZeRO-2 at the cost of added 1 AllGather per layer per forward and per backward. For $L$ layers, $2 \times L - 1$ additional AllGather ops per training step (it's not $2 \times L$ as the last layer needs only one AllGather).
    - Forward computation for layer $L$ overlapped with AllGather for layer $L + 1$. Backward computation for layer $L$ overlapped with AllGather for layer $L - 1$.
    - Aside: Toy implementation in <https://github.com/viswanathgs/LLM-Training-Puzzles> [[srush-ml-puzzles.md]].

## Tensor Parallelism (TP)

- So far: activation checkpointing + gradient accumulation + FSDP. Limitations:
  - What if even 1 layer can't fit in a single GPU?
  - Params, grads and optimizer states are sharded with FSDP, but activation memory can still be very high for large sequence lengths.
- Tensor Parallelism: Shard a tensor along a particular dimension.
  - Column-wise sharding (column-linear) and row-wise sharding (row-linear). Different communication primitives based on sharding type.
  - Example: $Y = X \times W$; $X$ = input tensor/activations, $W$ = weight matrix
  - Column-wise sharding (column-linear): $W_{col=i}$ - $W$ sharded among GPUs along column dim
    - 1. $X$ = Broadcast $X$ to all GPUs
    - 2. Parallel $Y_{col_i}$ = $X \times W_{col=i}$
    - 3. $Y$ = AllGather $Y_{col=i}$
  - Row-wise sharding (row-linear): $W_{row=i}$ - $W$ sharded among GPUs along row dim
    - 1. $X_{col=i}$ = Scatter $X$ to across GPUs along column dim
    - 2. Parallel $Y_{gpu=i}$ = $X_{col=i} \times W_{row=i}$
    - 3. $Y$ = AllReduce $Y_{gpu=i}$
- Tensor Parallelism in a transformer:
  - Feedforward block: 2 linear layers
    - Column-linear (without the final allgather) followed by row-linear (without needing the first step to scatter along column dim)
    - Goal: $Y1 = X \times W1$, $Y2 = Y1 \times W2$
    - Steps: $W1_{col=i}$ - $W1$ sharded along column dim, $W2_{row=i}$ - $W2$ sharded along row dim.
      - 1. $X$ = Broadcast $X$ to all GPUs
      - 2. Parallel $Y1_{col=i}$ = $X \times W1_{col=i}$
      - 3. Parallel $Y2_{gpu=i}$ = $Y1_{col=i} \times W2_{row=i}$
      - 4. $Y2$ = AllReduce $Y2_{gpu=i}$
  - Multihead attention block:
    - TODO
  - TODO

## Context Parallelism (CP)

- TODO

## Pipeline Parallelism (PP)

- Aside: Toy implementation in <https://github.com/viswanathgs/LLM-Training-Puzzles> [[srush-ml-puzzles.md]].
- Layers split among a few GPUs, sequential dataflow.
- Pro: Compared to Tensor Parallelism (TP), communication only at certain layer junctions. TP needs communications for each layer.
- Downside: Lots of GPU idle time due to "bubble" when other layers are executing.
- Size of the bubble / idle time can be reduced by microbatching (gradient accumulation) and pipelining execution across GPUs.
- Bubble size = (p - 1) / m. p = degree of pipeline parallelism, m = microbatch size.
- Methods:
  - (1) AFAB (All Forward All Backward):
    - Simplest approach, forward for all microbatches followed by backward.
    - Bubble time reduced by a factor of microbatch size $m$. Still need to retain activations in memory for all microbatches until backward is done.
  - (2) 1F1B (1 Forward 1 Backward):
    - Alternative forward and backward to clear out activations as soon as possible.
    - Need to store activations only for $p$ samples (where $p$ is the degree of pipeline paralleism) instead of $m$ (where $m$ is the total number of samples in the microbatch).
  - (3) Interleaving Stages:
    - Reduces bubble size (as opposed to just activation memory requirement)
    - Used in llama 3.1
  - (4) ZeroBubble and DualPipe
    - Close to a "zero bubble" regime
    - Used in DeepSeek V3
    - TODO

## Expert Parallelism (EP)

- TODO

## 5D Parallelism

- TODO

## Finding the best training configuration

- TODO
