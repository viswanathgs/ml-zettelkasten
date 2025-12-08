# JAX

- **Created**: 2025-12-02
- **Last Updated**: 2025-12-07
- **Status**: `In Progress`

---

- [X] Thinking in jax: <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>
- [X] The sharp bits: <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html>
- [ ] Jax 101: <https://docs.jax.dev/en/latest/jax-101.html>
  - [ ] `jax.jit`: <https://docs.jax.dev/en/latest/jit-compilation.html>
  - [ ] `jax.vmap`: <https://docs.jax.dev/en/latest/automatic-vectorization.html>
  - [ ] `jax.grad`: <https://docs.jax.dev/en/latest/automatic-differentiation.html>
  - [ ] pytrees: <https://docs.jax.dev/en/latest/pytrees.html>
  - [ ] `jax.random`: <https://docs.jax.dev/en/latest/random-numbers.html>
  - [ ] sharding and parallelism: <https://docs.jax.dev/en/latest/sharded-computation.html>
  - [ ] control flow: <https://docs.jax.dev/en/latest/control-flow.html>
  - [ ] tracing: <https://docs.jax.dev/en/latest/tracing.html>
  - [ ] stateful computations: <https://docs.jax.dev/en/latest/stateful-computations.html>
- [ ] Key concepts: <https://docs.jax.dev/en/latest/key-concepts.html>
- [ ] Advanced guides: <https://docs.jax.dev/en/latest/advanced_guides.html>
  - [ ] TODO
- [ ] <https://docs.jax.dev/en/latest/notes.html>
- [ ] pallas (kernels): <https://docs.jax.dev/en/latest/pallas/index.html>
  
---

## Notes

**Top-level concepts**:

1. `jax.numpy` (Interface): Runs NumPy-style code on GPUs/TPUs effortlessly.
2. `jax.jit` (Compilation): Fuses operations together for maximum speed (XLA).
3. `jax.grad` (Gradients): Transforms function $f$ $\to$ $f'$ (Functional Autodiff)
4. `jax.vmap` (Vectorization): Auto-converts single-item functions to process batches.
