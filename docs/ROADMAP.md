# JWarp roadmap — independent Julia implementation

## P0: executable architecture (this prototype)
- [x] Portable `@kernel` and `@index` surface via KernelAbstractions.jl.
- [x] `launch!` dispatching by array backend.
- [x] GPU-compatible `isbits` vector math.
- [x] CPU/GPU-shaped particle integration and batched sphere contact kernels.
- [x] Signed distance, sphere contact, analytic SDF gradient, ray/sphere query.
- [x] CPU tests and optional CUDA example.
- [x] Run Julia test suite (26 CPU assertions pass on Julia 1.12.6, 2026-10-27).
- [ ] Actually run CUDA parity tests on a supported NVIDIA GPU.

## P1: correctness, performance and core ergonomics
- CPU/GPU correctness and regression CI; AMDGPU and Metal smoke tests.
- Strict checks for array device IDs, aliasing and numerical dtype handling.
- Structure-of-arrays (SoA) layout in addition to AoS `Vec3`.
- Configurable stream, asynchronous launch and graph capture pathways.
- Defined errors / fallback behavior for coincident or degenerate geometry.
- GPU benchmarks against equivalent NVIDIA Warp kernels on identical hardware.

## P2: differentiable kernels
- Integrate Enzyme.jl, starting with side-effect-safe scalar kernels.
- Controlled reverse-mode through mutation and multiple kernel launches.
- Finite-difference gradient checks; contact non-smoothness conventions.
- Trajectory gradients, checkpointing and gradient memory analysis.

## P3: spatial / numerical primitives
- Triangle mesh closest-point and raycast; BVH, spatial hash grid.
- Quaternions, transforms, articulated dynamics and spatial Jacobians.
- BSR/CSR sparse operators, matrix-free CG, reductions and atomics.
- Tile programming where backend support is sufficiently mature.

## P4: robot / contact simulation tools (higher-level project)
- Batch FK, IK, retargeting and contact-pair assembly.
- Smooth vs hard-contact solver variants; explicit constraint audits.
- G1 motion test cases: feet, palm, knee, and seated contacts.
- Native dynamics solver interface with deterministic reproduction.

## Key principle
JWarp must remain a kernel / geometry / numerics foundation; a physics
engine comparable to Newton is a separate higher-level scope.

## Performance and reliability acceptance gates
1. CPU suite green and CUDA numerical parity within documented tolerances.
2. No known race conditions in GPU integration and contact workloads.
3. Enzyme gradient checks against numerical finite differences before AD claims.
4. Benchmark warm and cold paths separately; synchronize for GPU timings.
5. Record hardware / driver / Julia / backend versions for every result.
