# JWarp.jl (v0.1.0 proof-of-concept)

**An independent Julia-native, NVIDIA Warp-inspired kernel + spatial computing prototype.**

This is **not** NVIDIA Warp, not a port of its source, not an implementation of
all Warp APIs, and not affiliated with NVIDIA. This starter library is designed
as a foundation for future differentiable geometry, contact, and robotics work.

## What works in the source code

- Warp-style GPU/CPU kernel authoring with `@kernel`, `@index` and `launch!`
  using the existing Julia package `KernelAbstractions.jl`.
- `Vec3{T}` / `Vec3f` immutable, GPU-friendly vector arithmetic.
- One-kernel batched symplectic-Euler particle stepping.
- One-kernel batched sphere/sphere signed distance and normals.
- Geometric primitives: sphere/plane SDF, analytic sphere SDF gradient,
  ray/sphere intersection.
- CPU-focused tests; optional CUDA example.

**Execution status (2026-10-27):** Validated on Linux with Julia 1.12.6 and
KernelAbstractions 0.9.42. `Pkg.test()` passes all 26 CPU assertions; both CPU
examples and the CPU smoke benchmark run successfully. All nine Julia files
parse, and `Test.detect_ambiguities(JWarp; recursive=true)` reports no method
ambiguities. Julia 1.10 compatibility has not yet been tested. GPU kernel
compilation and CUDA execution remain unvalidated because the validation host
has no working NVIDIA driver. This is still an early proof of concept.

## Quick start

Install Julia 1.10+ on your system, then clone and test the package:

```bash
git clone https://github.com/beamiter/JWarp.jl.git
cd JWarp.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. examples/particles_cpu.jl
julia --project=. examples/contact_cpu.jl
```

To install directly into another Julia environment (the package is not yet
registered in General):

```julia
using Pkg
Pkg.add(url="https://github.com/beamiter/JWarp.jl", rev="master")
```

GPU example (NVIDIA with compatible drivers):

```bash
julia --project=. -e 'using Pkg; Pkg.add("CUDA")'
julia --project=. examples/particles_cuda.jl
```

This package does *not* require the CUDA dependency for CPU use. Backend selection
is based on the array type; supported GPU backends require separate installation
and compatible system drivers. This package has not been tested on GPU.

## Julia API sample

```julia
using JWarp

@kernel function scale_kernel!(output, input, factor)
    i = @index(Global, Linear)
    @inbounds output[i] = factor * input[i]
end

input = Float32[1, 2, 3, 4]
output = similar(input)
launch!(scale_kernel!, output, input, 2f0; sync=true)
@assert output == Float32[2, 4, 6, 8]

x = fill(Vec3f(0, 0, 0), 100_000)
v = fill(Vec3f(1, 0, 0), 100_000)
a = fill(Vec3f(0, -9.81f0, 0), 100_000)
integrate_particles!(x, v, a, 0.01f0; sync=true)
```

## Important limitations

- **No general automatic differentiation / tape / backward kernels yet.**
  `sphere_sdf_gradient` is an analytic special case, not a general AD system.
- No BVH, triangle mesh, hash grid, sparse linear algebra, FEM, physics engine,
  G1 solver, automatic contact Jacobians, tile programming or CUDA graph capture.
- `Vec3` array layout is AoS, not yet optimized against a SoA representation.
- `launch!` checks backend *types*, not the exact device ID of every array;
  use arrays on one device and do not mix CPU/device arrays.
- Geometry degenerate-case conventions are deliberately simple.
- `@kernel` is a re-export of KernelAbstractions, *not* a new compiler.

## Scope

The long-term goal is to capture Warp's **programming experience** in Julia,
not to reproduce Warp's implementation. NVIDIA's Newton is the higher-level
simulation engine built on Warp; physics-engine functionality is a separate
project layer. See `docs/ROADMAP.md` for planned milestones and validation gates.

## Upstream dependencies / design references

- https://nvidia.github.io/warp/
- https://juliagpu.github.io/KernelAbstractions.jl/
- https://cuda.juliagpu.org/stable/
- https://enzymead.github.io/Enzyme.jl/stable/
- https://newton-physics.github.io/newton/

## License

MIT license for this original proof-of-concept source.
