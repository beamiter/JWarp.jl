# Install CUDA into the local Julia environment first: ] add CUDA
using CUDA
using JWarp

CUDA.functional() || error("No functional CUDA backend")
n = 1_000_000
x = CuArray(fill(Vec3f(0, 0, 0), n))
v = CuArray(fill(Vec3f(1, 0, 0), n))
a = CuArray(fill(Vec3f(0, -9.81f0, 0), n))

integrate_particles!(x, v, a, 0.01f0; sync = true)
@show Array(x)[1] Array(v)[1]
