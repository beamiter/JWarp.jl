# CPU-only smoke benchmark: compiles/warmups before timing (not comparable to
# NVIDIA Warp without matching algorithm, hardware, layout and sync semantics).
using JWarp

n = 1_000_000
steps = 100
x = fill(Vec3f(0, 0, 0), n)
v = fill(Vec3f(0, 0, 0), n)
a = fill(Vec3f(0, -9.81f0, 0), n)

integrate_particles!(x, v, a, 0.001f0; sync = true)
elapsed = @elapsed begin
    for _ in 1:steps
        integrate_particles!(x, v, a, 0.001f0; sync = true)
    end
end
println("CPU steps/sec: ", round(steps / elapsed; digits = 1))
println("CPU particle-updates/sec: ", round(Int, n * steps / elapsed))
