using JWarp

n = 1_000_000
x = fill(Vec3f(0, 0, 0), n)
v = fill(Vec3f(1, 0, 0), n)
a = fill(Vec3f(0, -9.81f0, 0), n)

integrate_particles!(x, v, a, 0.01f0; sync = true)
@show x[1] v[1]
