using JWarp

n = 100_000
centers_a = fill(Vec3f(0, 0, 0), n)
centers_b = fill(Vec3f(1.75, 0, 0), n)
radii_a = fill(1f0, n)
radii_b = fill(1f0, n)
signed_distances = zeros(Float32, n)
normals = fill(Vec3f(0, 0, 0), n)

batch_sphere_contacts!(signed_distances, normals,
                       centers_a, radii_a, centers_b, radii_b; sync = true)
@show signed_distances[1] normals[1]
