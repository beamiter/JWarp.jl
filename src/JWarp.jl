module JWarp

using KernelAbstractions: @kernel, @index, get_backend, synchronize

# Re-export an already maintained, cross-backend kernel programming model.
export @kernel, @index

export Vec3, Vec3f, dot3, cross3, norm3, normalize3
export sphere_sdf, sphere_sdf_gradient, plane_sdf
export SphereContact, sphere_sphere_contact, RayHit, ray_sphere_hit
export launch!, integrate_particles!, batch_sphere_contacts!

include("math.jl")
include("geometry.jl")
include("kernels.jl")

end # module JWarp
