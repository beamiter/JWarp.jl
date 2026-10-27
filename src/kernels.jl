"""
    launch!(kernel, output, args...; ndrange=length(output), workgroupsize=256, sync=false)

Launch a `KernelAbstractions.@kernel` on the backend inferred from `output`.
Julia arrays use CPU; `CuArray` and other device arrays select their backends.
All array arguments must reside on the same device/backend; callers are
responsible for placement and transfers. `sync=true` waits for completion.
"""
function launch!(
    kernel, output::AbstractArray, args...;
    ndrange = length(output), workgroupsize::Integer = 256, sync::Bool = false,
)
    workgroupsize > 0 || throw(ArgumentError("workgroupsize must be positive"))
    backend = get_backend(output)

    # Reject obvious CPU-vs-GPU / cross-vendor mistakes before launching.
    # Device-index validation for multiple GPUs on the same backend remains TODO.
    for arg in args
        if arg isa AbstractArray && typeof(get_backend(arg)) != typeof(backend)
            throw(ArgumentError("array arguments must use the same backend"))
        end
    end

    kernel(backend, Int(workgroupsize))(output, args...; ndrange = ndrange)
    sync && synchronize(backend)
    return output
end

# Symplectic (semi-implicit) Euler: v[t+dt]=v[t]+a*dt; x[t+dt]=x[t]+v[t+dt]*dt.
@kernel function _integrate_particles_kernel!(positions, velocities, accelerations, dt)
    i = @index(Global, Linear)
    @inbounds begin
        v_next = velocities[i] + accelerations[i] * dt
        velocities[i] = v_next
        positions[i] = positions[i] + v_next * dt
    end
end

"""
    integrate_particles!(positions, velocities, accelerations, dt; ...)

Apply one semi-implicit Euler step to matching arrays of `Vec3{T}`.
Mutates `positions` and `velocities` in place; returns both arrays.
"""
function integrate_particles!(
    positions::AbstractVector{Vec3{T}},
    velocities::AbstractVector{Vec3{T}},
    accelerations::AbstractVector{Vec3{T}},
    dt::Real; workgroupsize::Integer = 256, sync::Bool = false,
) where {T<:AbstractFloat}
    n = length(positions)
    length(velocities) == n && length(accelerations) == n ||
        throw(DimensionMismatch("position, velocity and acceleration counts differ"))
    isfinite(dt) || throw(ArgumentError("dt must be finite"))
    n == 0 && return (positions, velocities)

    launch!(_integrate_particles_kernel!, positions, velocities, accelerations, T(dt);
            ndrange = n, workgroupsize = workgroupsize, sync = sync)
    return positions, velocities
end

@kernel function _batch_sphere_contacts_kernel!(
    signed_distances, normals, centers_a, radii_a, centers_b, radii_b,
)
    i = @index(Global, Linear)
    @inbounds begin
        contact = sphere_sphere_contact(
            centers_a[i], radii_a[i], centers_b[i], radii_b[i]
        )
        signed_distances[i] = contact.signed_distance
        normals[i] = contact.normal
    end
end

"""
    batch_sphere_contacts!(signed_distances, normals,
                           centers_a, radii_a, centers_b, radii_b; ...)

Compute batched sphere-pair separations and normals in one parallel kernel.
A negative separation means overlapping spheres.
"""
function batch_sphere_contacts!(
    signed_distances::AbstractVector{T},
    normals::AbstractVector{Vec3{T}},
    centers_a::AbstractVector{Vec3{T}},
    radii_a::AbstractVector{T},
    centers_b::AbstractVector{Vec3{T}},
    radii_b::AbstractVector{T};
    workgroupsize::Integer = 256, sync::Bool = false,
) where {T<:AbstractFloat}
    n = length(signed_distances)
    all(length(a) == n for a in (normals, centers_a, radii_a, centers_b, radii_b)) ||
        throw(DimensionMismatch("all contact arrays must have the same length"))
    n == 0 && return signed_distances, normals

    launch!(_batch_sphere_contacts_kernel!, signed_distances,
            normals, centers_a, radii_a, centers_b, radii_b;
            ndrange = n, workgroupsize = workgroupsize, sync = sync)
    return signed_distances, normals
end
