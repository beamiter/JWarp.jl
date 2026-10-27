"""Signed distance to a sphere: negative inside and positive outside."""
@inline sphere_sdf(point::Vec3{T}, center::Vec3{T}, radius::T) where {T} =
    norm3(point - center) - radius

"""Gradient of sphere SDF; chooses zero at the non-differentiable center."""
@inline sphere_sdf_gradient(point::Vec3{T}, center::Vec3{T}) where {T} =
    normalize3(point - center)

"""Signed distance to a plane with a unit normal and scalar plane offset."""
@inline plane_sdf(point::Vec3{T}, unit_normal::Vec3{T}, offset::T) where {T} =
    dot3(point, unit_normal) - offset

"""
Pairwise sphere contact. `normal` points from the center of A toward B.
`signed_distance < 0` represents penetration. For coincident centers,
the normal is deterministically chosen as the positive x-axis.
"""
struct SphereContact{T<:AbstractFloat}
    signed_distance::T
    normal::Vec3{T}
end

@inline function sphere_sphere_contact(
    center_a::Vec3{T}, radius_a::T,
    center_b::Vec3{T}, radius_b::T,
) where {T<:AbstractFloat}
    diff = center_b - center_a
    dist = norm3(diff)
    normal = dist > eps(T) ? diff / dist : Vec3{T}(one(T), zero(T), zero(T))
    return SphereContact{T}(dist - radius_a - radius_b, normal)
end

"""A ray-sphere result. `t` is +Inf on a miss; ray need not be normalized."""
struct RayHit{T<:AbstractFloat}
    hit::Bool
    t::T
    normal::Vec3{T}
end

@inline function _ray_miss(::Type{T}) where {T<:AbstractFloat}
    return RayHit{T}(false, T(Inf), zero(Vec3{T}))
end

"""
    ray_sphere_hit(origin, direction, center, radius)

Find the nearest non-negative ray parameter for a sphere intersection.
Uses a quadratic equation; supports non-unit direction vectors.
"""
@inline function ray_sphere_hit(
    origin::Vec3{T}, direction::Vec3{T},
    center::Vec3{T}, radius::T,
) where {T<:AbstractFloat}
    oc = origin - center
    a = dot3(direction, direction)
    a <= eps(T) && return _ray_miss(T)

    b = T(2) * dot3(oc, direction)
    c = dot3(oc, oc) - radius * radius
    discriminant = b * b - T(4) * a * c
    discriminant < zero(T) && return _ray_miss(T)

    root = sqrt(max(discriminant, zero(T)))
    inv_2a = inv(T(2) * a)
    t_near = (-b - root) * inv_2a
    t_far = (-b + root) * inv_2a
    t = t_near >= zero(T) ? t_near : t_far
    t < zero(T) && return _ray_miss(T)

    normal = normalize3(origin + direction * t - center)
    return RayHit{T}(true, t, normal)
end
