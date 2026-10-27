"""
    Vec3{T}(x, y, z)

A compact, immutable, `isbits` 3D vector usable in Julia CPU and GPU kernels.
For high-throughput memory-bound workloads, consider a structure-of-arrays
layout as an alternative to arrays of `Vec3`.
"""
struct Vec3{T<:AbstractFloat}
    x::T
    y::T
    z::T
end

const Vec3f = Vec3{Float32}

function Vec3(x::Real, y::Real, z::Real)
    xx, yy, zz = promote(float(x), float(y), float(z))
    return Vec3{typeof(xx)}(xx, yy, zz)
end

@inline Base.zero(::Type{Vec3{T}}) where {T} = Vec3{T}(zero(T), zero(T), zero(T))
@inline Base.zero(v::Vec3{T}) where {T} = zero(Vec3{T})

@inline Base.:+(a::Vec3{T}, b::Vec3{T}) where {T} =
    Vec3{T}(a.x + b.x, a.y + b.y, a.z + b.z)

@inline Base.:-(a::Vec3{T}, b::Vec3{T}) where {T} =
    Vec3{T}(a.x - b.x, a.y - b.y, a.z - b.z)

@inline Base.:-(a::Vec3{T}) where {T} = Vec3{T}(-a.x, -a.y, -a.z)

@inline Base.:*(a::Vec3{T}, s::Real) where {T} =
    Vec3{T}(a.x * T(s), a.y * T(s), a.z * T(s))

@inline Base.:*(s::Real, a::Vec3{T}) where {T} = a * s
@inline Base.:/(a::Vec3{T}, s::Real) where {T} = a * inv(T(s))

@inline Base.:(==)(a::Vec3{T}, b::Vec3{T}) where {T} =
    a.x == b.x && a.y == b.y && a.z == b.z

@inline dot3(a::Vec3{T}, b::Vec3{T}) where {T} =
    a.x * b.x + a.y * b.y + a.z * b.z

@inline cross3(a::Vec3{T}, b::Vec3{T}) where {T} = Vec3{T}(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x,
)

@inline norm3(a::Vec3) = sqrt(dot3(a, a))

"""Normalize a vector, returning zero at the origin (a defined convention)."""
@inline function normalize3(a::Vec3{T}) where {T}
    d2 = dot3(a, a)
    return d2 > eps(T) * eps(T) ? a * inv(sqrt(d2)) : zero(Vec3{T})
end
