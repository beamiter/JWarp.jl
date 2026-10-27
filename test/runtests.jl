using Test
using JWarp

# Define the kernel at module top level to keep macro-generated methods global.
@kernel function square_kernel!(out, input)
    i = @index(Global, Linear)
    @inbounds out[i] = input[i] * input[i]
end

@testset "JWarp.jl proof of concept" begin
    @testset "Vec3 and geometry" begin
        a = Vec3f(1, 2, 3)
        b = Vec3f(3, 2, 1)
        @test isbitstype(typeof(a))
        @test a + b == Vec3f(4, 4, 4)
        @test a - b == Vec3f(-2, 0, 2)
        @test 2f0 * a == Vec3f(2, 4, 6)
        @test dot3(a, b) == 10f0
        @test cross3(Vec3f(1, 0, 0), Vec3f(0, 1, 0)) == Vec3f(0, 0, 1)
        @test norm3(Vec3f(3, 4, 0)) == 5f0
        @test normalize3(Vec3f(0, 0, 0)) == Vec3f(0, 0, 0)
        @test sphere_sdf(Vec3f(2, 0, 0), Vec3f(0, 0, 0), 1f0) == 1f0
        @test sphere_sdf_gradient(Vec3f(2, 0, 0), Vec3f(0, 0, 0)) == Vec3f(1, 0, 0)
        @test plane_sdf(Vec3f(1, 2, 3), Vec3f(0, 0, 1), 1f0) == 2f0
    end

    @testset "Scalar contact / ray queries" begin
        c = sphere_sphere_contact(Vec3f(0, 0, 0), 1f0, Vec3f(1.5, 0, 0), 1f0)
        @test c.signed_distance == -0.5f0
        @test c.normal == Vec3f(1, 0, 0)
        coincident = sphere_sphere_contact(Vec3f(0, 0, 0), 1f0, Vec3f(0, 0, 0), 1f0)
        @test coincident.signed_distance == -2f0
        @test coincident.normal == Vec3f(1, 0, 0)

        hit = ray_sphere_hit(Vec3f(-3, 0, 0), Vec3f(1, 0, 0), Vec3f(0, 0, 0), 1f0)
        @test hit.hit
        @test hit.t ≈ 2f0
        @test hit.normal == Vec3f(-1, 0, 0)
        miss = ray_sphere_hit(Vec3f(-3, 3, 0), Vec3f(1, 0, 0), Vec3f(0, 0, 0), 1f0)
        @test !miss.hit
        @test isinf(miss.t)
    end

    @testset "User-defined kernel API (CPU)" begin
        input = Float32[1, 2, 3, 4]
        output = zeros(Float32, 4)
        launch!(square_kernel!, output, input; ndrange = 4, sync = true)
        @test output == Float32[1, 4, 9, 16]
    end

    @testset "Symplectic Euler integration (CPU)" begin
        n = 128
        x = fill(Vec3f(0, 0, 0), n)
        v = fill(Vec3f(1, 0, 0), n)
        a = fill(Vec3f(0, -9.8f0, 0), n)
        integrate_particles!(x, v, a, 0.1f0; sync = true)
        @test all(p -> p.x ≈ 0.1f0 && p.y ≈ -0.098f0, x)
        @test all(p -> p.y ≈ -0.98f0, v)
        @test_throws DimensionMismatch integrate_particles!(x, v[1:10], a, 0.1f0)
    end

    @testset "Batched contacts (CPU)" begin
        n = 100
        a = fill(Vec3f(0, 0, 0), n)
        b = fill(Vec3f(1.5, 0, 0), n)
        ra, rb = fill(1f0, n), fill(1f0, n)
        d = zeros(Float32, n)
        normals = fill(Vec3f(0, 0, 0), n)
        batch_sphere_contacts!(d, normals, a, ra, b, rb; sync = true)
        @test all(==(-0.5f0), d)
        @test all(==(Vec3f(1, 0, 0)), normals)
    end
end
