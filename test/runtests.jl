using Test
using ITensors, ITensorMPS, LocallyPurifiedDensityOperators, LinearAlgebra

@testset "Basic linear algebra" begin
    s = siteinds("Boson", 5; dim=4)
    x = random_mps(s; linkdims=2)
    normalize!(x)
    ρ = LPDO(x)
    u = random_itensor(ComplexF64, s[4]', dag(s[4]))
    t = random_itensor(ComplexF64, s[2]', s[3]', dag(s[2]), dag(s[3]))
    @test tr(ρ) ≈ 1
    @test inner(x, apply(u, x)) ≈ tr(apply(u, ρ))
    @test inner(x, apply(t, x)) ≈ tr(apply(t, ρ))
    @test norm(apply(u, x))^2 ≈ tr(apply(u, ρ; apply_dag=true))
    @test norm(apply(t, x))^2 ≈ tr(apply(t, ρ; apply_dag=true))
end
