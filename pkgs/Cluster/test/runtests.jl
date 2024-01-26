using Test
using Cluster
import Random


Random.seed!(1)
TOL=1.0e-4

M1 = [-1,-2] .+ rand(2, 100)
M2 = 3.0 .* [1,2] .+ rand(2, 100)
M3 = 6.0 .* [2,1] .+ rand(2, 100)
M4 = 9.0 .* [1,1] .+ rand(2, 100)
M5 = 12.0 .* [-1, 1] .+ rand(2, 100)
M6 = 15.0 .* [0.5, 3.0] .+ rand(2, 100)
M7 = 18.0 .+ [-2.4, 1.0] .+ rand(2, 100)
M8 = 21.0 .+ [0.3, -0.3] .* rand(2, 100)
M9 = 24.0 .+ rand(2, 100)
M10 = 27.0 .+ rand(2, 100)

M = hcat(M1, M2, M3, M4, M5, M6, M7, M8, M9, M10)

C = [1. 2.; 2. 5.]

@testset "Test Module \"Cluster\" Fidelity" begin
    @test length(detect_ambiguities(Cluster)) == 0
end


@testset "Test Metrics" begin
    @test L2([1., 2.], [3., -4.]   ) ≈  6.324555320336759   rtol=TOL
    @test L2([1., 2.], [3., -4.]; C) ≈ 11.661903789690601   rtol=TOL
    @test LP([1., 2.], [3., -4.], 3) ≈  6.0731779437513245  rtol=TOL
end


@testset "Test find_best_cluster" begin
    kbest, mp, xc, ds = find_best_cluster(M, 2:15, num_trials=300, N=1000, threshold=1.0e-2)
    C = [-0.464779  16.0674  21.1452  -11.4621   7.99334  25.9709  3.49345  10.9955;
         -1.51759   19.536   20.8398   12.4577  45.4783   25.9863  6.5038    8.01712]
    best_var = 1053.0476313368601
    @test size(xc)   == (2, kbest)
    @test kbest      == 8
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL
end
