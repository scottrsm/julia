using Test
using Cluster
import Random
import RDatasets


#---------------------------------------------------------
#------------ DATA PREP ----------------------------------
#---------------------------------------------------------

# Set random seed.
Random.seed!(1)

# Constants used to fix algo parameters.
const TOL            ::Float64 = 1.0e-4
const NUM_TRIALS_T1  ::Int64   = 300
const NUM_TRIALS_IRIS::Int64   = 1000
const NUM_ITERATIONS ::Int64   = 1000
const KM_THRESHOLD   ::Float64 = 1.0e-2

# Synthetic data for test: T1.
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

# Data for test: IRIS.
iris = RDatasets.dataset("datasets", "iris")
MI = permutedims(Matrix(iris[:, [:SepalWidth, :SepalLength]]), (2,1))


@testset "Test Module \"Cluster\" Fidelity" begin
    @test length(detect_ambiguities(Cluster)) == 0
end


@testset "Test Metrics" begin
    C = [1. 2.; 2. 5.]

    @test L2([1., 2.], [3., -4.]   ) ≈  6.324555320336759   rtol=TOL
    @test L2([1., 2.], [3., -4.]; C) ≈ 11.661903789690601   rtol=TOL
    @test LP([1., 2.], [3., -4.], 3) ≈  6.0731779437513245  rtol=TOL
end


@testset "Test find_best_cluster (T1)" begin
    kbest, mp, xc, ds = find_best_cluster(M, 2:15                    ;
                                          num_trials = NUM_TRIALS_T1 , 
                                          N          = NUM_ITERATIONS, 
                                          threshold  = KM_THRESHOLD   )
    C = [-0.464779  16.0674  21.1452  -11.4621   7.99334  25.9709  3.49345  10.9955;
         -1.51759   19.536   20.8398   12.4577  45.4783   25.9863  6.5038    8.01712]
    best_var = 1053.0476313368601

    @test size(xc)   == (2, kbest)
    @test kbest      == 8
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL
end


# Try clustering with metrics: L2 (default), L1, KL (Kullback-Liebler).
@testset "Test find_best_cluster (IRIS)" begin
    kbest, mp, xc, ds = find_best_cluster(MI, 2:7                     ; 
                                          dmetric=L2                  , 
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS , 
                                          threshold  = KM_THRESHOLD    )

    C = [3.428 2.6903846153846156 3.068749999999999         ; 
         5.005999999999999 5.76346153846154 6.802083333333332]
    best_var = 62.56479718603022

    @test size(xc)   == (2, kbest)
    @test kbest      == 3
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL


    L1_metric = (x,y;kwargs...) -> LP(x,y,1;kwargs...) 
    kbest, mp, xc, ds = find_best_cluster(MI, 2:7                     ; 
                                          dmetric    = L1_metric      ,
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS ,
                                          threshold  = KM_THRESHOLD    )

    C = [3.3109090909090915 2.7538461538461543 3.0999999999999996; 
         4.996363636363638 5.903846153846155 6.853488372093022    ] 
    best_var = 80.72790697674415
 
    @test size(xc)   == (2, kbest)
    @test kbest      == 3
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL

    kbest, mp, xc, ds = find_best_cluster(MI, 2:7                     ; 
                                          dmetric    = KL             , 
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS , 
                                          threshold  = KM_THRESHOLD    )

    C = [2.6510204081632653 3.451020408163265 3.0692307692307685; 
         5.7102040816326545 5.016326530612244 6.748076923076921  ] 
    best_var = 8.310123590013273 

    @test size(xc)   == (2, kbest)
    @test kbest      == 3
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL
end



