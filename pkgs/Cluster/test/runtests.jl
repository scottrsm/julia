using DataFrames
import Random
import RDatasets
using Test

using Cluster

#---------------------------------------------------------
#------------ DATA PREP ----------------------------------
#---------------------------------------------------------

# Set random seed.
Random.seed!(1)

# Constants used to set algo parameters.
const TOL            ::Float64 = 1.0e-4
const NUM_TRIALS_T1  ::Int64   = 300
const NUM_TRIALS_IRIS::Int64   = 1000
const NUM_ITERATIONS ::Int64   = 1000
const KM_THRESHOLD   ::Float64 = 1.0e-2

# Synthetic data for test: T1.
# There are 10 "natural" clusters.
M1 = [-1, -2] .+ rand(2, 100)
M2 = 3.0 .* [1, 2] .+ rand(2, 100)
M3 = 6.0 .* [2, 1] .+ rand(2, 100)
M4 = 9.0 .* [1, 1] .+ rand(2, 100)
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


@testset "Cluster (Fidelity)" begin
    @test length(detect_ambiguities(Cluster)) == 0
end


@testset "Cluster (Test Metrics)" begin
    C = [1. 2.; 2. 5.]

    @test L2([1., 2.], [3., -4.]     )   ≈  6.324555320336759   rtol=TOL
    @test L2([1., 2.], [3., -4.]; M=C)   ≈ 11.661903789690601   rtol=TOL
    @test LP([1., 2.], [3., -4.], 3  )   ≈  6.0731779437513245  rtol=TOL
end


@testset "Cluster (Test find_best_cluster: T1)" begin
    kbest, mp, xc, ds = find_best_cluster(M, 2:15                    ;
                                          num_trials = NUM_TRIALS_T1 , 
                                          N          = NUM_ITERATIONS, 
                                          threshold  = KM_THRESHOLD   )
    C = [-0.4647786335684582 3.4934544090790887 12.511540899137428 9.479449965138636 -11.462074155872033 7.993343670939024 16.067365037149926 21.145246393930183 24.49422035218324 27.447603324472404; 
         -1.5175945768316743 6.503798597479442 6.524309174201606 9.509921208077595 12.457669306098992 45.478311604029514 19.536020332393772 20.839828692792324 24.483990255724194 27.488695232426544  ] 

    best_var = 350.15757352935907 

    @test size(xc)   == (2, kbest)
    @test kbest      == 10
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL
end


# Try clustering with metrics: L2 (default), L1, KL (Kullback-Liebler).
@testset "Cluster (Test find_best_cluster: IRIS)" begin

    #--------------------------
    #----- Default metric, L2.
    #--------------------------
    kbest, mp, xc, ds = find_best_cluster(MI, 2:7                     ; 
                                          dmetric=L2                  , 
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS , 
                                          threshold  = KM_THRESHOLD    )

    C = [3.409803921568628 2.6999999999999997 3.0782608695652165; 
         5.003921568627451 5.800000000000001 6.823913043478258   ] 
    CM = [50 0 0  ;
           0 38 12;
           1 15 34 ]


    best_var = 62.69987288875754

    @test size(xc)   == (2, kbest)
    @test kbest      == 3
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL

    N, M = size(iris)
    iris[!, :Cluster] = map(i -> mp[i], 1:N)
    specs = Symbol.(iris[!, :Species])
    clus = Int64.(iris[!, :Cluster])
    res = raw_confusion_matrix(specs, clus)

    @test res[3] == CM

    #--------------------------
    #----- L1 metric.
    #--------------------------
    L1_metric = (x,y;kwargs...) -> LP(x,y,1;kwargs...) 
    kbest, mp, xc, ds = find_best_cluster(MI, 2:7                     ; 
                                          dmetric    = L1_metric      ,
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS ,
                                          threshold  = KM_THRESHOLD    )

    C = [3.3109090909090915 2.753846153846154 3.0999999999999996; 
         4.996363636363635 5.903846153846152 6.8530488372093022  ]
    best_var = 81.04100016262802

 
    @test size(xc)   == (2, kbest)
    @test kbest      == 3
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL


    #--------------------------
    #----- Kullback-Leibler metric.
    #--------------------------
    kbest, mp, xc, ds = find_best_cluster(MI, 2:7                     ; 
                                          dmetric    = KL             , 
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS , 
                                          threshold  = KM_THRESHOLD    )

    C = [3.4510204081632656 2.672222222222222 3.089361702127659; 
         5.016326530612244 5.757407407407408 6.804255319148933  ]
    best_var = 8.310219089521219

    @test size(xc)   == (2, kbest)
    @test kbest      == 3
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL


    #--------------------------
    #----- Cosine metric.
    #--------------------------
    kbest, mp, xc, ds = find_best_cluster(MI, 2:7                     ; 
                                          dmetric    = CD             , 
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS , 
                                          threshold  = KM_THRESHOLD    )

    C = [3.4510204081632656 2.985714285714286 2.717777777777777; 
         5.016326530612244 6.058928571428571 6.4755555555555535 ]
    best_var = 0.05437932001281265

    @test size(xc)   == (2, kbest)
    @test kbest      == 3
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL


    #--------------------------
    #----- Jaccard metric.
    #--------------------------
    kbest, mp, xc, ds = find_best_cluster(MI, 2:7                     ; 
                                          dmetric    = JD             , 
                                          num_trials = NUM_TRIALS_IRIS, 
                                          N          = NUM_ITERATIONS , 
                                          threshold  = KM_THRESHOLD    )

    C = [3.057333333333334; 
         5.843333333333335 ]
    best_var = 150.0

    @test size(xc)   == (2, kbest)
    @test kbest      == 1
    @test xc ≈ C          rtol=TOL
    @test ds ≈ best_var   rtol=TOL


end


