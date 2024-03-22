using Test
using Wquantiles
import Random


@testset "Wquantiles (Fidelity)" begin
    @test length(detect_ambiguities(Wquantiles)) == 0
end

@testset "Wquantiles (wquantiles (vector x, vector w) Types)" begin

    @test wquantile([1,3,5,7,9,11,15,7,9,11], [1,1,2,1,1,2,1,3,5,1], [0.25, 0.5, 0.6, 0.75])             == [7, 9, 11, 11]
    @test wquantile([1,3,5,7,9,11,15,7,9,11], [1.,1.,2.,1.,1.,2.,1.,3.,5.,1.], [1//4, 1//2, 3//5, 3//4]) == [7, 9, 11, 11]
    @test wquantile([1,3,5,7,9,11,15,7,9,11], [1,1,2,1,1,2,1,3,5,1], [1//4, 1//2, 3//5, 3//4])           == [7, 9, 11, 11]
    @test wquantile(UInt8[1,3,5,7,9,11,15,7,9,11], [1,1,2,1,1,2,1,3,5,1], [0.25, 0.5, 0.6, 0.75])        == [0x07, 0x09, 0x0b, 0x0b]
    @test wquantile([2,4,6,8,10,14,32,4,10,21], [1,1,2,1,1,2,1,3,5,1], [0.25, 0.5, 0.6, 0.75])           == [6   , 10  , 14  , 14  ]
    @test wquantile(UInt8[1,2,3,4,5,6,7,8,9,10], [1, 1, 2, 1, 1, 2, 1, 3, 5, 1], [0.25, 0.5, 0.6, 0.75]) == [0x05, 0x08, 0x09, 0x0a]
end

const TOL = 1.0e-6
Random.seed!(1)
X = rand(10,3)
W = rand(10,3)
qs = [0.1, 0.25, 0.5, 0.75, 0.9]
Result = [0.0491718  0.0994036  0.347513;
          0.119079   0.138227   0.438058;
          0.767518   0.347737   0.802561;
          0.855718   0.801055   0.972564;
          0.89077    0.977264   0.983662]

@testset "Wquantiles (wquantile (matrix X, matrix W))" begin
    @test all(isapprox.(vec(wquantile(X, W, qs)), vec(Result), atol=TOL))
end

