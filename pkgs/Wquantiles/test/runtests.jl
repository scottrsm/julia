using Test
using Wquantiles

@testset "Test Module Fidelity" begin
    @test length(detect_ambiguities(Wquantiles)) == 0
end

@testset "wquantiles Types" begin

    @test wquantile([1,3,5,7,9,11,15,7,9,11], [1,1,2,1,1,2,1,3,5,1], [0.25, 0.5, 0.6, 0.75])             == [7, 9, 11, 11]
    @test wquantile([1,3,5,7,9,11,15,7,9,11], [1.,1.,2.,1.,1.,2.,1.,3.,5.,1.], [1//4, 1//2, 3//5, 3//4]) == [7, 9, 11, 11]
    @test wquantile([1,3,5,7,9,11,15,7,9,11], [1,1,2,1,1,2,1,3,5,1], [1//4, 1//2, 3//5, 3//4])           == [7, 9, 11, 11]
    @test wquantile(UInt8[1,3,5,7,9,11,15,7,9,11], [1,1,2,1,1,2,1,3,5,1], [0.25, 0.5, 0.6, 0.75])        == [0x07, 0x09, 0x0b, 0x0b]
    @test wquantile([2,4,6,8,10,14,32,4,10,21], [1,1,2,1,1,2,1,3,5,1], [0.25, 0.5, 0.6, 0.75])           == [6   , 10  , 14  , 14  ]
    @test wquantile(UInt8[1,2,3,4,5,6,7,8,9,10], [1, 1, 2, 1, 1, 2, 1, 3, 5, 1], [0.25, 0.5, 0.6, 0.75]) == [0x05, 0x08, 0x09, 0x0a]
end

