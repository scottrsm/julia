
using Test
using Boolean

@testset "Boolean" begin
    ## Set up
    init_logic(3)
    f1 = "x1 + x2 * x3"
    f2 = "(x1 * x3) ⊕ (x2 * x3)"

    ## Test Module for function ambiguities.
    @test length(detect_ambiguities(Boolean)) == 0

    ## Tests Functions
    @test create_bool_rep("z1 + z2") == Blogic("z1 + z2", "z", BitVector(Bool[0, 1, 1, 1, 0, 1, 1, 1]))
    @test BitArray([((i-1) >> (j-1)) & 1  for i in 1:2^3, j in 1:3]) == BitMatrix(Bool[ 0  0  0;
                                                                                        1  0  0;
                                                                                        0  1  0;
                                                                                        1  1  0;
                                                                                        0  0  1;
                                                                                        1  0  1;
                                                                                        0  1  1;
                                                                                        1  1  1])
    @test simplifyLogic(Meta.parse("x1 * (1 ⊕ x1)")) == :(x1 * ~x1)
    @test simplifyLogic(Meta.parse("(z1 ⊕ z3) ⊕ (0 ⊕ z3)")) == :(z3 ⊕ (z1 ⊕ z3))
    @test simplifyLogic(Meta.parse("((x1 + x2) * x3) * (x1 * 0 + x2 * 1)")) == :(x2 * (x3 * (x1 + x2)))
    @test simplifyLogic(Meta.parse("0 + x1")) == :x1
    @test simplifyLogic(Meta.parse("x1 * (x1 + 0)")) == :x1
    @test isEquiv(f1, f1)
    @test ~ isEquiv(f1,f2)
end

