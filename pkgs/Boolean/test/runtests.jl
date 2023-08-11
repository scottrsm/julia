
using Test
using Boolean

@testset "Boolean" begin
    init_logic(3)
    @test create_bool_rep("z1 + z2") == Blogic("z1 + z2", "z", BitVector(Bool[0, 1, 1, 1, 0, 1, 1, 1]))
    @test simplifyLogic(Meta.parse("x1 * (1 ⊕ x1)")) == :(x1 * ~x1)
    @test simplifyLogic(Meta.parse("(z1 ⊕ z3) ⊕ (0 ⊕ z3)")) == :(z3 ⊕ (z1 ⊕ z3))
    @test simplifyLogic(Meta.parse("((x1 + x2) * x3) * (x1 * 0 + x2 * 1)")) == :(x2 * (x3 * (x1 + x2)))
    @test simplifyLogic(Meta.parse("0 + x1")) == :x1
    @test simplifyLogic(Meta.parse("x1 * (x1 + 0)")) == :x1
end

