using Test
using Wordle

@testset "Test Module \"Wordle\" Fidelity" begin
    @test length(detect_ambiguities(Wordle)) == 0
end

@testset "create_wordle_info" begin

    winfo, d = create_wordle_info("which", "where")
    @test  winfo == [('w', 1), ('h', 2)]
    @test  d ==  Dict('h' => (0, 0), 'c' => (0, 0), 'i' => (0, 0))
    winfo, d = create_wordle_info("teens", "where")
    @test  winfo == [('e', 3)]
    @test  d == Dict('n' => (0, 0), 's' => (0, 0), 't' => (0, 0), 'e' => (1, 1))
end

@testset "filter_universe" begin
    ## Universe of words.
    words    = ["state", "which", "where", "child", "there", "taste"]

    winfo, d = create_wordle_info("which", "where")
    filter_words = filter_universe((winfo, d), words)
    @test filter_words == ["where"]
end

@testset "solve_wordle" begin
    res = solve_wordle("taste"; init_guess="their")
    @test res == (Any[
                      ("their", [('t', 1)], 3027), 
                      ("taken", [('t', 1), ('a', 2)], 31), 
                      ("table", [('t', 1), ('a', 2), ('e', 5)], 5), 
                      ("taste", [('t', 1), ('a', 2), ('s', 3), ('t', 4), ('e', 5)], 2), 
                     ], 4, :SUCCESS)  
end

