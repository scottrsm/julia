using Test
using Wordle
import InlineStrings: String7

@testset "Test Module \"Wordle\" Fidelity" begin
    @test length(detect_ambiguities(Wordle)) == 0
end

@testset "create_wordle_info" begin

    winfo, d = create_wordle_info(String7("which"), String7("where"))
    @test  winfo == [('w', 1), ('h', 2)]
    @test  d ==  Dict('h' => (0, 0), 'c' => (0, 0), 'i' => (0, 0))
    winfo, d = create_wordle_info(String7("teens"), String7("where"))
    @test  winfo == [('e', 3), ('e', -2)]
    @test  d == Dict('n' => (0, 0), 's' => (0, 0), 't' => (0, 0), 'e' => (1, 1))
end

@testset "filter_universe" begin
    ## Universe of words.
    words    = String7["state", "which", "where", "child", "there", "taste"]

    winfo, d = create_wordle_info(String7("which"), String7("where"))
    filter_words = filter_universe((winfo, d), words)
    @test filter_words == String7["where"]
end

@testset "solve_wordle with String7 Database with String7 Inputs" begin
    res = solve_wordle(String7("taste"); init_guess=String7("their"))
    @test res == (Any[(String7("their"), [('t', 1), ('e', -3)], 3585), 
                      (String7("taken"), [('t', 1), ('a', 2), ('e', -4)], 34), 
                      (String7("table"), [('t', 1), ('a', 2), ('e', 5)], 3), 
                      (String7("taste"), [('t', 1), ('a', 2), ('s', 3), ('t', 4), ('e', 5)], 2)], 4, :SUCCESS)

end

@testset "solve_wordle with String7 Database with String Inputs" begin
    res = solve_wordle("taste"; init_guess="their")
    @test res == (Any[("their", [('t', 1), ('e', -3)], 3585), 
                      ("taken", [('t', 1), ('a', 2), ('e', -4)], 34), 
                      ("table", [('t', 1), ('a', 2), ('e', 5)], 3), 
                      ("taste", [('t', 1), ('a', 2), ('s', 3), ('t', 4), ('e', 5)], 2)], 4, :SUCCESS)

end



