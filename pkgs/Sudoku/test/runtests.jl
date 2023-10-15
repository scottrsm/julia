using Test
using Sudoku

@testset "Test Module \"Sudoku\" Fidelity" begin
    @test length(detect_ambiguities(Sudoku)) == 0
end

@testset "Test Sudoku 'E' Puzzles" begin

    # Corresponds to puzzle ../puzzles/easy1.csv.
    EP1 = Int8[[ 0 0 0 2 6 0 7 0 1]
               [ 6 8 0 0 7 0 0 9 0]
               [ 1 9 0 0 0 4 5 0 0]
               [ 8 2 0 1 0 0 0 4 0]
               [ 0 0 4 6 0 2 9 0 0]
               [ 0 5 0 0 0 3 0 2 8]
               [ 0 0 9 3 0 0 0 7 4]
               [ 0 4 0 0 5 0 0 3 6]
               [ 7 0 3 0 1 8 0 0 0]]

    # Corresponding solution to puzzle ../puzzles/easy1.csv.
    ES1 = Int8[[ 4 3 5 2 6 9 7 8 1]
               [ 6 8 2 5 7 1 4 9 3]
               [ 1 9 7 8 3 4 5 6 2]
               [ 8 2 6 1 9 5 3 4 7]
               [ 3 7 4 6 8 2 9 1 5]
               [ 9 5 1 7 4 3 6 2 8]
               [ 5 1 9 3 2 6 8 7 4]
               [ 2 4 8 9 5 7 1 3 6]
               [ 7 6 3 4 1 8 2 5 9]]

    # Corresponds to puzzle ../puzzles/easy2.csv.
    EP2 = Int8[[ 5 3 0 0 7 0 0 0 0]
               [ 6 0 0 1 9 5 0 0 0]
               [ 0 9 8 0 0 0 0 6 0]
               [ 8 0 0 0 6 0 0 0 3]
               [ 4 0 0 8 0 3 0 0 1]
               [ 7 0 0 0 2 0 0 0 6]
               [ 0 6 0 0 0 0 2 8 0]
               [ 0 0 0 4 1 9 0 0 5]
               [ 0 0 0 0 8 0 0 7 9]]

    # Corresponding solution to puzzle ../puzzles/easy2.csv.
    ES2 = Int8[[ 5 3 4 6 7 8 9 1 2]
               [ 6 7 2 1 9 5 3 4 8]
               [ 1 9 8 3 4 2 5 6 7]
               [ 8 5 9 7 6 1 4 2 3]
               [ 4 2 6 8 5 3 7 9 1]
               [ 7 1 3 9 2 4 8 5 6]
               [ 9 6 1 5 3 7 2 8 4]
               [ 2 8 7 4 1 9 6 3 5]
               [ 3 4 5 2 8 6 1 7 9]]

    # Corresponds to puzzle ../puzzles/easy3.csv.
    EP3 = Int8[[ 0 0 0 6 0 0 4 0 0]
               [ 7 0 0 0 0 3 6 0 0]
               [ 0 0 0 0 9 1 0 8 0]
               [ 0 0 0 0 0 0 0 0 0]
               [ 0 5 0 1 8 0 0 0 3]
               [ 0 0 0 3 0 6 0 4 0]
               [ 0 4 0 2 0 0 0 6 0]
               [ 9 0 3 0 0 0 0 0 0]
               [ 0 2 0 0 0 0 0 0 0]]

    # Corresponding solution to puzzle ../puzzles/easy3.csv.
    ES3 = Int8[[ 3 9 1 6 7 8 4 5 2]
               [ 7 8 4 5 2 3 6 1 9]
               [ 5 6 2 4 9 1 3 8 7]
               [ 1 3 8 9 4 2 5 7 6]
               [ 4 5 6 1 8 7 2 9 3]
               [ 2 7 9 3 5 6 1 4 8]
               [ 8 4 7 2 3 5 9 6 1]
               [ 9 1 3 7 6 4 8 2 5]
               [ 6 2 5 8 1 9 7 3 4]]

    # Easy1 test.
    (ok, chk_sol, SS) = solve_sudoku(EP1)

    @test ok
    @test chk_sol
    @test SS == ES1 

    # Easy2 test.
    (ok, chk_sol, SS) = solve_sudoku(EP2)

    @test ok
    @test chk_sol
    @test SS == ES2 

    # Easy3 test.
    (ok, chk_sol, SS) = solve_sudoku(EP3)

    @test ok
    @test chk_sol
    @test SS == ES3 

end

@testset "Test Sudoku 'M' Puzzles" begin
    # Corresponds to puzzle ../puzzles/medium1.csv.
    MP1 = Int8[[ 0 0 0 6 0 0 4 0 0]
               [ 7 0 0 0 0 3 6 0 0]
               [ 0 0 0 0 9 1 0 8 0]
               [ 0 0 0 0 0 0 0 0 0]
               [ 0 5 0 1 8 0 0 0 3]
               [ 0 0 0 3 0 6 0 4 5]
               [ 0 4 0 2 0 0 0 6 0]
               [ 9 0 3 0 0 0 0 0 0]
               [ 0 2 0 0 0 0 1 0 0]]

    # Corresponding solution to puzzle ../puzzles/medium1.csv.
    MS1 = Int8[[ 5 8 1 6 7 2 4 3 9]
               [ 7 9 2 8 4 3 6 5 1]
               [ 3 6 4 5 9 1 7 8 2]
               [ 4 3 8 9 5 7 2 1 6]
               [ 2 5 6 1 8 4 9 7 3]
               [ 1 7 9 3 2 6 8 4 5]
               [ 8 4 5 2 1 9 3 6 7]
               [ 9 1 3 7 6 8 5 2 4]
               [ 6 2 7 4 3 5 1 9 8]]

    # Corresponds to puzzle ../puzzles/medium2.csv.
    MP2 = Int8[[ 2 0 0 3 0 0 0 0 0]
               [ 8 0 4 0 6 2 0 0 3]
               [ 0 1 3 8 0 0 2 0 0]
               [ 0 0 0 0 2 0 3 9 0]
               [ 5 0 7 0 0 0 6 2 1]
               [ 0 3 2 0 0 6 0 0 0]
               [ 0 2 0 0 0 9 1 4 0]
               [ 6 0 1 2 5 0 8 0 9]
               [ 0 0 0 0 0 1 0 0 2]]

    # Corresponding solution to puzzle ../puzzles/medium2.csv.
    MS2 = Int8[[ 2 7 6 3 1 4 9 5 8]
               [ 8 5 4 9 6 2 7 1 3]
               [ 9 1 3 8 7 5 2 6 4]
               [ 4 6 8 1 2 7 3 9 5]
               [ 5 9 7 4 3 8 6 2 1]
               [ 1 3 2 5 9 6 4 8 7]
               [ 3 2 5 7 8 9 1 4 6]
               [ 6 4 1 2 5 3 8 7 9]
               [ 7 8 9 6 4 1 5 3 2]]

    # Corresponds to puzzle ../puzzles/medium3.csv.
    MP3 = Int8[[ 0  0  0  6  0  0  4  0  0]
               [ 7  0  0  0  0  3  6  0  0]
               [ 0  0  0  0  9  1  0  8  0]
               [ 0  0  0  0  0  0  0  0  0]
               [ 0  5  0  1  8  0  0  0  3]
               [ 0  0  0  3  0  6  0  4  0]
               [ 0  4  0  2  0  0  0  6  0]
               [ 9  0  3  0  0  0  0  0  0]
               [ 0  2  0  0  0  0  1  0  0]]

    # Corresponding solution to puzzle ../puzzles/medium3.csv.
    MS3 = Int8[[ 1  9  5  6  7  8  4  3  2]
               [ 7  8  4  5  2  3  6  1  9]
               [ 3  6  2  4  9  1  7  8  5]
               [ 8  3  1  9  4  2  5  7  6]
               [ 4  5  6  1  8  7  9  2  3]
               [ 2  7  9  3  5  6  8  4  1]
               [ 5  4  8  2  1  9  3  6  7]
               [ 9  1  3  7  6  4  2  5  8]
               [ 6  2  7  8  3  5  1  9  4]]      


    # Medium1 test.
    (ok, chk_sol, SS) = solve_sudoku(MP1)

    @test ok
    @test chk_sol
    @test SS == MS1 

    # Medium2 test.
    (ok, chk_sol, SS) = solve_sudoku(MP2)

    @test ok
    @test chk_sol
    @test SS == MS2 

    # Medium3 test.
    (ok, chk_sol, SS) = solve_sudoku(MP3)

    @test ok
    @test chk_sol
    @test SS == MS3 

end

@testset "Test Sudoku 'S' Puzzles" begin

    # Corresponds to puzzle ../puzzles/sm1.csv.
    SP1 = Int8[[ 0  0  0  6  0  0  4  0  0]
               [ 7  0  0  0  0  3  6  0  0]
               [ 0  0  0  0  9  1  0  8  0]
               [ 0  0  0  0  0  0  0  0  0]
               [ 0  5  0  1  8  0  0  0  3]
               [ 0  0  0  3  0  6  0  4  0]
               [ 0  4  0  2  0  0  0  6  0]
               [ 0  0  3  0  0  0  0  0  0]
               [ 0  2  0  0  0  0  1  0  0]]

    # Corresponding solution to puzzle ../puzzles/sm1.csv.
    SS1 = Int8[[ 3 1 9 6 7 8 4 5 2]
               [ 7 8 4 5 2 3 6 1 9]
               [ 2 6 5 4 9 1 3 8 7]
               [ 1 3 8 7 4 9 5 2 6]
               [ 4 5 6 1 8 2 7 9 3]
               [ 9 7 2 3 5 6 8 4 1]
               [ 8 4 1 2 3 7 9 6 5]
               [ 6 9 3 8 1 5 2 7 4]
               [ 5 2 7 9 6 4 1 3 8]]

    # Corresponds to puzzle ../puzzles/sm2.csv.
    SP2 = Int8[[ 0 0 0 6 0 0 4 0 0]
               [ 7 0 0 0 0 3 6 0 0]
               [ 0 0 0 0 9 1 0 8 0]
               [ 0 0 0 0 0 0 0 0 0]
               [ 0 5 0 1 8 0 0 0 3]
               [ 0 0 0 3 0 6 0 4 0]
               [ 0 4 0 2 0 0 0 6 0]
               [ 0 0 3 0 0 0 0 0 0]
               [ 0 2 0 0 0 0 1 0 0]]

    # Corresponding solution to puzzle ../puzzles/sm2.csv.
    SS2 = Int8[[ 3 1 9 6 7 8 4 5 2]
               [ 7 8 4 5 2 3 6 1 9]
               [ 2 6 5 4 9 1 3 8 7]
               [ 1 3 8 7 4 9 5 2 6]
               [ 4 5 6 1 8 2 7 9 3]
               [ 9 7 2 3 5 6 8 4 1]
               [ 8 4 1 2 3 7 9 6 5]
               [ 6 9 3 8 1 5 2 7 4]
               [ 5 2 7 9 6 4 1 3 8]]

    # S1 test.
    (ok, chk_sol, SS) = solve_sudoku(SP1)

    @test ok
    @test chk_sol
    @test SS == SS1 

    # S2 test.
    (ok, chk_sol, SS) = solve_sudoku(SP2)

    @test ok
    @test chk_sol
    @test SS == SS2 

end
