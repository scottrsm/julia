module Sudoku

export  solve_sudoku, solve_sudoku_file, check_sudoku_solution

import CSV
import DataFrames
import OrderedCollections: OrderedDict


# -------------------------------------------
# -------   Module constants    -------------
# -------------------------------------------
const SUDOKU_SIZE     ::Int8         = 9
const SUDOKU_BLK_SIZE ::Int8         = 3
const SUDOKU_INIT_VALS::Vector{Int8} = Int8[0,1,2,3,4,5,6,7,8,9]
const SUDOKU_VALS     ::Set{Int8}    = Set{Int8}([1,2,3,4,5,6,7,8,9])
const SUDOKU_SUM      ::Int8         = div(SUDOKU_SIZE * (SUDOKU_SIZE + 1), 2) 

# Used to compute the maximum recursion depth of the solve_sudoku function.
MAX_RECUR_DEPTH=1

"""
    get_block(S, i, j)

Get the `(i, j)` sub-block matrix of `S`.

## Arguments
- `S::Matrix{Int8}` -- A sudoku matrix.
- `i::Int64`        -- The ``i^{i\\rm th}`` block row entry.
- `j::Int64`        -- The ``j^{j\\rm th}`` block column entry.

## Return
`::Matrix{Int8}` -- The ``(i^{\\rm th}, j^{\\rm th})`` sub-block matrix. 
"""
function get_block(S::Matrix{Int8}, i::Int64, j::Int64)
    S[SUDOKU_BLK_SIZE * (i-1) + 1:SUDOKU_BLK_SIZE * i, SUDOKU_BLK_SIZE * (j-1) + 1:SUDOKU_BLK_SIZE * j]
end


"""
    get_blk_idx(h)

Get the sub-block index pair for a given index pair of a Sudoku matrix.

## Arguments
- `h::CartesianIndex` -- The index into a Sudoku matrix.

## Examples
The code below gets the sub-block index pair, (2,2) (the index representing the middle 3x3 block matrix),
when passed the Sudoku matrix index pair (4,6).
```jdoctest
julia> get_blk_idx(CaresianIndex(4, 6))
(2,2)
```

## Return
`::Tuple{Int64, Int64}` -- The index pair of the sub-matrix block.
"""
function get_blk_idx(h::CartesianIndex)
    (div(h[1]-1, SUDOKU_BLK_SIZE) + 1, div(h[2]-1, SUDOKU_BLK_SIZE) + 1)
end

"""
    check_sudoku_solution(SP, SS)

Checks that `SS` is a solution of the puzzle, `SP`.

It does this by doing the following:
- Checks for any (non-zero) duplicate entries in 
  any rows, columns, or sub-blocks.
- Checks that the row, column, and sub-block sums
  are `SUDOKU_SUM`.
- Checks that the proposed solution is consistent
  with the puzzle matrix, `SP`. 

## Arguments
- `SP::Matrix{Int8}` -- A Sudoku puzzle in matrix form.
                        Zeros represent blanks.
- `SS::Matrix{Int8}` -- Proposed solution for `SP`.

## Returns
`::Bool` -- If `true`, the proposed solution is correct.
"""
function check_sudoku_solution(SP, SS)
    # Check for consistency -- no (non-zero) duplicates.
    if !consist_chk(SS)
        return false
    end

    # Number of blanks in puzzle.
    num_puzzle_blanks = count(x -> x == 0, SP)

    # Check the row sums are correct
    # If the sums are correct and we know there are
    # no duplicates, then all non-zero values
    # appear once and only once.
    for i in 1:SUDOKU_SIZE
        sum(SS[i, :]) != SUDOKU_SUM && return false 
    end

    for j in 1:SUDOKU_SIZE
        sum(SS[:, j]) != SUDOKU_SUM && return false 
    end

    for i in 1:SUDOKU_BLK_SIZE
        for j in 1:SUDOKU_BLK_SIZE
            B = get_block(SS, i, j)
            sum(B) != SUDOKU_SUM && return false
        end
    end

    # Check that `SS` is consistent with the puzzle `SP`.
    # Use the fact that 0 represents blanks.
    idx = findall(x -> x != 0, SP)
    return SS[idx] == SP[idx]
end


"""
    has_dups(v)

Check if a vector/matrix has duplicate numeric (other than 0) entries.

## Arguments
- `v :: Union{Vector{Int8}, Matrix{Int8}}`

## Return
`::Bool` -- If `true` there exists at least one duplicate.
"""
function has_dups(v::Union{Vector{Int8}, Matrix{Int8}})
    vn = filter(x -> x != 0, v)
    vs = sort(vn)
    if length(filter(x -> x == 0, diff(vs))) > 0 
        return true
    else
        return false
    end
end


"""
    consist_chk(S)

Checks the consistency of a Sudoku matrix, `S`.

This means that we check that there are no (non-zero)
duplicate entries in any rows, columns, or sub-blocks.

## Argumens
- `S::Matrix{Int8}` -- A Sudoku puzzle, proposed solution, or
                       intermediate solution.

## Return
`::Bool` -- Returns `true` if Sudoku matrix is consistent.
"""
function consist_chk(S)
    # Check all rows for non-zero duplicates.
    for i in 1:SUDOKU_SIZE
        has_dups(S[i, :]) && return false
    end

    # Check all columsn for non-zero duplicates.
    for j in 1:SUDOKU_SIZE
        has_dups(S[:, j]) && return false
    end

    # Check all sub-blocks for non-zero duplicates.
    for i in 1:SUDOKU_BLK_SIZE
        for j in 1:SUDOKU_BLK_SIZE
            B = get_block(S, i, j) 
            has_dups(B) && return false
        end
    end

    # No duplicates found, return `true`.
    return true
end




"""
    solve_sudoku(SP, rec_count; verbose=false)

Helper function that does the work of the top level solver.

## Arguments
- `S::Matrix{Int8}`  -- A Sudoku puzzle matrix.
- `rec_count::Int64` -- The count of the number of times this function has been called.

## Keyword Arguments
- `verbose::Bool=false` -- If `true`, print out extra information.

## Return 
(ok, SS) 
- `ok::Bool` -- If `true`, a *proposed solution* was found.
- `S::Matrix{Int8}}` -- A proposed, or inconsistent solution matrix.
"""
function solve_sudoku(SP::Matrix{Int8}, rec_count::Int64; verbose::Bool=false)
    # We copy the input Sudoku matrix as this function mutates its values.
    S = copy(SP)

    # Check input contract
    if rec_count == 1
        @assert size(S) == (SUDOKU_SIZE, SUDOKU_SIZE)
        global MAX_RECUR_DEPTH = rec_count
    end

    # Keep track of maximum recursion depth.
    if rec_count > MAX_RECUR_DEPTH
        if verbose
            println("New MAX_RECUR_DEPTH = $rec_count")
        end
        global MAX_RECUR_DEPTH = rec_count
    end

    last_num_holes = SUDOKU_SIZE * SUDOKU_SIZE + 1

    # Create dictionary, `CartesianIndex => Vector{Int8}`, of possible 
    # solution values for each hole.
    dict = OrderedDict{CartesianIndex, Set{Int8}}()

    # Here we do the naive filling of holes. These are cells which only
    # have one possible entry to go in that slot.
    # We break out of this loop, when the number of holes (0 entries)
    # of `S` doesn't change from one loop to the next.
    while true 
        # Get the indices where there are unknowns.
        # See if we made any progress with naive filling.
        holes = findall(x -> x == 0, S)
        nholes = length(holes)

        # Leave this loop.
        if nholes == last_num_holes
            break
        end

        # Record the number of holes before we look again.
        last_num_holes = nholes

        # Get the potential solution values for each hole.
        for hole in holes
            # Get the set of values in its row.
            row_fills  = Set(S[hole[1], :])
    
            # Get the set of values in its column.
            col_fills  = Set(S[:, hole[2]])
    
            # Get the set of values in its block.
            i, j       = get_blk_idx(hole)
            b          = get_block(S, i, j)
            blk_fills  = Set(b)

            # Get the valid Sudoku values not in the union of the above.
            dict[hole] = setdiff(SUDOKU_VALS, union(row_fills, col_fills, blk_fills))
        end

        # Sort the dictionary by value using the length of the value (a set) as the metric.
        sort!(dict, byvalue=true, by=x -> length(x))

        # Loop over the dict and fill in holes in S where there is only one choice for a potential solution.
        for k in keys(dict)
            pots = dict[k]
            pl = length(pots)
            # Return failure if there are no choices for some hole.
            if pl == 0
                return (false, S)
            elseif pl == 1
                S[k[1], k[2]] = collect(pots)[1]
                delete!(dict, k) 
            else
                break
            end
        end
    end

    # At this point `last_num_holes == nholes` -- we have finished with the naive hole filling.
    # We need to check for consistency -- no duplicates.
    # This can happen after recursing; where we try out different potential 
    # solutions for a given hole. 
    if ! consist_chk(S)
        return (false, S)
    end

    # One of two cases now: We've filled in all holes and have a solution;
    # or, we need to start making guesses and recurse.
    if last_num_holes == 0
        return (true, S)
    else
        # Recurse by filling in the holes with the smallest number
        # of potential solutions first, progressing to holes with 
        # more potential solutions.
        # This is because the `dict` keys are sorted by length of potential solutions.
        # NOTE: The code below employs the following heuristic:
        #       After exhausting all potential solutions for a given hole, `k`,
        #       we set S[k[1], k[2]] to the last value from `dict[k]`
        #       and continue on with the next hole.
        #       It is possible that this will lead to an incorrect solution.
        #       A final check of the proposed solution is done in 
        #       the function,`solve_sudoku_file`, which calls this function.
        for k in keys(dict)
            # For this hole, k, try each of the alternatives.
            for val in collect(dict[k])
                S[k[1], k[2]] = val
                # Note: `solve_sudoku` copies `S` so we are not wedded to the
                #       mutations it performs on `S`.
                ok, SS = solve_sudoku(S, rec_count+1, verbose=verbose)
                if ok
                    return (true, SS)
                end
                # Technically, one should set S[k[1], k[2]] = 0.
                # But this seems to take the solver a very long time
                # to come up with a proposed solution.
            end
        end
        # We were unable to find a solution.
        return (false, S)
    end
end

"""
    solve_sudoku(S; verbose=false)

Solves a Sudoku puzzle represented as a matrix. 

The value, `0`, is used in a puzzle matrix to represent a blank.

## Arguments
- `S::Matrix{Int8}`  -- A Sudoku puzzle matrix.

## Keyword Arguments
- `verbose::Bool=false`    -- If `true`, print out extra information.

## Return 
(ok, chk_sol, SS) 
- `ok::Bool` -- If `true`, a *proposed solution* was found.
- `chk_sol::Bool` -- If `true`, the proposed solution is *correct*. 
- `SS::Matrix{Int8}}` -- A proposed, or inconsistent/incomplete solution matrix.
"""
function solve_sudoku(SP::Matrix{Int8}; verbose::Bool=false) 
    (ok, SS) = solve_sudoku(SP, 1; verbose=verbose)
    chk_sol=false
    if ok
        chk_sol = check_sudoku_solution(SP, SS)
    end
    return (ok, chk_sol, SS)
end


"""
    solve_sudoku_file(puzzle_file_name; <keyword arguments>])

Solves and prints a solution of a Sudou puzzle file.

The file is the name of a CSV file (without extension).
The file format is `9` rows of values, `0-9`, with "0" representing a blank.

**NOTE:** The CSV file should not have a *header*.

## Arguments
- `puzzle_file_name::AbstractString` -- The puzzle file name without the extension.

## Keyword Arguments 
- `puzzle_dir=joinpath(@__DIR__, "../puzzles") :: AbstractString` -- The path to the puzzle file directory.
- `verbose=false :: Bool` -- If `true`, print more output.

## Return
`::Nothing`
"""
function solve_sudoku_file(puzzle_file_name :: AbstractString                           ; 
                           puzzle_dir=joinpath(@__DIR__, "../puzzles" :: AbstractString), 
                           verbose::Bool=false                                           )
    # Read the puzzle -- as a matrix.
    SP = Matrix{Int8}(CSV.read(joinpath(puzzle_dir, puzzle_file_name*".csv"), DataFrames.DataFrame; header=false, types=Int8))

    ## Check input contract -- should be a matrix of size: (SUDOKU_SIZE, SUDOKU_SIZE).
    @assert size(SP) == (SUDOKU_SIZE, SUDOKU_SIZE)
    println("Sudoku Puzzle:")
    display(SP)

    # Attempt to solve the sudoku puzzle using its matrix representation.
    ok, chk_sol, SS = solve_sudoku(SP; verbose=verbose)

    # Check the proposed solution.
    if ok && chk_sol
        ## Check that the proposed solution is correct.
        println("\nSOLUTION:")
        display(SS)
    elseif ok # Otherwise, print failure.
        println("FAILURE: No Solution:")
        display(SS)
    else
        println("FAILURE: Inconsistent Solution:")
        display(SS)
    end

    println("\n(MAX_RECUR_DEPTH = $MAX_RECUR_DEPTH)")

    # Return nothing.
    return(nothing)
end


end # Sudoku module
