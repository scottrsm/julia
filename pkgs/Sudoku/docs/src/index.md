# Sudoku.jl Documentation

```@meta
CurrentModule = Sudoku
```

# Overview
Below are two are two Sudoku solvers: One takes a puzzle as a matrix while the
other takes a file. In addition, a few utility functions that help 
in the process are listed.

- `get_block`            : This function retrieves sub-blocks of a Sudoku matrix.
- `get_blk_idx`          : This function determines which sub-block matrix to pick. 
- `has_dups`             : Checks a given vector/matrix for non-zero duplicates.
- `consist_chk`          : Checks for consistency of puzzle, partial, or full Sudoku solution matrix. 
- `check_sudoku_solution`: Checks a proposed solution matrix against its puzzle matrix.
- `solve_sudoku`         : Solves a Sudoku puzzle represented as a matrix.
- `solve_sudoku_puzzle`  : Solves a Sudoku puzzle represented as a file.

## Utilities

```@docs
get_block
```

```@docs
get_blk_idx
```

```@docs
has_dups
```

```@docs
consist_chk
```

```@docs
check_sudoku_solution
```

## Sudoku Solvers

```@docs
solve_sudoku
```

```@docs
solve_sudoku_file
```

## Index

```@index
```

