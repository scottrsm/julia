module Wordle

export create_wordle_info, filter_universe, pick_guess, solve_wordle

using DataFrames
import CSV

## Load Wordle database -- stored as a CSV file. 
const LFA = collect("etaoinshrdlcumwfgypbvkjxqz")
const WORDLE_DF =  DataFrame(CSV.File(joinpath(@__DIR__, "../data", "wordle_db.csv"); header=3, types=[String, Float64], comment="#"));

"""
## Solver Strategy
- Initial Conditions:
  - Set puzzle_word
  - X = Start with universe of 5 letter words along with freqency of usage.
  - Set current_universe = X
 - Start
  - Pick guess (pick the most frequent word in current_universe that we haven't picked previously)
  - If guess == puzzle_word)
    - Goto End
  - Get wordle info about how close guess is to the correct word:
    - wordle_info = create_wordle_info(<guess>, <puzzle_word>)
      - Example wordle_info, create_wordle_info("exact", "crane") = ( [('a', 3)], Dict('x' => (0, 0), 'c' => (1, 1), 'e' => (1, 1), 't' => (0, 0)) )
  - Use this match info to filter existing universe.
    - current_universe = filter_universe(wordle_info, current_universe)
  - Goto Start
- End
   - Return guess
"""


"""
    create_wordle_info[guess, pword]

Create an information structure of the form: ([LETTER, EXACT_MATCH_POSITION)], Dict(LETTER => (NUMBER_OF_MATCHES, MATCH_FLAG))
    
Here, the dictionary has the in-exact match information:
    LETTER : A matching letter 
    NUMBER_OF_MATCHES : The number of matches.
    The latter iis interpreted thusly: 
        If MATCH_FLAG is 0, there are *exactly* NUMBER_OF_MATCHES with this letter that should occur in the puzzle word.
        Else              , there are *at least* NUMBER_OF_MATCHES with this letter that should occur in the puzzle word.

## Arguments
- `guess::String`: The guess for the puzzle.
- `pword::String`: The puzzle word.
     
    
## Returns
    A tuple of a vector of tuples of exact matches and a dictionary of non-exact match info.

## Examples
    `(winfo, d) = create_wordle_info("which", "where")`
    Output: `([('w', 1), ('h', 2)], Dict('h' => (0, 0), 'c' => (0, 0), 'i' => (0, 0)))`
    `(winfo, d) = create_wordle_info("teens", "where")`
    Output: `([('e', 3)], Dict('n' => (0, 0), 's' => (0, 0), 't' => (0, 0), 'e' => (1, 1)))`
"""
function create_wordle_info(guess :: String, # Guess
                            pword :: String, # Puzzle word
                           ) :: Tuple{Vector{Tuple{Char, Int64}}, Dict{Char, Tuple{Int64, Int64}}}
    n     :: Int64                            = length(pword)
    e_idx :: Vector{Int64}                    = []
    f_idx :: Vector{Int64}                    = collect(1:n)
    c_idx :: Vector{Int64}                    = []

    ary :: Vector{Tuple{Char, Int64}} = []
  
    ## Push exact info onto `ary`.
    for i in 1:n
        if guess[i] == pword[i]
            push!(ary, (guess[i], i))
            push!(e_idx, i)
        end
    end
    
    c_idx = setdiff(f_idx, e_idx)

    dp = Dict{Char, Int64}()
    dg = Dict{Char, Int64}()
    for i in c_idx
        dp[pword[i]] = 1 + get(dp, pword[i], 0)
        dg[guess[i]] = 1 + get(dg, guess[i], 0)
    end

    d = Dict{Char, Tuple{Int64, Int64}}()
    for i in c_idx
        ## We know that there is a AT LEAST `dg[guess[i]]` of character `guess[i]` in the puzzle word.
        if dg[guess[i]] <= get(dp, guess[i], 0)
            d[guess[i]] = (dg[guess[i]], 1)         
        else # We know that there is EXACTLY `dp[guess[i]]` of character `guess[i]` in the puzzle word.
            d[guess[i]] = (get(dp, guess[i], 0), 0) 
        end
    end
    
    return((ary, d))
end


"""
    Filter an existing universe of words based on match info.

    Parameters
    ----------
    wordle_info : Wordle info of the form ([(LETTER, EXACT_POSITION)], Dict( LETTER => (k, n)))
                  The wordle info as the same type as the return value form create_wordle_info.
    words       : A list of words.

    Return
    ------
    A subset of the <words> list based on the filter information from <wordle_info>.

    Examples
    --------
            Input : (winfo, d)   = create_wordle_info("which", "where")
                    words        = ["state", "which", "where", "child", "there", "taste"]
            Output: (winfo, d)   = ([('w', 1), ('h', 2)], Dict('h' => (0, 0), 'c' => (0, 0), 'i' => (0, 0)))

            Input : filter_words = filter_universe((winfo, d), words)
            Output: filter_words = 1-element Vector{String}:
                                     "where"
"""
function filter_universe(wordle_info :: Tuple{Vector{Tuple{Char, Int64}}, Dict{Char, Tuple{Int64, Int64}}},
                         words       :: Vector{String}                                                    ,
                        ) :: Vector{String}

    ## Nothing left to filter.
    if length(words) == 0
        return(words)
    end

    ## Destructure the worlde_info, get the length of the words used in word lists.
    (winfo, d) = wordle_info
    word_len = length(words[1])

    ## This is the list of all the indices in any given puzzle word.
    f_idxs:: Vector{Int64} = collect(1:word_len)

    ## Filter words on exact matches...
    e_idxs = map(x -> x[2], winfo)
    if length(e_idxs) > 0
        cstr = String(map(x -> x[1], winfo))
        words = filter(x -> cstr == x[e_idxs], words)
    end

    ## These are the indices of non-exact matches.
    c_idx = setdiff(f_idxs, e_idxs)
    m = length(c_idx)

    ## Adjust filtering based on match flag (d[k][2]).
    if m > 0
        for k in keys(d)
            if d[k][2] == 0
                words = filter(x -> sum(collect(x[c_idx]) .== fill(k, m)) == d[k][1], words)
            else
                words = filter(x -> sum(collect(x[c_idx]) .== fill(k, m)) >= d[k][1], words)
            end
        end
    end

    ## Return the filtered words.
    return(words)
end

"""
    Strategy to pick a guess for Wordle.    
        - Take the words in the current universe.
        - Take the complement of the indices where we have exact information.
          For each of these indices create a dictionary with letter => count.
        - Pick the index where the corresponding dictionary has the largest count value for some letter.
        - If for a given dictionary, there are several letterrs with the same count, pick the letter from the letter freq string below.
        - Do the same now across dictionaries, find the letter that is the most frequent and its dictionary index.
        - For this index, find all words with this letter in this slot. Pick the one that is most frequent.
          This will be our guess.

    Parameters
    ----------

    swords : A Vector of sorted strings (sorted by frequency of occurrence.
    lfa    : This is the alphabet in lower case as a character list from most to least used.
    c_idx  : This is the index values of words to analyze. This list is usually the complement 
             of exact match indices from a previous guess.

    Return
    ------

    A guess word.

    Assumes
    -------

    The characters in swords are lowercase letters: [a-z].

"""
function pick_guess(swords::Vector{String}, # The sorted list of words to choose from. 
                    lfa   ::Vector{Char}  , # The letter frequency order of the alphabet.
                    c_idx ::Vector{Int64} , # The complement of the indices that are exact.
                   ) :: String
    
    ## Create corresponding dictionaries for each index.
    ds = [Dict{Char, Int64}() for i in c_idx]
    ary = []
    
    ## Fill each of the dicts: at index i, ds[i]: char => count (using swords) 
    for i in 1:length(c_idx)
        for word in swords
            ds[i][word[c_idx[i]]] = 1 + get(ds[i], word[c_idx[i]], 0)
        end
    end

    ## Fill the array `ary` with tuples of the form: (idx, char, num_of_occurrences, lfa_order)
    for i in 1:length(c_idx)
        mx = maximum(values(ds[i]))
        for (k,v) in ds[i]
            if v == mx
                push!(ary, (c_idx[i], k, v, (findall(x -> x == k, lfa))[1]))
            end
        end
    end

    ## Sort `ary` by occurrence followed by `lfa` order.
    sary = sort(ary, lt=((x,y) -> (x[3] < y[3]) | (x[3] == y[3] & (x[4] > y[4]))), rev=true)
    
    ## Get the index and character of the most frequent/most-used character.
    idx  = sary[1][1]
    c    = sary[1][2]

    ## Return the first word(which is sorted by frequency of occurrence) which has character `c` at index, `idx`.
    return((filter(x -> x[idx] == c, swords))[1])
end


"""
    Solves a Wordle puzzle.
    Makes guesses based on the most frequently used word in the uniniverse.
    ASSUMES: The universe DataFrame is sorted from highest frequency to lowest.

    Parameters
    ----------

    puzzle_word : The puzzle word.
    universe_df : A DataFrame with schema: word(words of the same length), freq(freq fraction by use)
                  *NOTE:* The universe is assumed to be sorted in reverse order by the :freq column.
    rec_count   : The number of calls to this function.
    sol_path    : Vector containing the current list of guesses: [ (guess, exact_info, universe_size) ...]
    last_guess  : The previous guess.
    lfa         : The lowercase alphabet listed in frequency of use order.

    Key_Word_params
    ---------------

    chk_inputs     : If true, check the input contract.
    guess_strategy : If not `nothing`, apply this function to pick the next guess.
                     If `nothing`, pick the next guess as the most frequent word
                     in the current universe.
    Here,
     - <exact_info> has the form: [(LETTER, POSITION) ...]
     - <universe_size> is the size the word list when <guess> was made.

    Return
    ------
    (sol_path, number-of-guesses, :SUCCESS/:FAILURE)

    Examples
    --------
            Input : solve_wordle("taste")
            Output: (Any[("which", Tuple{Char, Int64, Char}[], 3034),
                         ("about", Tuple{Char, Int64, Char}[], 1382),
                         ("after", Tuple{Char, Int64, Char}[], 133),
                         ("state", [('t', 4, 'E'), ('e', 5, 'E')], 44),
                         ("taste", [('t', 1, 'E'), ('a', 2, 'E'), ('s', 3, 'E'), ('t', 4, 'E'), ('e', 5, 'E')], 2)
                        ], 5, :SUCCESS)

    INPUT CONTRACT
    --------------
    1. universe_df, schema is (:word, :freq)
    2. ∃ N,m > 0, ∀ i∈[1,N], |words| = N ∧|words[i]| = m
    3. ∃ N > 0  ,            words = words[argsort[universe_df[:freq]]]
"""
function solve_wordle(puzzle_word :: String                      , # Puzzle word.
                      universe_df :: DataFrame     = WORDLE_DF   , # Wordle database as DataFrame.
                      rec_count   :: Int64         = 1           , # Number of calls to this function.
                      sol_path    :: Vector{Any}   = []          , # The solution path of guessed words, so far.
                      last_guess  :: String        = ""          , # The last guess.
                      lfa         :: Vector{Char}  = LFA         ; # The frequency of use of the alphabet.
                      chk_inputs  :: Bool          = true        , # Do we check the input contract?
                      guess_strategy               = nothing     , # Function to pick the next guess.
                     ):: Tuple{Any, Int64, Symbol}

    ## Check input contract?
    if chk_inputs
        ## 1. Does <universe_df> have the correct schema?
        @assert(Set(names(universe_df)) == Set(["word", "freq"]))

        ## 2. Do :words from <universe_df> have the same length?
        words = universe_df[!, :word]
        sidx = sortperm(universe_df[!, :freq], rev=true)
        dw = Dict{String, Int64}()
        for word in words
            dw[word] = 1 + get(dw, word, 0)
        end
        @assert(length(values(dw)) > 1)
        dw = nothing # Set for garbage collection.

        ## 3. Is <universe_df> sorted from hightest to lowest word usage?
        @assert(words[sidx] == words)
    end

    ## Get a copy of the word universe.
    univs = universe_df[!, :word]

    ## Current guessing strategy is to take the most frequently used word in the current universe.

    guess    = univs[1]
    if last_guess != ""
        univs = filter(x -> x != last_guess, univs)
        if length(univs) == 0
            return((sol_path, rec_count  , :FAILURE))
        end
        guess = univs[1]
    end
    word_len = length(guess)

    ## If we specified a picking strategy, modify the guess.
    ## The strategy is based on:
    ## 1. The existing universe
    ## 2. The letter frequency order.
    ## 3. The indices to focus on.
    if guess_strategy !== nothing
        if length(sol_path) != 0
            exact_info = sol_path[end][2]
            if length(exact_info) != 0
                f_idx = collect(1:word_len)
                e_idx = map(x -> x[2], exact_info)
                c_idx = setdiff(f_idx, e_idx)
                guess = guess_strategy(univs, lfa, c_idx)
            end
        end
    end

    ## Get the Wordle match info:
    ##  Exact match list: [(LETTER, POSITION)...]
    ##  Dictionary with info about letters that are not exact matches:
    ##    LETTER => (k,n)  k : The number of matches out of position.
    ##                         A value of 0 means that the letter is not in the puzzle.
    ##                     n : 0|1 If 0 there are    *exactly*  k matches out of position.
    ##                             If 1 there are    *at least* k matches out of position.
    (exact_info, ino_dct) = create_wordle_info(guess, puzzle_word)

    ## Get the size of the current search universe.
    ## Push the guess and it's exact match info on the `sol_path`.
    n = length(univs)
    push!(sol_path, (guess, exact_info, n))

    ## if we guessed the puzzle word, return success.
    if guess == puzzle_word
        return((sol_path, rec_count, :SUCCESS))
    end

    ## Filter the current universe based on the match info to get the new universe.
    new_universe = filter_universe((exact_info, ino_dct), univs)

    ## Look at the size of the new universe -- we can make conclusions in some instances.
    n = length(new_universe)
    if n == 0 # The information does not lead to a solution -- the puzzle word is not in our initial universe.
        return((sol_path, rec_count  , :FAILURE))
    elseif n == 1 && puzzle_word == new_universe[1] # We know the solution without having to recurse again.
        return((sol_path, rec_count, :SUCCESS))
    elseif n == 1 # The puzzle word is not in our initial universe.
        return((sol_path, rec_count+1, :FAILURE))
    end

    ## If we recursed too much, there must be an error.
    if rec_count > (10 * word_len)
        return((sol_path, rec_count, :FAILURE))
    end

    ## Get the new universe as a dataframe and sort it based on frequency of occurrence from hightest to lowest.
    nuniv_df = filter(:word => x -> x in new_universe, universe_df)
    sort!(nuniv_df, order(:freq, rev=true))

    ## Recurse...
    solve_wordle(puzzle_word, nuniv_df, rec_count+1, sol_path, guess, lfa; chk_inputs=false, guess_strategy = guess_strategy)
end


end # module Wordle
