module Wquantiles

export wquantile, Wquantile, WquantileM

import Folds

## Make a sortable Trait.
function isSortable(::Type{T}) :: Bool where {T <: Number}
    x = one(T)
    try
       isless(x, x)
    catch _ 
        return(false) 
    end
    return(true)
end

"""
    wquantile(x, w, q[; chk=true, norm_wgt=true, sort_q=true])

Finds the `q` weighted quantile values from the vector `x`.

## Type Constraints
- `S <: Real`
- `V <: Real`

## Arguments
- `x  ::Vector{T}`: Vector(n) of values from which to find quantiles.
- `w  ::Vector{S}`: Vector(n) of weights to use.
- `q  ::Vector{V}`: Vector(l) of quantile values.

## Keyword Args
- `chk     ::Bool`: If `true`, check the input contract described below.
- `norm_wgt::Bool`: If `true`, normalize the weights.

   **NOTE:** If `norm_wgt` is `false`, it is *ASSUMED* that `w` is already normalized.
- `sort_q  ::Bool`: If `true`, sort the quantile vector, `q`.

   **NOTE:** If `sort_q` is `false`, it *ASSUMED* that `q` is already sorted.

### Input Contract
- The type of `x` implements sortable.
-    `|x|  = |w|`     -- Length of `x` matches length of weights.
- `∀i,  w[i] >= 0`    -- The weights are non-negative.
-    `Σ w[i] >  0`    -- The sum of the weights is positive.
- `∀i,  q[i] <= 1`    -- The quantile values are in ``[0,1]``.
- `∀i,  q[i] >= 0`

## Return
The vector(l) of weighted quantile values from `x`.
Letting `qs` be the sorted quantiles of `q`.
The entry `i` is the ``i^{\\rm th}`` quantile (in `qs`) of `x`.

"""
function wquantile(x::Vector{T}, w::Vector{S}, q::Vector{V};
                   chk::Bool = true, norm_wgt::Bool = true, sort_q::Bool = true) :: Vector{T} where {T, S <: Real, V <: Real}
    ## We report back the quantiles of `x` in sorted `q` order, so we need to sort `q`.
    ## **NOTE:** If we don't explicitly sort `q`, it means that you are *ASSUMING* `q` is sorted.
    if sort_q
      q = sort(q)
    end
    m  = length(q)
    n  = length(x)

    ## Get 0 and 1 for `q` types.
    zeroq = zero(eltype(q[1]))
    oneq  = one(eltype(q[1]))

    ## Check input contract...
    if chk
        @assert(isSortable(T))               #  x sortable?
        @assert(n == length(w))              # |x| = |w| ?
    end

    ## Get the permutation of indices that sort `x`.
    idx = sortperm(x)

    ## Convert sorted weights.
    @inbounds wsc = convert(Vector{promote_type(eltype(w[1]), eltype(q[1]))}, w[idx])

    ## Check input contract...
    if chk
        @assert(all(wsc   .>= zeroq)      )  # ∀i,      w[i] >= 0  -- (Weights are non-negative?)
        @assert(sum(wsc)    > zeroq       )  #        Σ w[i]  > 0  -- (The sum of the weights is positive?)
        @assert(all(zeroq .<= q .<= oneq) )  # ∀i, 0 <= q[i] <= 1  -- (Quantile values are in [0,1]?)
    end

    ## Normalize sorted weights?
    if norm_wgt
      wsc ./= sum(wsc)
    end

    ## Apply permutation to `x`.
    @inbounds xs = x[idx]

    ## Create an index vector to get the list of quantiles of `x`.
    ## Default the indices to the largest element of `x`.
    ## Why is this important: If one chooses 1.0 as a quantile, it could easily be
    ##                        the case, due to numeric inaccuracy, that we do not
    ##                        reach the 1.0 threshold. For this reason we want
    ##                        to pick the default index value to
    ##                        be the largest index in `x`, so if the threshold
    ##                        is not reached we do the right thing.
    qxsi = fill(n, m)

    ## Using the fact that the quantile values are in sorted order,
    ## find the index for each associated value in `xs`, placing them in `qxsi`.
    j = 1
    s = zeroq
    for i in 1:n
        ## If we exceed the current quantile threshold, `s`.
        ## Set the index at `j` of the index vector.
        if s >= q[j]
            @inbounds qxsi[j] = i
            while true
                j += 1
                if j == m+1
                    @goto done
                end
                if q[j] > s
                    break
                end
                @inbounds qxsi[j] = i
            end
        end
        ## Finished with all quantiles that hit the quantile threshold, `s`.
        ## Now update the threshold.
        @inbounds s += wsc[i]
    end
    @label done

    ## Return the quantile values (in quantile sorted order).
    @inbounds return(xs[qxsi])
end


"""
    Wquantile(X, w, q[; chk=true, norm_wgt=true, sort_q=true])

Finds the `q` weighted quantile values from the columns of the matrix `X`.

## Type Constraints
- `S <: Real`
- `V <: Real`

## Arguments
- `X  ::Matrix{T}`: Matrix(n,m) of values from which to find quantiles.
- `w  ::Vector{S}`: Vector(n) of weights to use.
- `q  ::Vector{V}`: Vector(l) of quantile values.

## Keyword Args
- `chk     ::Bool`: If `true`, check the input contract described below.
- `norm_wgt::Bool`: If `true`, normalize the weights.

   **NOTE:** If `norm_wgt` is `false`, it is *ASSUMED* that `w` is already normalized.
- `sort_q  ::Bool`: If `true`, sort the quantile vector, `q`.

   **NOTE:** If `sort_q` is `false`, it is *ASSUMED* that `q` is already sorted.
### Input Contract
-  The type of `X` implements sortable.
- `∀i, |X[:, i]|   = |w|` -- Length of each column of `X` matches length of weights.
- `∀i,      w[i]  >= 0`   -- Weights are non-negative.
- `Σ w[i]  >  0`          -- The sum of the weights is positive.
- `∀i,      q[i]  <= 1`   -- The quantiles values are in ``[0,1]``.
- `∀i,      q[i]  >= 0`

## Return
The `(l,m)` matrix of weighted quantile values from `X`.
Letting `qs` be the sorted quantiles of `q`.
The entry `(i,j)` is the ``i^{\\rm th}`` quantile (in `qs`) from the ``j^{\\rm th}`` column of `X`.

"""
function Wquantile(X::Matrix{T}, w::Vector{S}, q::Vector{V};
                   chk::Bool = true, norm_wgt::Bool = true, 
                   sort_q::Bool = true) :: Matrix{T} where {T, S <: Real, V <: Real}

    _, m = size(X)

    ## Normalize the weights if needed.
    if norm_wgt
        w = convert(Vector{promote_type(eltype(w[1]), eltype(q[1]))}, w)
        w ./= sum(w)
    end

    ## Sort the quantiles if needed.
    if sort_q
        q = sort(q)
    end

    ## Create a closure that will be threaded -- computing the weighted quantiles of the columns of `X`.
    ## If `chk` is true, only do the input check for the first column
    ## as checking the rest of the columns is redundant.
    wquant_vec_func = p -> wquantile(p[1], w, q, 
                                     chk=p[2]==1 ? chk : false, 
                                     norm_wgt=false, sort_q=false)

    ## Computation: (from right to left)
    ## - Zip up the columns of `X` along with the column number.
    ## - Use Folds.map to apply multiple threads to compute the weighted quantiles on each column of `X`.
    ## - Place them back as an array using reduce hcat.
    return(reduce(hcat, Folds.map(wquant_vec_func, 
                                  zip([X[:, i] for i in 1:m], 1:m))))

end


"""
    WquantileM(X, w, q[; chk=true])

Finds the `q` weighted quantile values from the columns of the matrix `X`.

## Type Constraints
- `S <: Real`
- `V <: Real`

## Arguments
- `X  ::Matrix{T}`: Matrix(n,m) of values from which to find quantiles.
- `w  ::Vector{S}`: Vector(n) of weights to use.
- `q  ::Vector{V}`: Vector(l) of quantile values.

## Keyword Args
- `chk::Bool`     : If `true`, check the input contract described below.

### Input Contract
-  The type of `X` is sortable.
- `∀i, |X[:, i]|  = |w|` -- Length of each column of `X` matches length of weights.
- `∀i,      w[i] >= 0`   -- Weights are non-negative.
- `Σ w[i] >  0`          -- The sum of the weights is positive.
- `∀i, 0 <= q[i] <= 1`   -- The quantiles values are in ``[0,1]``.

## Return
The `(l,m)` matrix of weighted quantile values from `X`.
Letting `qs` be the sorted quantiles of `q`.
The entry `(i,j)` is the ``i^{\\rm th}`` quantile (in `qs`) from the ``j^{\\rm th}`` column of `X`.

"""
function WquantileM(X::Matrix{T}, w::Vector{S}, q::Vector{V}; 
                    chk::Bool = true) :: Matrix{T} where {T, S <: Real, V <: Real}
    ## We report back the quantiles of `X` in sorted `q` order, so we sort `q`.
    qs    = sort(q)
    l     = length(qs)
    n, m  = size(X)

    ## Get 0 and 1 for `q` types.
    zeroq = zero(eltype(qs[1]))
    oneq  = one(eltype(qs[1]))

    ## Check input contract...
    if chk
        @assert(isSortable(T))               # x sortable?
        @assert(n == length(w))              # ∀i, |X[:,i]| = |w| ?
    end

    ## Convert weights.
    wc = convert(Vector{promote_type(eltype(w[1][1]), eltype(q[1]))}, w)

    ## Check input contract...
    if chk
        @assert(all(wc     .>= zeroq         )  )  # ∀i,       wc[i]  >= 0 -- (non-neg weights?)
        @assert(sum(wc)    .>  zeroq            )  #         Σ wc[i]  >  0 -- (sum of weights is positive?)
        @assert(all(zeroq  .<= q     .<= oneq)  )  # ∀i,  0 <= q[i]   <= 1 -- (quantile values are in [0,1]?)
    end

    ## Get the permutation of indices that sort the columns of `X`. This will be used to create
    ## a matrix of permutations of `wc` that align with this sorting.
    Idx = sortperm(X; dims=1)

    ## Convert the indices to column specific indices that `wc` understands
    ## and sort the columns of `wc` as columns of `X` are sorted.
    @inbounds Wsc = wc[(Idx .- 1) .% n .+ 1]

    ## Normalize sorted weights by column.
    Wsc ./= sum(Wsc, dims=1)

    ## Apply permutation to `X` -- sorting each column of `X`.
    @inbounds Xs = X[Idx]

    ## Create an index matrix to get the list of quantiles of `X`.
    ## Default the indices to the index of the largest element of X for each column.
    ## Why is this important: If one chooses 1.0 as a quantile, it could easily be
    ##                        the case, due to numeric inaccuracy, that we do not
    ##                        reach the 1.0 threshold. For this reason we want
    ##                        to pick the default index value for each column to
    ##                        be the largest index in the column, so if the threshold
    ##                        is not reached we do the right thing.
    Qxsi = Array{Int64}(undef, l, m)
    for i in 1:m
      @inbounds Qxsi[:, i] .= i * n
    end

    ## Using the fact that the quantile values are in sorted order,
    ## find the index for each associated value in `Xs`, placing them in `Qxsi`.

    ## For each column...
    for k in 1:m
        j = 1
        s = zeroq

        ## For each row...
        for i in 1:n
            ## If we exceed the current quantile threshold, `s`.
            ## Set the index at (`j`,`k`) of the index matrix.
            if s >= qs[j]
                @inbounds Qxsi[j,k] = i + (k-1) * n
                if j == l
                    break
                end
                while true
                    j += 1
                    if j == l+1
                        @goto column_done
                    end
                    if qs[j] > s
                        break
                    end
                    @inbounds Qxsi[j,k] = i + (k-1) * n
                end
            end

            ## Finished with all quantiles that hit the quantile threshold, `s`.
            ## Now update the threshold.
            @inbounds s += Wsc[i, k]
        end

        ## We've finished off a column, onto the next.
        @label column_done
    end

    ## Return the quantile values as an (`l`,`m`) matrix in quantile sorted order.
    @inbounds return(Xs[Qxsi])
end

end # module Wquantiles

