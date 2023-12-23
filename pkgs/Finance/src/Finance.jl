module Finance

import Random, Distributions, Statistics


export sig_cumsum, tic_diff1, tic_diff2, isConvertible 
export ema, ema_std, ema_stats, std, WWsum
export entropy_index, pow_n


"""
    isConvertible(S, T)

Boolean function which returns `true` if a value of type `S` 
can be converted to a value of type `T`.

# Type Constraints
- `S <: Real`
- `T <: Real`

# Arguments
- `::Type{S}` -- A numeric type.
- `::Type{T}` -- A numeric type.

# Return
`::Bool`
"""
function isConvertible(::Type{S}, ::Type{T})  where {S <:Real, T <:Real}
    try
        convert(one(S), one(T))
    catch _
        return(false)
    end

    return(true)
end


"""
    tic_diff1(t, x; chk_inp=false)

Compute the numerical derivative of a function represented by `x`
with respect to `t` when the values in `t` are possibly irregular.

# Type Constraints
- `S <: Real`
- `T <: Real`

# Arguments
- `t :: AbstractVector{T}`   -- A vector of times.
- `x :: AbstractVector{T}`   -- A vector of values.

# Keyword Arguments
- `chk_inp=false :: Bool`  -- Check the input contract?

# Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint                     | Description                                                               |
|:------------------------------:|:------------------------------------------------------------------------- |
| `\\|t\\| = \\|x\\|`            | The length of the time and data vectors match.                            |
| `S => T`                       | Type `S` can be converted to type `T`.                                    |
| ``\\forall i, t_{i+1} > t_i``  | The times are increasing; consequently, we have a 1-1 map from `t` to `x`.|

# Return
`:: AbstractVector{T}`
"""
@noinline function tic_diff1(t::AbstractVector{S} , 
                             x::AbstractVector{T} ;
                             chk_inp::Bool = false,
                  ) :: AbstractVector{T} where {S <: Real, T <: Real}
    n = length(x)

    if chk_inp
        n != length(t)           && throw(DomainError(n-t, "The length of the time and data series must match."))
        !all(diff(t) .> zero(S)) && throw(DomainError(n-t, "The time series must have be strictly increasing."))
        !isConvertible(S, T)     && throw(DomainError(0, "Type S is not convertible to type T."))
    end

    tc = map(x -> convert(T, x), t)
    df = zeros(T, n-2)
    @simd for i in 2:(n-1)
        @inbounds h1 = tc[i  ] - tc[i-1]
        @inbounds h2 = tc[i+1] - tc[i  ]
        @inbounds df[i] = (x[i+1] - x[i-1]) / (h1 + h2)
    end

    return(df)
end

"""
    tic_diff2(t, x; chk_inp=false)

Compute the numerical second derivative of a function represented by `x`
with respect to `t` when the values in `t` are possibly irregular.

# Type Constraints
- `S <: Real`
- `T <: Real`

# Arguments
- `t :: AbstractVector{S}` -- A vector of times.
- `x :: AbstractVector{T}` -- A vector of values.

# Keyword Arguments
- `chk_inp=false :: Bool`  -- Check the input contract?

# Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint                     | Description                                                               |
|:------------------------------:|:--------------------------------------------------------------------------|
| `\\|t\\| = \\|x\\|`            | The length of the time and data vectors match.                            |
| `S => T`                       | Type `S` can be converted to type `T`.                                    |
| ``\\forall i, t_{i+1} > t_i``  | The times are increasing; consequently, we have a 1-1 map from `t` to `x`.|

# Return
`:: AbstractVector{T}`
"""
@noinline function tic_diff2(t::AbstractVector{S}, 
                   x::AbstractVector{T}          ;
                   chk_inp::Bool = false         ,
                  ) :: AbstractVector{T} where {S <: Real, T <: Real}
    n = length(x)

    if chk_inp
        n != length(t)           && throw(DomainError(n-t, "The length of the time and data series must match."))
        !all(diff(t) .> zero(S)) && throw(DomainError(n-t, "The time series must have be strictly increasing."))
        !isConvertible(S, T)     && throw(DomainError(0, "Type S is not convertible to type T."))
    end

    tc = map(x -> convert(T, x), t)
    df = zeros(T, n-2)
    @simd for i in 2:(n-1)
        @inbounds h1 = tc[i  ] - tc[i-1]
        @inbounds h2 = tc[i+1] - tc[i  ]
        @inbounds df[i] = (h2 * x[i+1] - (h1 + h2) * x[i] + h1 * x[i-1]) / (h1 * h2 * (h1 + h2))
    end

    return(df)
end



"""
    sig_cumsum(t, x, w, h; chk_inp=false)

Return a tuple of two vectors: tics, signals.

The signals are the collections of all deviations from 
a running mean (with window length `w`) of the series. 
The threshold of the deviation is `h`.

Deviation is determined by:
- ``S_t^+ = {\\rm max}(0, S_{t-1} + x_t - E[x_{t-1}]; S^+_0 = 0``
- ``S_t^- = {\\rm min}(0, S_{t-1} + x_t - E[x_{t-1}]; S^-_0 = 0``
- ``S_t^{\\hphantom{+}} = {\\rm max}(S^+_t, -S^-_t)``

Collect all ``t, S_t`` where ``h \\ge S_t``.

# Type Constraints
- `S <: Real`
- `T <: Real`

# Arguments
- `t :: AbstractVector{S}` -- The tic series to examine.
- `x :: AbstractVector{T}` -- The series to examine.
- `w :: Int64`             -- The width of the moving average.
- `h :: T`                 -- The threshold for the deviation to register.

# Keyword Arguments
- `chk_inp=false :: Bool`  -- Check the input contract?

# Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint                          | Description                                                               |
|:-----------------------------------:|:--------------------------------------------------------------------------|
|`\\|x\\|` ``\\ge`` `2`               | The length of x is at least ``2``.                                        | 
|`\\|x\\| = \\|t\\|`                  | The length of `x` is equal to the length of `t`.                          |
|`w > 1`                              | The window length is greater than ``1``.                                  |
|`h > 0`                              | The deviation threshold is greater than 0.                                |
|``\\forall i, t_{i+1} > t_{i}``      | The times are increasing; consequently, we have a 1-1 map from `t` to `x`.|

# Output Components
- `td :: AbstractVector{S}` -- Values of `t` where deviations occurred.
- `xd :: AbstractVector{T}` -- Values of `x` where deviations occurred.

# Output Contract
- `|td| = |xd|`

# Return
`(td, xd) :: Tuple{AbstractVector{S}, AbstractVector{T}}`
"""
function sig_cumsum(t::AbstractVector{S}, 
                    x::AbstractVector{T}, 
                    w::Int64            , 
                    h::T                ;
                    chk_inp::Bool=false ,
                   ) :: Tuple{AbstractVector{S}, AbstractVector{T}} where {S <: Real, T <: Real}
    n = length(x)

    ## Input contract.
    if chk_inp
        nt = length(t)
        n < 2                    && throw(DomainError(n, "Vector length must be >= 2."))
        w <= 1                   && throw(DomainError(n, "Window length must be > 1."))
        h <= zero(T)             && throw(DomainError(n, "Devitation threshold must be > 0."))
        n != length(t)           && throw(DomainError(nt, "Length of time sequence should match length of data."))
        !all(diff(t) .> zero(S)) && throw(DomainError(0, "Sequential differences of time seq must always be > 0."))
    end

    Sp = zeros(T, n)
    Sn = zeros(T, n)
    z  = zero(T)

    xm   = x[1]  # Running mean of the input `x` computed based on window, `w`.
    sigs = T[]   # The signals/deviations to be returned. 
    tics = S[]   # The tics where the deviations occurred.

    ## Loop over the series and populate, `tics` and `sigs`.
    for i in 2:n
        @inbounds xm = ( (w - 1) * xm + x[i-1] ) / w
        @inbounds Sp[i] = max(z, Sp[i-1] +  x[i]  - xm)
        @inbounds Sn[i] = min(z, Sp[i-1] +  x[i]  - xm)
        @inbounds delta = max(Sp[i], -Sn[i])
        if delta >= h
            push!(sigs, delta) 
            @inbounds push!(tics, t[i])
        end
    end

    return((tics, sigs))
end



"""
    ema(x,m; h=div(m,2))

Compute the Exponential Moving Average of the sequence `x`.

# Type Constraints
- `T <: Real`

# Arguments
- `x :: AbstractVector{T}` -- The series to work with.
- `m :: Int64`             -- The width of the decay window.

# Keyword Arguments
- `h=div(m,2) :: Int64`       -- The exponential decay *half-life*. 

# Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint | Description                                   | 
|:----------:|:----------------------------------------------|
| `m > 1`    | Averaging window length is greater than ``1``.|
| `h > 1`    | Exponential *half-life* is greater than ``1``.|

# Output 
- `ema :: AbstractVector{T}` -- The exponential moving average of `x`.

# Output Contract
- `|x| = |ema|`

# Return
`ema::AbstractVector{T}`

"""
@noinline function ema(x :: AbstractVector{T}          , 
                       m :: Int64                      ; 
                       h = div(m, 2) :: Int64          ,
                      ) :: AbstractVector{T} where { T <: Real }

    ## Check input constraints.
    m <= 1 && throw(DomainError(m, "The window length must be > 1."))
    h <= 1 && throw(DomainError(h, "The half-life must be > 1."))

    N = length(x)
    ma = zeros(T, N)
    xadj = zeros(T, N+m)
    @inbounds xadj[1:m] .= x[1]
    @inbounds xadj[(m+1):end] = x
    w  = zeros(T, m)

    ## Term by term decay factor.
    l = exp(-log(2 * one(T)) / h)

    w[1] = l
    @simd for i in 2:m
        @inbounds w[i] = l * w[i-1]
    end
    w ./= sum(w)

    ## Compute the EMA using the difference equation recursion.
    ma[1] = xadj[m+1]
    @simd for i in 2:N
        @inbounds ma[i] =  l * (ma[i-1] - w[m] * xadj[i]) + w[1] * xadj[i+m]     
    end

    return(ma)
end


"""
    ema_std(x, m; h=div(m,2), init_sig=nothing)

Compute the Moving Exponential Standard Deviation of the sequence `x`.
By default, the initial std is taken to be the standard deviation of 
the first window (of length `m`). However, a user specified value
may be used instead.

# Type Constraints
- `T <: Real`

# Arguments
- `x :: AbstractVector{T}` -- The series to work with.
- `m :: Int64`             -- The width of the decay window.

# Keyword Arguments
- `h=div(m,2)       :: Int64`             -- The exponential decay *half-life*. 
- `init_sig=nothing :: Union{T, Nothing}` -- An optional user supplied initial standard deviation for the start of the series.      

# Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint           | Description                                             |   
|:--------------------:|:------------------------------------------------------- |
| `m > 1`              | Averaging window length is greater than ``1``.          |
| `h > 1`              | Exponential *half-life* is greater than ``1``.          |
| `\\|x\\| > 1`        | The length of the series is greater than ``1``.         |
| `init_sig` ``\\ge 0``| User supplied starting ``\\sigma`` should be ``\\ge 0``.|

# Output 
- `stda :: AbstractVector{T}` -- The moving exponential standard deviation of `x`.

# Output Contract
- `|x| = |stda|`

# Return
`stda::AbstractVector{T}`
"""
@noinline function ema_std(x                  :: AbstractVector{T},
                           m                  :: Int64            ;
                           h = div(m, 2)      :: Int64            ,
                           init_sig = nothing :: Union{T, Nothing},
                )  :: AbstractVector{T} where {T <: Real}

    N = length(x)

    ## Check input constraints.
    N <= 1                                      && throw(DomainError(N, "The length of the data series must be > 1."))
    m <= 1                                      && throw(DomainError(m, "The window length must be > 1."))
    h <= 1                                      && throw(DomainError(h, "The half-life must be > 1."))
    typeof(init_sig) == T && init_sig < zero(T) && throw(DomainError(init_sig, "The initial sigma must be non-negative."))

    ## Compute the ema for `x`.
    ma = ema(x, m, h=h) 

    ## Variance estimates.
    mvar = zeros(T, N)

    ## Set the initial estimated/supplied variance.
    mvar[1] = init_sig !== nothing ? init_sig : std(x[1:min(m,N)])
    mvar[1] *= mvar[1]

    ## Add history (`m` zeros) for variance.
    ## We do this by augmenting the length of the "x"'s -- `(x - ma)^2`
    ## to have size `N + m`, so we can go "back" m. This means that
    ## xadj has to be indexed differently than the way 
    ## the formula does indexing.
    xadj = zeros(T, N+m)

    ## We don't need the following line (like what we have in the corresponding ema code)
    ## as `(x - ma)[1]` = 0, and the xadj array is already set to 0.
    ## @inbounds xadj[1:m] .= (x - ma)[1] * (x - ma)[1] 
    @inbounds xadj[(m+1):end] = (x - ma) .* (x - ma)
    w  = zeros(T, m)

    ## Term by term decay factor.
    l = exp(-log(2 * one(T)) / h)

    ## Use this to define the weights; then normalize.
    ## Weights go from large to small.
    w[1] = l
    @simd for i in 2:m
        @inbounds w[i] = l * w[i-1]
    end
    w ./= sum(w)
    w2 = sum(w .* w)

    ## Recursive formula for variance.
    @simd for n in 1:(N-1)
        @inbounds mvar[n+1] = l * (mvar[n] - xadj[n+1] * w[m]) + xadj[n+m+1] * w[1] 
    end

    ## Return corrected variances (unbiased).
    return(sqrt.(mvar ./ (one(T) - w2)))
end


"""
    ema_stats(x, m; h=div(m,2), init_sig=nothing)

Compute the Moving Exponential Stats of the sequence `x`: `ema`, `ema_std`, `ema_rel_skew`, `ema_rel_kurtosis`.

The recursive formulas for the moving statistics as well as the adjustments necessary to 
render the estimates *unbiased* come from the paper:
[exponential\\_moving\\_average.pdf](https://github.com/scottrsm/math/tree/main/pdf/exponential_moving_average.pdf).

Returns these stats as a matrix with four columns, each representing the stats above in the order listed.

# Type Constraints
- `T <: Real`

# Arguments
- `x :: AbstractVector{T}` -- The series to work with.
- `m :: Int64`             -- The width of the decay window.

# Keyword Arguments
- `h=div(m,2) :: Int64`     -- The exponential decay *half-life*. 
- `init_sig=nothing:: Union{T, Nothing}` -- An optional user supplied initial standard deviation for the start of the series.      

# Input Contract
The inputs are assumed to satisfy the constraints below:

| Constraint     | Description                                    | 
|:--------------:|:---------------------------------------------- |
| `m > 1`        | Averaging window length is greater than ``1``. |
| `h > 1`        | Exponential *half-life* is greater than ``1``. |
| `\\|x\\| > 3`  | The length of the series is greater than ``3``.|

# Output 
- `stat :: Matrix{T}` -- A matrix of EMA stats: `ema`, `ema_std`, `ema_rel_skew`, `ema_rel_kurtosis`.

# Output Contract
- `|stat| = (N, 4)` 

# Return
`stat::Matrix{T}`
"""
@noinline function ema_stats(x      :: AbstractVector{T},
                   m                :: Int64            ;
                   h=div(m,2)       :: Int64            ,
                   init_sig=nothing :: Union{Nothing, T},
                  ) :: Matrix{T} where {T <: Real}
    N = length(x)
    
    ## Check input constraints.
    m <= 1                                      && throw(DomainError(m, "The window length must be > 1."))
    h <= 1                                      && throw(DomainError(h, "The half-life must be > 1."))
    N <= 3                                      && throw(DomainError(N, "N must be > 3."))
    typeof(init_sig) == T && init_sig < zero(T) && throw(DomainError(init_sig, "The initial sigma must be non-negative."))

    ## Compute the EMA of `x`.
    ma = ema(x, m, h=h) 

    mstat = zeros(T, (N,4))
    mstat[1, 1] = x[1] 

    ## Set the initial estimated/supplied variance.
    mstat[1, 2]  = init_sig !== nothing ? init_sig : std(x[1:min(m,N)])
    mstat[1, 2] *= mstat[1, 2]

    ## Add history (`m` zeros) for variance, etc.
    ## We do this by augmenting the length of the "x"'s -- `(x - ma)^2`, `(x - ma)^3`, etc.
    ## to have size `N + m`, so we can go "back" `m`. This means that
    ## `xadj` has to be indexed differently than the way 
    ## the formula does indexing.
    xadj  = zeros(T, (N+m,4))
    v = (x - ma).*(x - ma)
    @inbounds xadj[1:m, 1]       .= x[1]
    @inbounds xadj[(m+1):end, 1] .= x
    @inbounds xadj[(m+1):end, 2] .= v
    @inbounds xadj[(m+1):end, 3] .= v.*(x - ma) 
    @inbounds xadj[(m+1):end, 4] .= v.*v
    w  = zeros(T, m)

    ## Term by term decay factor.
    l = exp(-log(2 * one(T)) / h)

    w[1] = l
    for i in 2:m
        @inbounds w[i] = l * w[i-1]
    end
    w ./= sum(w)

    ## Compute the sums of `w` to powers from 2 to 5.
    W2 = zero(T)
    W3 = zero(T)
    W4 = zero(T)
    W5 = zero(T)
    @simd for i in 1:m
        @inbounds wt = w[i]
        w2 = wt * wt
        W2 += w2
        W3 += w2 * wt
        W4 += w2 * w2
        W5 += w2 * W3
    end
    WW = WWsum(w)

    ## Expressions needed to unbias our estimates.
    C1 = 6 * W2 * W5 - 6 * W2 + 12 * W2^2 - 12 * W2 * W4 + W2 * W3 - W5 - 6 * WW 
    C2 = 1 - 3 * W2 + 6 * W3 - 3 * W4

    ## Recursion to compute the moving stats.
    for i in 1:4
        @simd for n in 1:(N-1)
            @inbounds mstat[n+1, i] = l * (mstat[n,i] - xadj[n+1,i] * w[m]) + xadj[n+m+1,i] * w[1] 
        end
    end

    ## Unbias the estimates.
    mstat[:, 2] ./= one(T) - W2
    mstat[:, 2]   = sqrt.(mstat[:, 2])
    mstat[:, 3] ./= ( mstat[:, 2].^1.5 .* (one(T) - 3*W2 + 2*W3) )
    mstat[:, 4] = ( mstat[:, 4] ./ mstat[:, 2].^2 .+ C1 ) ./ C2

    return(mstat)
end


"""
    std(x)
Compute the "sample" standard deviation of a series, `x`.

# Type Constraints
- `T <: Real`

# Arguments
- `x :: AbstractVector{T}` -- The series to work with.

# Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint     | Description                                    | 
|:--------------:|:---------------------------------------------- |
| `\\|x\\| > 1`  | The length of the series is greater than ``1``.|

# Return
`std::T` -- The sample standard deviation.
"""
@noinline function std(x::AbstractVector{T}) where {T <: Real}
    sd = zero(T)
    mn = zero(T)
    N = length(x)
    @simd for i in 1:N
        @inbounds mn += x[i]
    end
    mn /= N

    @simd for i in 1:N
        @inbounds sd += (x[i] - mn) * (x[i] - mn)
    end
    return( sqrt(sd / (N-1)) )
end

"""
    WWsum(w)

Computes the sum: ``\\sum_{i=1}^{N-1} \\sum_{j=i+1}^N w_i^2 w_j^2``.

Here, `N = |w|`. 

# Type Constraints
- `T <: Real`

# Arguments
- `w :: AbstractVector{T}` -- Vector of weights.

# Return
The above sum.

"""
@noinline function WWsum(w::AbstractVector{T}) :: T  where {T <: Real}
    WW = zero(T)
    m = length(w)
    for i in 1:(m-1)
        @inbounds wwi = w[i] * w[i]
        wwj = zero(T)
        @simd for j in (i+1):m
            @inbounds wwj += w[j] * w[j] 
        end
        WW += wwi * wwj
    end 
    return(WW)
end



"""
    entropy_index(x; <keyword arguments>)

Computes a (Discounted) Binned Entropy Index.
This is the ratio of entropy of the binned distribution of `x` against
the entropy of the uniform distribution.
The vector `x` is first filtered by the lower and upper quantiles; then
binned into `n` number of equal width bins. A distribution is formed 
from the bins and the entropy computed. If `λ` is not 1, then a discounted
entropy is computed. This is an exponentially based discounting of the 
bin distribution based
on their "freshness". In either event, the ratio of this entropy to 
the entropy of the corresponding uniform distribution is returned.

# Type Constraints
- `T <: Real`

# Arguments
- `x::Vector{T}`                        -- Number to exponentiate.

# Keyword Arguments
- `n=10::Int64`                         -- Exponential.
- `tol=1.0/(100 * n)::Float64`          -- Error tolerance used with equivalency test of number to 0 or 1.
- `probs=[0.01, 0.99]::Vector{Float64}` -- Vector of quantile min and max.
- `λ=1.0::Float64`                      -- Discount value.

# Input Contract
- `n > 2`
- `0 < tol < 0.01` 
- `|probs| == 2`
- ``0 < \\lambda \\le 1``

# Return
`::Real` -- The (discounted) binned entropy index.
"""
function entropy_index(x::Vector{T}                ; 
                n::Int64=10                        , 
                tol::Float64=1.0 / (100 * n)       , 
                probs::Vector{Float64}=[0.01, 0.99], 
                λ=1.0                              ) where T <: Real

    # Check Input contract.
    n > 2              || throw(DomainError(n    , "Bad number of bins."))
    0.0 < tol < 0.1    || throw(DomainError(tol  , "Bad tolerance value."))
    length(probs) == 2 || throw(DomainError(probs, "Bad quantile vector, must have length 2."))
    0.0 < λ <= 1.0     || throw(DomainError(λ    , "Bad discount parameter."))

    # Get the data extrema for the quantile filtered data.
    qmin, qmax = Statistics.quantile(x, probs)
    @fastmath xf = filter(x -> qmin <= x <= qmax, x)
    minx, maxx = extrema(xf)

    # This will be the data distribution structure based on the granularity (`n`).
    bdist::Vector{Float64} = fill(0.0, n)
    width = (maxx - minx) / n

    # For each filtered data point assign it to its bin index.
    @fastmath idxs = collect(zip(1:n, Int64.(1.0 .+ (div.(xf .- minx .- tol, width)))))
    sort!(idxs, rev=true)

    # Increment all bins for each occurrence from the series discounted from
    # the end of the time series.
    lm = 1.0
    for (_,j) in idxs
        @inbounds bdist[j] += lm
        lm *= λ
    end

    # Finish off the binned empirical distribution.
    bdist ./= sum(bdist)

    # Get the discounted entropy of the binned distribution.
    ent = 0.0
    @simd for i in 1:n
        @inbounds prb = bdist[i]
        @fastmath ent -= isapprox(prb, 0.0; atol=tol) ? 0.0 : prb * log(prb)
    end

    # Return the normalized discounted binned entropy.
    # Normalize by the entropy of the uniform distribution over `n` values.
    return ent / log(n) 
end


"""
    pow_n(x, n)

Fast (non-negative) integer powers: ``x^n``.
Uses repeated squaring in combination with the bit vector
representation of `n`.

# Type Constraints
- `T <: Number`

# Arguments
- `x::T`     -- The base value.
- `n::Int64` -- The power.

# Input Contract
- ``n \\ge 0`` 

# Return
`::T`        -- The Power Value.
"""
function pow_n(x::T, n::Int64) where T <: Number

    # Check input contract.
    if n < 0
        throw(DomainError(n, "Parameter `n` must be non-negative."))
    end

    o = one(T)

    # Anything to the 0'th power is 1.
    if n == 0
        return(o)
    end

    # -- Do repeated squaring based on the digits of `n-1`. --
    # Initialize values.
    s = x
    n2d = digits(n-1, base=2)

    # Repeated squaring.
    @simd for d in n2d
        @fastmath s *= d == 1 ? x : o
        @fastmath x *= x
    end
    return s
end


"""
    pow_n(x, n, m)

Fast integer (non-negative) powers with modulus: ``x^n \\; {\\rm mod } \\; m``.
Uses repeated squaring in combination with the bit vector
representation of `n`.

The output will be of the type determined by the promotion rules
for subtypes, `S` and `T`, of the type `Real`: ``T^*``.

# Type Constraints
- `T <: Real`
- `S <: Real`

# Arguments
- `x::T`     -- The base value.
- `n::Int64` -- The power.
- `m::S`     -- The modulus.

# Input Contract
- ``n \\ge 0`` 

# Return
``::T^*``    -- The Power Value mod `m`.
"""
function pow_n(x::T, n::Int64, m::S) where {T <: Real, S <: Real}

    # Check input contract.
    if n < 0
        throw(DomainError(n, "Parameter `n` must be non-negative."))
    end

    # Promote to a common type, this will be the type of the output.
    x, m, o = promote(x, m, Int8(1)) 

    # Anything to the 0'th power is 1.
    if n == 0
        return(o)
    end

    # Get modulus value.
    x %= m

    # -- Do repeated squaring based on the digits of `n-1`. --
    # Initialize values.
    s   = x
    n2d = digits(n-1, base=2)

    # Repeated squaring.
    @simd for d in n2d
        @fastmath s *= d == 1 ? x : o
        @fastmath s %= m
        @fastmath x *= x
        @fastmath x %= m
    end

    return s
end


end # module Finance

