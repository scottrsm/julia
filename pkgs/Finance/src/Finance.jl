module Finance

import Random, Distributions, Statistics

export sig_cumsum, tic_diff1, tic_diff2, isConvertible, ema, ema_std, ema_stats, std, WWsum

"""
    isConvertible(S, T)

Boolean function which returns `true` if a value of type `S` 
can be converted to a value of type `T`.

## Type Constraints
- `S <: Real`
- `T <: Real`

## Arguments
- `::Type{S}` -- A numeric type.
- `::Type{T}` -- A numeric type.

## Return
::Bool
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
    tic_diff1(t, x[; chk_inp=false])

Compute the numerical derivative of a function represented by `x`
with respect to `t` when the values in `t` are possibly irregular.

## Type Constraints
- `S <: Real`
- `T <: Real`

## Arguments
- `t :: Vector{T}`   -- A vector of times.
- `x :: Vector{T}`   -- A vector of values.

## Keyword Arguments
- `chk_inp :: Bool`  -- Check the input contract?

## Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint                     | Description                                                               |
|:------------------------------:|:------------------------------------------------------------------------- |
| `\\|t\\| = \\|x\\|`            | The length of the time and data vectors match.                            |
| `S => T`                       | Type `S` can be converted to type `T`.                                    |
| ``\\forall i, t_{i+1} > t_i``  | The times are increasing; consequently, we have a 1-1 map from `t` to `x`.|

## Return
:: Vector{T}
"""
function tic_diff1(t::Vector{S}         , 
                   x::Vector{T}         ;
                   chk_inp::Bool = false,
                  ) :: Vector{T} where {S <: Real, T <: Real}
    n = length(x)

    if chk_inp
        n != length(t)           && throw(DomainError(n-t, "The length of the time and data series must match."))
        !all(diff(t) .> zero(S)) && throw(DomainError(n-t, "The time series must have be strictly increasing."))
        !isConvertible(S, T)     && throw(DomainError(0, "Type S is not convertible to type T."))
    end

    tc = map(x -> convert(T, x), t)
    df = zeros(T, n-2)
    for i in 2:(n-1)
        @inbounds h1 = tc[i  ] - tc[i-1]
        @inbounds h2 = tc[i+1] - tc[i  ]
        @inbounds df[i] = (x[i+1] - x[i-1]) / (h1 + h2)
    end

    return(df)
end

"""
    tic_diff2(t, x[; chk_inp=false])

Compute the numerical second derivative of a function represented by `x`
with respect to `t` when the values in `t` are possibly irregular.

## Type Constraints
- `S <: Real`
- `T <: Real`

## Arguments
- `t :: Vector{S}` -- A vector of times.
- `x :: Vector{T}` -- A vector of values.

## Keyword Arguments
- `chk_inp :: Bool`  -- Check the input contract?
- `S` is convertible to `T`.

## Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint                     | Description                                                               |
|:------------------------------:|:--------------------------------------------------------------------------|
| `\\|t\\| = \\|x\\|`            | The length of the time and data vectors match.                            |
| `S => T`                       | Type `S` can be converted to type `T`.                                    |
| ``\\forall i, t_{i+1} > t_i``  | The times are increasing; consequently, we have a 1-1 map from `t` to `x`.|

## Return
:: Vector{T}
"""
function tic_diff2(t::Vector{S}, 
                   x::Vector{T};
                   chk_inp::Bool = false,
                  ) :: Vector{T} where {S <: Real, T <: Real}
    n = length(x)

    if chk_inp
        n != length(t)           && throw(DomainError(n-t, "The length of the time and data series must match."))
        !all(diff(t) .> zero(S)) && throw(DomainError(n-t, "The time series must have be strictly increasing."))
        !isConvertible(S, T)     && throw(DomainError(0, "Type S is not convertible to type T."))
    end

    tc = map(x -> convert(T, x), t)
    df = zeros(T, n-2)
    for i in 2:(n-1)
        @inbounds h1 = tc[i  ] - tc[i-1]
        @inbounds h2 = tc[i+1] - tc[i  ]
        @inbounds df[i] = (h2 * x[i+1] - (h1 + h2) * x[i] + h1 * x[i-1]) / (h1 * h2 * (h1 + h2))
    end

    return(df)
end



"""
    sig_cumsum(t, x, w, h[; chk_inp=false ])

Return a tuple of two vectors: tics, signals.
The signals are the collections of all deviations from 
a running mean (with window length `w`) of the series. 
The threshold of the deviation is `h`.

Deviation is determined by:
- ``S^+_t = {\\rm max}(0, S_{t-1} + x_t - E[x_{t-1}]; S^+_0 = 0``
- ``S^-_t = {\\rm min}(0, S_{t-1} + x_t - E[x_{t-1}]; S^-_0 = 0``
- ``S_t = {\\rm max}(S^+_t, -S^-_t)``

Collect all ``t, S_t`` where ``h \\ge S_t``.

## Type Constraints
- `S <: Real`
- `T <: Real`

## Arguments
- `t :: Vector{S}` -- The tic series to examine.
- `x :: Vector{T}` -- The series to examine.
- `w :: Int64`     -- The width of the moving average.
- `h :: T`         -- The threshold for the deviation to register.

## Keyword Arguments
- `chk_inp :: Bool`  -- Check the input contract?

## Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint                          | Description                                                               |
|:-----------------------------------:|:--------------------------------------------------------------------------|
|`\\|x\\|` ``\\ge`` `2`               | The length of x is at least ``2``.                                        | 
|`\\|x\\| = \\|t\\|`                  | The length of `x` is equal to the length of `t`.                          |
|`w > 1`                              | The window length is greater than ``1``.                                  |
|`h > 0`                              | The deviation threshold is greater than 0.                                |
|``\\forall i, t_{i+1} > t_{i}``      | The times are increasing; consequently, we have a 1-1 map from `t` to `x`.|

## Output Components
- `td :: Vector{S}` -- Values of `t` where deviations occurred.
- `xd :: Vector{T}` -- Values of `x` where deviations occurred.

## Output Contract
- `|td| = |xd|`

## Return
(td, xd) :: Tuple{Vector{S}, Vector{T}}
"""
function sig_cumsum(t::Vector{S}, 
                    x::Vector{T}, 
                    w::Int64, 
                    h::T;
                    chk_inp::Bool=false
                   ) :: Tuple{Vector{S}, Vector{T}} where {S <: Real, T <: Real}
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
    ema(x,h,m)

Compute the Exponential Moving Average of the sequence `x`.

## Type Constraints
- `T <: Real`

## Arguments
- `x :: Vector{T}`   -- The series to work with.
- `m :: Int64`       -- The width of the decay window.

## Keyword Arguments
- `h :: Int64`       -- The exponential decay *half-life*. 

## Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint | Description                                   | 
|:----------:|:----------------------------------------------|
| `m > 1`    | Averaging window length is greater than ``1``.|
| `h > 1`    | Exponential *half-life* is greater than ``1``.|

## Output 
- `ema :: Vector{T}` -- The exponential moving average of `x`.

## Output Contract
- `|x| = |ema|`

## Return
ema::Vector{T}

"""
function ema(x :: Vector{T}              , 
             m :: Int64                  ; 
             h = div(m, 2) :: Int64      ,
            ) :: Vector{T} where { T <: Real }

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
    for i in 2:m
        @inbounds w[i] = l * w[i-1]
    end
    w ./= sum(w)

    ## Compute the EMA using the difference equation recursion.
    ma[1] = xadj[m+1]
    for i in 2:N
        @inbounds ma[i] =  l * (ma[i-1] - w[m] * xadj[i]) + w[1] * xadj[i+m]     
    end

    return(ma)
end


"""
    ema_std(x,h,m)

Compute the Moving Exponential Standard Deviation of the sequence `x`.
By default, the initial std is taken to be the standard deviation of 
the first window (of length `m`). However, a user specified value
may be used instead.

## Type Constraints
- `T <: Real`

## Arguments
- `x :: Vector{T}` -- The series to work with.
- `m :: Int64`     -- The width of the decay window.

## Keyword Arguments
- `h        :: Int64`             -- The exponential decay *half-life*. 
- `init_sig :: Union{T, Nothing}` -- A user supplied initial std for the start of the series.      

## Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint           | Description                                             |   
|:--------------------:|:------------------------------------------------------- |
| `m > 1`              | Averaging window length is greater than ``1``.          |
| `h > 1`              | Exponential *half-life* is greater than ``1``.          |
| `\\|x\\| > 1`        | The length of the series is greater than ``1``.         |
| `init_sig` ``\\ge 0``| User supplied starting ``\\sigma`` should be ``\\ge 0``.|

## Output 
- `stda :: Vector{T}` -- The moving exponential standard deviation of `x`.

## Output Contract
- `|x| = |stda|`

## Return
stda::Vector{T}
"""
function ema_std(x                  :: Vector{T}        ,
                 m                  :: Int64            ;
                 h = div(m, 2)      :: Int64            ,
                 init_sig = nothing :: Union{T, Nothing},
                )  :: Vector{T} where {T <: Real}

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
    for i in 2:m
        @inbounds w[i] = l * w[i-1]
    end
    w ./= sum(w)
    w2 = sum(w .* w)

    ## Recursive formula for variance.
    for n in 1:(N-1)
        @inbounds mvar[n+1] = l * (mvar[n] - xadj[n+1] * w[m]) + xadj[n+m+1] * w[1] 
    end

    ## Return corrected variances (unbiased).
    return(sqrt.(mvar ./ (one(T) - w2)))
end


"""
    ema_stats(x,h,m)

Compute the Moving Exponential Stats of the sequence `x`: `ema`, `ema_std`, `ema_rel_skew`, `ema_rel_kurtosis`.
The recursive formulas for the moving statistics as well as the adjustments necessary to 
render the estimates *unbiased* come from the paper:
[exponential\\_moving\\_average.pdf](https://github.com/scottrsm/math/tree/main/pdf/exponential_moving_average.pdf).

Returns these stats as a matrix with four columns, each representing the stats above in the order listed.

## Type Constraints
- `T <: Real`

## Arguments
- `x :: Vector{T}` -- The series to work with.
- `m :: Int64`     -- The width of the decay window.

## Keyword Arguments
- `h :: Int64`     -- The exponential decay *half-life*. 

## Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint     | Description                                    | 
|:--------------:|:---------------------------------------------- |
| `m > 1`        | Averaging window length is greater than ``1``. |
| `h > 1`        | Exponential *half-life* is greater than ``1``. |
| `\\|x\\| > 3`  | The length of the series is greater than ``3``.|

## Output 
- `stat :: Matrix{T}` -- A matrix of EMA stats: `ema`, `ema_std`, `ema_rel_skew`, `ema_rel_kurtosis`.

## Output Contract
- `|stat| = (N, 4)` 

## Return
stat::Matrix{T}
"""
function ema_stats(x                :: Vector{T}        ,
                   m                :: Int64            ;
                   h                :: Int64 = div(m, 2),
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
    for i in 1:m
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
        for n in 1:(N-1)
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

## Type Constraints
- `T <: Real`

## Arguments
- `x :: Vector{T}` -- The series to work with.

## Input Contract
The inputs are assumed to satisfy the constraints below.

| Constraint     | Description                                    | 
|:--------------:|:---------------------------------------------- |
| `\\|x\\| > 1`  | The length of the series is greater than ``1``.|

## Return
std::T -- The sample standard deviation.
"""
function std(x::Vector{T}) where {T <: Real}
    sd = zero(T)
    mn = zero(T)
    N = length(x)
    for i in 1:N
        @inbounds mn += x[i]
    end
    mn /= N

    for i in 1:N
        @inbounds sd += (x[i] - mn) * (x[i] - mn)
    end
    return( sqrt(sd / (N-1)) )
end

"""
    WWsum(w)

Computes the sum: ``\\sum_{i=1}^{N-1} \\sum_{j=i+1}^N w_i^2 w_j^2``.

Here, `N = |w|`. 

## Type Constraints
- `T <: Real`

## Arguments
- `w :: Vector{T}` -- Vector of weights.

## Return
The above sum.

"""
function WWsum(w::Vector{T}) :: T  where {T <: Real}
    WW = zero(T)
    m = length(w)
    for i in 1:(m-1)
        @inbounds wwi = w[i] * w[i]
        wwj = zero(T)
        for j in (i+1):m
            @inbounds wwj += w[j] * w[j] 
        end
        WW += wwi * wwj
    end 
    return(WW)
end



end # module Finance
