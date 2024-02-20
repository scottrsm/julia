# Finance.jl Documentation

```@meta
CurrentModule = Finance
```

# Overview
Below is a collection of functions which are useful to find signals 
in time series data.
- `sigcumsum:` This function looks for imbalance in tics.
- `isConvertible:` This function determines if one type can be converted to another.
- `tic_diff1:` As tics are irregular, this function provides a 
                way to compute the numerical derivative 
                with respect to an irregular time series.
- `tic_diff2:` This function finds the numerical second derivative
                with respect to irregular time series data.
- `WWsum:`     A utility function used in `ema_stat`.
- `ema:`       Computes the Exponential Moving Average of a time series. 
- `ema_std:`   Computes the standard deviation of an
               Exponential Moving Average of a time series. 
- `ema_stat:`  Computes the Exponential Moving Average along with
               the associated moving standard deviation, relative skewness, 
               relative kurtosis.
- `ewt_mean`   Computes a exponentially decayed temporal weighted moving average.
- `pow_n:`     Computes the power of a value to a positive integer power 
               using repeated squaring.

## Signals

```@docs
sig_cumsum
```

## Moving Averages
```@docs
ema
```

```@docs
ema_std
```

```@docs
ema_stats
```

```@docs
std
```

## Utilities

```@docs
isConvertible
```

```@docs
tic_diff1(::AbstractVector{T}, ::AbstractVector{S}; ::Bool=false) where {T <: Real, S <: Real}
```

```@docs
tic_diff2(::AbstractVector{T}, ::AbstractVector{S}; ::Bool=false) where {T <: Real, S <: Real}
```

```@docs
pow_n(::T, ::Int64) where {T <: Number} 
```

```@docs
pow_n(::T, ::Int64, ::S) where {T <: Real, S <: Real} 
```

```@docs
entropy_index(::Vector{T}; ::Int64=10, ::Float64=1.0 / (100 * n), ::Vector{Float64}=[0.01, 0.99], ::Float64=1.0) where {T <: Real}
```

```@docs
ewt_mean(::Vector{Float64}, ::Vector{Float64}, ::Int64, ::Float64)
```

## Index

```@index
```

