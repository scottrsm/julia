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
tic_diff1
```

```@docs
tic_diff2
```

```@docs
WWsum
```

```@docs
pow_n(::T, ::Int64) where {T <: Number} 
```

```@docs
pow_n(::T, ::Int64, ::T) where {T <: Number} 
```

```@docs
entropy_index
```


## Index

```@index
```

