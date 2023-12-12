# Cluster.jl Documentation

```@meta
CurrentModule = Cluster
```

# Overview
This module contains functions to determine natural clusters using unsupervised learning.

Cluster functions:
- kmeans_cluster: 
    - This clustering algorithm can use any of the metrics: `L2`, `LP`, `DL`, and `cos_dist`.
    - In the case of `L2` and `cos_dist`, the metrics allow for weighted distances.

## Metric Definitions:
- `L2`: The standard ``L_2`` norm. ``{\rm L2}({\bf x}, {\bf y}) = \sqrt{\sum_{i=1}^N (x_i - y_i)^2}``
    -  With a symmetric, positive semi-definite weight matrix `W` this becomes: ``{\rm L2}({\bf x}, {\bf y}, W) = \sqrt{{\bf x} {\boldsymbol \cdot} (W {\bf y})}``
- `LP`: The standard ``L_p`` norm. ``{\rm LP}({\bf x}, {\bf y}) = \left(\sum_{i=1}^N |x_i - y_i|^p)\right)^{\frac{1}{p}}``
- `DL`: A symmetrized Kullback-Leibler divergence: ``{\rm DL}({\bf x}, {\bf y}) = \sum_{i=1}^N x_i \log(x_i/y_i) + y_i \log(y_i/x_i)``
- `cos_dist`: The "cosine" distance: ``{\rm cos\_dist}({\bf x}, {\bf y}) = 1 - {\bf x} {\boldsymbol \cdot} {\bf y} / (\|{\bf x}\|  \|{\bf y}\|)``
    - With a symmetric strictly positive definite weight matrix `W` this becomes: 
        ``\\ {\rm cos\_dist}({\bf x}, {\bf y}, W) = 1 - {\bf x} {\boldsymbol \cdot} \left( W {\bf y}\right) / (|\!|\!|{\bf x}|\!|\!|  |\!|\!|{\bf y}|\!|\!|)`` 
        - Here: ``|\!|\!| {\bf z} |\!|\!| = \sqrt{{\bf z} {\boldsymbol \cdot} \left( W {\bf z}\right)}``

## What is the best cluster number?
The function `find_best_cluster` attempts to find the best cluster number.
To do this, it monitors the total variation as one increases the cluster number. The total variation goes 
down generally as we find (potentially locally) optimal solutions for each cluster number.
If we pick a cluster number using only the total variation, we will miss the "natural cluster" number.

To avoid this, we adjust the total variation by a function that depends on the dimension of the space
we are working in as well as the cluster number. The reasoning follows:

The idea is to look at the natural rate at which the total variation decreases with cluster number when 
there are no clusters. In this way we can adjust the total variation to take into account 
this "ambient" decay.

To do this, we start by assuming that the data is uniformly distributed in our domain 
(with respect to the metric used) with `k` clusters; `m` points; and the domain is in `n` dimensions.
We assume that the `k` clusters have the same number of points and fill a sphere 
of radius, `R`. This means that ``R^n \approx k r_k^n``.

Solving for ``r_k`` we have ``r_k = R {\\\frac{1}{k}}^{\\\frac{1}{n}}``.
The total variation of `k` clusters is then roughly: ``k r_k {\\\frac{m}{k}}``. This becomes: 
``\\\frac{m R}{k^{\\\frac{1}{n}}}``.
Thus, even in the absence of any true clusters, the total variation decays like ``k^{\\\frac{1}{n}}``.

The function `find_best_cluster` compares the total variation of cluster numbers in a range.
It chooses the cluster number, `k`, which minimizes the adjusted total variation.
The adjusted variation modifies the total variation for each `k` by the multiplicative factor  ``k^{\\\frac{1}{n}}``. 
The variation is further adjusted by the 
fraction of unused cluster centroids.

**NOTE:** This analysis may not be as useful if the natural clusters (or a substantial subset) 
lie in some lower dimensional hyperplane.


## Cluster Functions

```@docs
kmeans_cluster(::Matrix{T}, ::Int64 = 3; ::F = L2, ::Float64 = 1.0e-3,::Union{Nothing, AbstractMatrix{T}} = nothing,::Int64 = 1000, ::Int64=0) where {T <: Real, F <: Function}
```

```@docs
find_best_cluster(::Matrix{T}, ::UnitRange{Int64}; ::F=L2, ::Float64=1.0e-3, ::Union{Nothing, AbstractMatrix{T}}=nothing, ::Int64=1000, ::Int64=100, ::Int64=1, ::Bool=false) where{T <: Real, F <: Function}
```

```@docs
find_best_info_for_ks(::Matrix{T}, ::UnitRange{Int64}; ::F=L2, ::Float64=1.0e-3, ::Union{Nothing, AbstractMatrix{T}}=nothing, ::Int64=1000, ::Int64=1000, ::Int64=1) where{T <: Real, F <: Function}
```

## Metric Functions

```@docs
L2(::Vector{T},::Vector{T}; ::Float64 = 1.0e-3, ::Union{Nothing, AbgstractMatrix{T}} = nothing) where {T <: Real}
```

```@docs
LP(::Vector{T},::Vector{T},::Int64; ::Float64 = 1.0e-3, ::Union{Nothing, AbstractMatrix{T}}= nothing) where {T <: Real}
```

```@docs
cos_dist(::Vector{T},::Vector{T}; ::Float64 = 1.0e-3, ::Union{Nothing, AbstractMatrix{T}} = nothing) where {T <: Real}
```

## Index

```@index
```

