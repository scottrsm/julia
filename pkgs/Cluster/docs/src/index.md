# Cluster.jl Documentation

```@meta
CurrentModule = Cluster
```

# Overview
This module contains functions to determine natural clusters using unsupervised learning.
A point of interest in this module over other libraries is the rich set of *metrics* 
that one can use with K-means clustering. 
Additionally, some of the metrics may be weighted which can be used to help alleviate  
**K-means** attraction to spherical clusters.

This module also has the functionality to determine the cluster size of a data set.

Cluster functions (from lowest to highest level):
- `kmeans_cluster`: Given a cluster number and metric, determine clusters. 
    - This clustering algorithm can use any of the metrics: 
        `L2` (``L_2`` -- default), `LP` (``L_p``), `LI` (``L_\infty``), `KL` (Kullback-Leibler), `CD` (Cosine Distance), and `JD` (Jaccard Distance).
    - In the case of `L2` and `CD`, the metrics allow for weighted distances represented as a positive definite matrix.

- `find_best_info_for_ks`: Given a metric and a range of cluster numbers, determine clusters and gather fitness data for each cluster.
    - This function uses the functions:
        - kmeans\_cluster.

- `find_best_cluster`: Given a metric, find the best cluster number and determine the clusters. 
    - This function uses the functions:
        - `kmeans_cluster`
        - `find_best_info_for_ks`


## Metric Definitions:
Given `N` vectors, ``{\bf x}, {\bf y}`` :
- `L2`: The standard ``L_2`` norm: ``{\rm L2}({\bf x}, {\bf y}) = \sqrt{\sum_{i=1}^N (x_i - y_i)^2}``
    -  With a symmetric, positive semi-definite weight matrix `W` 
       this becomes: ``{\rm L2}({\bf x}, {\bf y}, \hbox{M=W}) = \sqrt{{\bf x} {\boldsymbol \cdot} (M {\bf y})}``
- `LP`: The standard ``L_p`` norm: ``{\rm LP}({\bf x}, {\bf y}, p) = \left(\sum_{i=1}^N |x_i - y_i|^p)\right)^{\frac{1}{p}}``
    - **Note:** To use this metric with `find_best_cluster` for a given value of `p`, 
        you will need to pass the closure, `(x,y; kwargs...) -> LP(x,y, p; kwargs...)`,
        to the keyword parameter `dmetric`.
- `LI`: The standard ``L_\infty`` norm: ``{\rm LI}({\bf x}, {\bf y}) = \mathop{\rm max}_{i \in [1,N]}\limits |x_i - y_i|`` 
- `KL`: A symmetrized Kullback-Leibler divergence: ``{\rm KL}({\bf x}, {\bf y}) = \sum_{i=1}^N x_i \log(x_i/y_i) + y_i \log(y_i/x_i)``
- `CD`: The "cosine" distance: ``{\rm CD}({\bf x}, {\bf y}) = 1 - {\bf x} {\boldsymbol \cdot} {\bf y} / (\|{\bf x}\|  \|{\bf y}\|)``
    - With a symmetric *strictly* positive definite weight matrix `W` this becomes: 
        ``\\ {\rm CD}({\bf x}, {\bf y}, \hbox{M=W}) = 1 - {\bf x} {\boldsymbol \cdot} \left( M {\bf y}\right) / (|\!|\!|{\bf x}|\!|\!|  |\!|\!|{\bf y}|\!|\!|)`` 
        - Here: ``|\!|\!| {\bf z} |\!|\!| = \sqrt{{\bf z} {\boldsymbol \cdot} \left( M {\bf z}\right)}``
- `JD`: The Jaccard distance.

## What is the best cluster number?
The function `find_best_cluster` attempts to find the best cluster number.
To do this, it monitors the total variation as one increases the cluster number. The total variation goes 
down (generally) as we find (potentially locally) optimal solutions for each cluster number.
If we pick a cluster number using only the total variation, we will miss the "natural cluster" number.

To avoid this, we adjust the total variation by a function that depends on the dimension of the space
we are working in as well as the cluster number. The reasoning follows:

The idea is to look at the natural rate at which the total variation decreases with cluster number when 
there are no clusters. In this way we can adjust the total variation to take into account 
this "ambient" decay.

To do this, we start by assuming that the data is uniformly distributed in our domain 
(with respect to the metric used) when given the data: `k` clusters; `m` points; the domain in `n` dimensions.
We assume that the `k` clusters have the same number of points and fill a sphere 
of radius, `R`. This means that ``R^n \approx k \, r_k^n``.

Solving for ``r_k`` we have ``{r_k=R\\\left(\\\frac{1}{k}\\\right)^{\\\frac{1}{n}}}``.
The total variation of `k` clusters is then roughly: ``{k \\\, r_k\\\left(\\\frac{m}{k}\\\right)}``. 
This becomes: ``\\\frac{m R}{k^{\\\frac{1}{n}}}``.
Thus, even in the absence of any true clusters, the total variation decays like ``k^{\\\frac{1}{n}}``.

The function `find_best_cluster` compares the total variation of cluster numbers in a range.
It chooses the cluster number, `k`, with the largest relative *rate* 
of decrease (with respect to cluster size) in adjusted total variation.
The adjusted variation modifies the total variation for each `k` by the multiplicative factor  ``k^{\\\frac{1}{n}}``. 
The variation is further adjusted by the 
fraction of unused cluster centroids. Finally, before computing the relative rate of variation decrease, the 
series is further adjusted to be monotonically non-increasing.

**NOTE:** This analysis may not be as useful if the "natural" clusters (or a substantial subset) 
lie in some lower dimensional hyperplane in the ambient space.


## Cluster Functions

```@docs
kmeans_cluster(::Matrix{T}, ::Int64 = 3; ::F = L2, ::Float64 = 1.0e-3,::Union{Nothing, AbstractMatrix{T}} = nothing,::Int64 = 1000, ::Int64=0) where {T <: Real, F <: Function}
```

```@docs
find_best_info_for_ks(::Matrix{T}, ::UnitRange{Int64}; ::F=L2, ::Float64=1.0e-3, ::Union{Nothing, AbstractMatrix{T}}=nothing, ::Int64=1000, ::Int64=300, ::Int64=1) where{T <: Real, F <: Function}
```

```@docs
find_best_cluster(::Matrix{T}, ::UnitRange{Int64}; ::F=L2, ::Float64=1.0e-3, ::Union{Nothing, AbstractMatrix{T}}=nothing, ::Int64=1000, ::Int64=300, ::Int64=1, ::Bool=false) where{T <: Real, F <: Function}
```

## Metric Functions

```@docs
L2(::AbstractVector{T},::AbstractVector{T}; M=::Union{Nothing, AbstractMatrix{T}} = nothing) where {T <: Real}
```

```@docs
LP(::AbstractVector{T},::AbstractVector{T}, ::Int64) where {T <: Real}
```

```@docs
LI(::AbstractVector{T},::AbstractVector{T}) where {T <: Real}
```

```@docs
KL(::AbstractVector{T},::AbstractVector{T}) where {T <: Real}
```

```@docs
CD(::AbstractVector{T},::AbstractVector{T}; M=::Union{Nothing, AbstractMatrix{T}} = nothing) where {T <: Real}
```

```@docs
JD(::AbstractVector{T},::AbstractVector{T}) where {T <: Real}
```

## Fit Metric Functions

```@docs
raw_confusion_matrix(::AbstractVector{A},::AbstractVector{P}) where {A, P}
```

```@docs
confusion_matrix(::AbstractVector{A},::AbstractVector{P}) where {A, P}
```

## Index

```@index
```

