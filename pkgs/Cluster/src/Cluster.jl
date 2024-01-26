module Cluster


export L2, KL, LP, cos_dist 
export kmeans_cluster, find_best_cluster, find_best_info_for_ks

import LinearAlgebra as LA
import Statistics as S
import StatsBase as SB
import Random as R
import DataStructures as DS


"""
    L2(x,y[;tol=1.0e-3, C=nothing])

Computes the ``L_2`` distance between two vectors.
One of the features that may be different from other packages
is the use of weighted metrics in some instances.

## Type Constraints
- `T <: Real`

## Arguments
- `x::Vector{T}` : A numeric vector of dimension `n`.
- `y::Vector{T}` : A numeric vector of dimension `n`.

## Keyword Arguments
- `tol::Float64` : A tolerance -- **NOT** used.
- `C::Union{Nothing, Matrix{T}` : Optional Weight matrix.

## Input Contract (Low level function -- Input contract not checked)
- ``|{\\bf x}| = |{\\bf y}|``
- ``C = {\\rm nothing} ∨ \\left( ({\\rm typeof}(C) = {\\rm Matrix}\\{T\\}) ∧ C \\in {\\boldsymbol S}_{+}^{|{\\bf x}|} \\right)``

## Return
``L_2`` (optionally weighted) distance measure between the two vectors.
"""
function L2(x::Vector{T},
            y::Vector{T};
            tol::Float64=1.0e-10,
            C::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:Real}

    d = x .- y
    if C === nothing
        return LA.norm2(d)
    else
        return sqrt(LA.dot(d, C, d))
    end
end

"""
    LP(x,y,p[;tol=1.0e-3, C=nothing])

Computes the ``L_p`` distance between two vectors.

## Type Constraints
- `T <: Real`

## Arguments
- `x::Vector{T}` : A numeric vector of dimension `n`.
- `y::Vector{T}` : A numeric vector of dimension `n`.
- `p::Int64`     : The power of the norm.

## Keyword Arguments
- `tol::Float64` : A tolerance -- **NOT** used.
- `C::Union{Nothing, Matrix{T}` : Optional Weight matrix -- **NOT** used.

## Input Contract (Low level function -- Input contract not checked)
- ``|{\\bf x}| = |{\\bf y}|``
- `p > 0`

## Return
``L_p`` distance measure between the two vectors.
"""
function LP(x::Vector{T},
            y::Vector{T},
            p::Int64;
            tol::Float64=1.0e-10,
            C::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:Real}
    return LA.norm(x .- y, p)
end


"""
    JD(x,y[;tol=1.0e-3, C=nothing])

Computes the Jaccard metric between two vectors of a discrete type.
For instance, the vectors could be integers; however, they can 
also be of non-numeric type.
If both `x` and `y`, a distance of 0 is returned.

## Arguments
- `x::Vector{T}` : A numeric vector of dimension `n`.
- `y::Vector{T}` : A numeric vector of dimension `n`.

## Keyword Arguments
- `tol::Float64`                : A tolerance -- **NOT** used.
- `C::Union{Nothing, Matrix{T}` : Optional Weight matrix -- **NOT** used.

## Input Contract (Low level function -- Input contract not checked)
- ``| {\\bf x} | = | {\\rm unique}({\\bf x}) |``
- ``| {\\bf y} | = | {\\rm unique}({\\bf y}) |``
- ``\\forall i \\in [1, N]: x_i \\ge 0``
- ``\\forall i \\in [1, N]: y_i \\ge 0``
- ``\\sum_{i=1}^N x_i = 1``
- ``\\sum_{i=1}^N y_i = 1``

## Return
`JD` distance measure between the two vectors.
"""
function JD(x::Vector{T},
            y::Vector{T};
            tol::Float64=1.0e-10,
            C::Union{Nothing,AbstractMatrix{T}}=nothing) where {T}
    d = length(symdiff(x,y))
    u = length(union(x,y)) 
    return length(u) == 0 ? 0.0 : d / u
end


"""
    KL(x,y[;tol=1.0e-3, C=nothing])

Computes the ``Kullback-Leibler`` distance between two vectors.

## Type Constraints
- `T <: Real`

## Arguments
- `x::Vector{T}` : A numeric vector of dimension `n`.
- `y::Vector{T}` : A numeric vector of dimension `n`.

## Keyword Arguments
- `tol::Float64`                : A tolerance -- **NOT** used.
- `C::Union{Nothing, Matrix{T}` : Optional Weight matrix -- **NOT** used.

## Input Contract (Low level function -- Input contract not checked)
Let ``N = |{\\bf x}|``.
- ``|{\\bf x}| = |{\\bf y}|``
- ``\\forall i \\in [1, N]: x_i \\ge 0``
- ``\\forall i \\in [1, N]: y_i \\ge 0``
- ``\\sum_{i=1}^N x_i = 1``
- ``\\sum_{i=1}^N y_i = 1``

## Return
`KL` distance measure between the two vectors.
"""
function KL(x::Vector{T},
            y::Vector{T};
            tol::Float64=1.0e-10,
            C::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:Real}
    z = zero(T)
    d1 = map((a, b) -> a == z ? z : a * log(a / b), x, y)
    d2 = map((a, b) -> b == z ? z : b * log(b / a), x, y)
    return sum(d1 .+ d2)
end


"""
    cos_dist(x,y[;tol=1.0e-3, C=nothing])

Computes the "cosine" distance between two vectors.

## Type Constraints
- `T <: Real`

## Arguments
- `x::Vector{T}` : A numeric vector of dimension `n`.
- `y::Vector{T}` : A numeric vector of dimension `n`.

## Keyword Arguments
- `tol::Float64` : A tolerance used to test for zero vectors.
- `C::Union{Nothing, Matrix{T}` : Optional Weight matrix.

## Input Contract (Low level function -- Input contract not checked)
- ``|{\\bf x}| = |{\\bf y}|``
- ``C = {\\rm nothing} \\wedge \\left( ({\\rm typeof}(C) = {\\rm Matrix}\\{T\\}) \\vee C \\in {\\boldsymbol S}_{++}^{|{\\bf x}|} \\right)``

## Return
Cosine distance measure between the two vectors.

"""
function cos_dist(x::Vector{T},
                  y::Vector{T};
                  tol::Float64=1.0e-10,
                  C::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:Real}
    z = zero(T)
    o = one(T)
    if all(abs.(x .- y .< tol))
        return z
    elseif all(abs.(x .- z .< tol))
        return o
    elseif all(abs.(y .- z .< tol))
        return o
    elseif C === nothing
        return o - LA.dot(x, y) / sqrt(LA.dot(x, x) * LA.dot(y, y))
    end

    return o - LA.dot(x, C, y) / sqrt(LA.dot(x, C, x) * LA.dot(y, C, y))
end




"""
    kmeans_cluster(X, k[; dmetric, threshold, W, N, seed])

Groups a set of points into `k` clusters based on the distance metric, `dmetric`.

## Type Constraints
- `T <: Real`
- `F <: Function`

## Arguments
- `X::Matrix{T}`  : (n,m) Matrix representing `m` points of dimension `n`.
- `k::Int64=3`    : The number of clusters to form.

## Keyword Arguments
- `dmetric::F=L2` : The distance metric to use.
- `threshold::Float=1.0e-2`  : The relative error improvement threshold (using total variation)
- `W::Union{Nothing, AbstractMatrix{T}}=nothing` : Optional Weight matrix for metric.
- `N::Int64=1000`    : The maximum number of iterations to try.
- `seed::Int64=0`    : Set the random seed if value > 0 -- used for initial clustering.
    
## Input Contract
- ``W = {\\rm nothing} ∨ \\left( ({\\rm typeof}(W) = {\\rm Matrix}\\{T\\}) ∧ W \\in {\\boldsymbol S}_{++}^{|{\\bf x}|} \\right)``
- `k > 0`
- `N > 0`
- `threshold > 0.0`
- `dmetric <: Function`

## Return
A Tuple:
- `Dict{Int64, Int64}`: Mapping of points (`n`-vectors) indices to centroid indices.
- `Matrix{T}`         : (nxk) Matrix representing `k` centroids of `n`-vectors.
- `Float64`           : The total variation between points and their centroids (using `dmetric`).
- `Vector{Int64}`     : Unused centroids (by index).
- `Int64`             : The number of iterations to use for the algorithm to complete.
- `Bool`              : Did algorithm converge.
"""
function kmeans_cluster(X::Matrix{T},
                        k::Int64=3;
                        dmetric::F=L2,
                        threshold::Float64=1.0e-2,
                        W::Union{Nothing,AbstractMatrix{T}}=nothing,
                        N::Int64=1000,
                        seed::Int64=0) where {T<:Real,F<:Function}
    # Get the size of the matrix.
    # `m` vectors of length `n`.
    n, m = size(X)

    # Check input contract: 
    # NOTE: We only check that if W is a matrix it has the right shape.
    #       We do not check for symmetry or strict positive definiteness.
    if !((W === nothing) || ((typeof(W) <: AbstractMatrix{T}) && (size(W) == (n, n))))
        throw(DomainError(W, "The variable, `W`, which is not of type `Nothing` must be of type `Matrix{T}` with size(W) = $((n,n))"))
    elseif !(k > 0)
        throw(DomainError(k, "The variable, `k`, is less than 1."))
    elseif !(N > 0)
        throw(DomainError(N, "The variable, `N`, is less than 1."))
    elseif !(threshold > 0.0)
        throw(DomainError(threshold, "The variable, `threshold`, is <= 0.0."))
    elseif !(typeof(dmetric) <: Function)
        throw(DomainError(typeof(dmetric), "The variable, `dmetric`, is not a subtype of `Function`."))
    end

    rng=nothing
    # If seed > 0, set the random seed.
    if seed > 0
        rng = R.Xoshiro(seed)
    end

    # Randomly sample the `m` (`n`-vectors)
    # We are permuting the columns as Julia stores matrices by columns.
    if rng === nothing
        idx = SB.sample(1:m, m, replace=false)
    else
        idx = SB.sample(rng, 1:m, m, replace=false)
    end
    XR = X[:, idx]

    # Group the `m` vectors into `k` groups, find each of their means.
    # These will be the initial `k` centers.
    ck = div(m, k)
    idx = 1 .+ (unique(div.(1:m, ck)) * ck)
    idx[end] = min(idx[end], m)

    # Average the vectors in each group to form the group centers.
    XC = Array{T}(undef, n, k)
    for j in 1:k
        XC[:, j] = S.mean(XR[:, idx[j]:idx[j+1]], dims=2)
    end

    # A dictionary to map the original points (their indices) to centroids indices.
    # The map will change as the centroids change.
    cmap = Dict{Int64,Int64}()

    # Now loop until convergence -- or max iterations: 
    # - Map the `m` values of (`n`-vectors) from X into the nearest cluster.
    # - Form new centers by averaging groups.

    # `ds_last`: The previous total variation.
    tmax = typemax(T)
    ds_last = tmax

    # The number of iterations we allow `N`.
    for l in 1:N
        ds = zero(T)
        cmini = -1

        # Loop over the `m` points.
        # For each point, find the nearest cluster (by centroid index).
        for i in 1:m
            dvmin = tmax
            dt = zero(T)
            for j in 1:k
                dt = dmetric(X[:, i], XC[:, j]; C=W)
                if dt < dvmin
                    dvmin = dt
                    cmini = j
                end
            end
            ds += dvmin
            cmap[i] = cmini
        end
        
        # IF: No appreciable change based on relative error, return.
        # 1. The mapping dictionary:
        #     (Original point index -> centroid index)
        # 2. The Centroids.
        # 3. The overall total distance from points and their centroids.
        # 4. Unused centroid indices.
        # 5. Number of runs to completion.
        # 6. Did algorithm converge.
        if abs(ds_last - ds) / max(ds, ds_last) < threshold
            return (cmap, XC, ds, setdiff(1:k, unique(values(cmap))), l, true)
        end

        # ELSE: Update last total distance measure.
        ds_last = ds

        # Compute the new centroids, for each cluster.
        XC = zeros(T, n, k)
        cntC = Dict{Int64,Int64}()

        # Accumulate vectors in each centroid mapping.
        for mi in keys(cmap)
            ci          = cmap[mi]
            XC[:, ci] .+= X[:, mi]
            cntC[ci]    = get(cntC, ci, 0) + 1
        end

        # Compute new centroids.
        for ci in unique(values(cmap))
            XC[:, ci] ./= cntC[ci]
        end
    end
    return (cmap, XC, ds_last, setdiff(1:k, unique(values(cmap))), N, false)
end



"""
    find_best_info_for_ks(X, kRng[; dmetric=L2, threshold=1.0e-3, W, N=1000, num_trials=100, seed=1])

Groups a set of`m` points (`n`-vectors) as an (nxm) matrix, `X`, into `k` clusters where `k` is in the range, `kRng`.
The groupings are determined based on the distance metric, `dmetric`.

## Type Constraints
- `T <: Real`
- `F <: Function`

## Arguments
- `X::Matrix{T}`           : (n,m) Matrix representing `m` points of dimension `n`.
- `kRng::UnitRange{Int64}` : The number of clusters to form.

## Keyword Arguments
- `dmetric::F=L2`          : The distance metric to use.
- `threshold::Float=1.0e-2`: The relative error improvement threshold (using total variation)
- `W::Union{Nothing, AbstractMatrix{T}}=nothing` : Optional Weight matrix for metric.
- `N::Int64=1000`          : The maximum number of kmeans_clustering iterations to try for each cluster number.
- `num_trials::Int64=300`  : The number of times to run kmeans_clustering for a given cluster number. 
- `seed::Int64=1`          : The random seed to use. (Used by kmeans_cluster to do initial clustering.)
    
## Input Contract
- ``W = {\\rm nothing} ∨ \\left( ({\\rm typeof}(W) = {\\rm Matrix}\\{T\\}) ∧ W \\in {\\boldsymbol S}_{++}^{|{\\bf x}|} \\right)``
- `N > 0`
- ``∀ i \\in {\\rm kRng}, i > 1``
- `threshold > 0.0`
- `dmetric <: Function`

## Return
A Tuple with entries:
- `OrderedDict{Int64, Float}`             : 1:k -> The Total Variation for each cluster number.
- `OrderedDict{Int64, Dict{Int64, Int64}}`: 1:k -> Mapping of index of points (n-vectors in `X`) to centroid indices.
- `OrderedDict{Int64, Matrix{T}`          : 1:k -> (nxk) Matrix representing `k` `n`-vector centroids.
- `OrderedDict{Int64, Vector{In64}}`      : 1:k -> Vector of unused centroids by index.
"""
function find_best_info_for_ks(X::Matrix{T},
                               kRng::UnitRange{Int64};
                               dmetric::F=L2,
                               threshold::Float64=1.0e-2,
                               W::Union{Nothing,AbstractMatrix{T}}=nothing,
                               N::Int64=1000,
                               num_trials::Int64=300,
                               seed::Int64=1) where {T<:Real,F<:Function}

    ds_by_k   = DS.OrderedDict{Int64,T}()
    cmap_by_k = DS.OrderedDict{Int64,Dict{Int64,Int64}}()
    XC_by_k   = DS.OrderedDict{Int64,Matrix{T}}()
    sd_by_k   = DS.OrderedDict{Int64,Vector{Int64}}()
    tmax = typemax(T)
    cnt = 0

    # Check input contract.
    for i in kRng
        if i < 2
            throw(DomainError(typeof(kRng), "The variable, `kRng`, has at least one value in its range that is < 2."))
        end
    end

    # Loop over the cluster range.
    # Find best cluster for each cluster size.
    # For each cluster size store the following data:
    #  - The mapping of points (index) to cluster points (index).
    #  - The cluster points.
    #  - Total variation.
    #  - The list of cluster indices that were not used.
    #  - The number of iterations used to complete kmeans_cluster.
    #  - Did kmeans_cluster converge before max iterates used? 
    for k in kRng
        ds_by_k[k] = tmax
        for _ in 1:num_trials
            cnt += 1
            cmap, XC, ds, sd, N, _ = kmeans_cluster(X, k;
                                                    dmetric=dmetric,
                                                    threshold=threshold,
                                                    W=W,
                                                    N=N,
                                                    seed=(seed+cnt))
            if ds < ds_by_k[k]
                ds_by_k[k]   = ds
                cmap_by_k[k] = cmap
                XC_by_k[k]   = XC
                sd_by_k[k]   = sd
            end
        end
    end

    return (ds_by_k, cmap_by_k, XC_by_k, sd_by_k)

end



"""
    find_best_cluster(X, kRng[; dmetric=L2, threshold=1.0e-3, W, N=1000, num_trials=100, seed=1, verbose=false])

Groups a set of points into the "best" number of clusters based on the distance metric, `dmetric`.
It does this by examining the total variation between the points and the centroids for groups of `k`
where `k` is in the range, `kRng`. 

**NOTE:** If the value `k` was determined to be the best cluster number but some of the
centroids were not used, then the value of `k` will be set to the number of centroids that
are used and the centroids that were not used will be removed. In this case it may be
that the returned value of `k` is less that any value in the cluster range, `kRng`.

## Type Constraints
- `T <: Real`
- `F <: Function`

## Arguments
- `X::Matrix{T}`           : (n,m) Matrix representing `m` points of dimension `n`.
- `kRng::UnitRange{Int64}` : The range of potential cluster values to try.

## Keyword Arguments
- `dmetric::F=L2`          : The distance metric to use.
- `threshold::Float=1.0e-2`: The relative error improvement threshold (using total variation)
- `W::Union{Nothing, AbstractMatrix{T}}=nothing` : Optional Weight matrix for metric.
- `N::Int64=1000`          : The maximum number of kmeans_clustering iterations to try for each cluster number.
- `num_trials::Int64=300`  : The number of times to run kmeans_clustering for a given cluster number. 
- `seed::Int64=1`          : The random seed to use. (Used by kmeans_cluster to do initial clustering.)
- `verbose::Bool=false`    : The random seed to use. (Used by kmeans_cluster to do initial clustering.)
    
## Input Contract
- ``W = {\\rm nothing} ∨ \\left( ({\\rm typeof}(W) = {\\rm Matrix}\\{T\\}) ∧ W \\in {\\boldsymbol S}_{++}^{|{\\bf x}|} \\right)``
- `N > 0`
- ``∀ i \\in {\\rm kRng}, i > 1``
- `threshold > 0.0`
- `dmetric <: Function`

## Return
A Tuple:
- `Int64`             : The "best" cluster number, `k`.
- `Dict{Int64, Int64}`: Mapping of points (`n`-vectors) indices to centroid indices.
- `Matrix{T}`         : Cluster centroids, represented as an `(n,k)` matrix.
- `Float64`           : The total variation between points and their centroids (using `dmetric`).
"""
function find_best_cluster(X::Matrix{T},
                           kRng::UnitRange{Int64};
                           dmetric::F=L2,
                           threshold::Float64=1.0e-2,
                           W::Union{Nothing,AbstractMatrix{T}}=nothing,
                           N::Int64=1000,
                           num_trials::Int64=300,
                           seed::Int64=1,
                           verbose::Bool=false ) where {T<:Real, F<:Function}

    # Get the info for the best clusters in the range: `kRng`.
    ds, cmap, xc, sd = find_best_info_for_ks(X,
                                             kRng;
                                             dmetric=dmetric,
                                             threshold=threshold,
                                             W=W,
                                             N=N,
                                             num_trials=num_trials,
                                             seed=seed)

    # Get the dimension of the points.
    n, _ = size(X)

    # Used to adjust to cluster variation by data dimension and
    # number of clusters.
    fact = map(j -> j^(1.0 / n), kRng)

    # Get all of the cluster choices.
    mv = collect(values(kRng))

    # Get the total variation for each cluster number.
    dsv = collect(values(ds))

    # Get the number of unused cluster nodes for each cluster number.
    sdv = length.(collect(values(sd)))

    # Adjust the total variation, `dsv`, by `kfact`.
    # The `kfact` values adjust for the natural tendency for more clusters
    # to give less variation.
    # Also, penalize the variation by multiplying by a fraction that
    # takes into account unused centroids.
    var_by_k_mod = dsv .* fact .* (mv .+ sdv) ./ mv
    if verbose
        println("var_by_k     = $dsv")
        println("var_by_k_mod = $var_by_k_mod")
        println("rel change of var $(diff(var_by_k_mod) ./ var_by_k_mod[2:end])") 
    end

    # Find the cluster number with the least adjusted total variation.
    kbest = argmin(var_by_k_mod) + (kRng.start - 1)

    #kbest = argmin(var_by_k_mod) + (kRng.start - 1)
    kbest = argmin(diff(var_by_k_mod) ./ var_by_k_mod[2:end]) + kRng.start
    
    # Number of unused clusters.
    sdl = length(sd[kbest])

    # If no unused centroids, return.
    if sdl == 0
        return (kbest, cmap[kbest], xc[kbest], ds[kbest])
    end

    # Else we need to remove unused centroids and re-index the used centroids.
    viable_centroid_idxs = setdiff(1:kbest, sd[kbest])
    reindex_centroids = DS.OrderedDict{Int64, Int64}()
    bcmap = DS.OrderedDict{Int64, Int64}()
    cnt = 1
    for i in viable_centroid_idxs
        reindex_centroids[i] = cnt
        cnt += 1
    end

    # Remap the points to the index of the nearest centroid using the re-index map.
    for k in keys(cmap[kbest])
        bcmap[k] = reindex_centroids[cmap[kbest][k]]
    end
    
    # Return (number-of-clusters, map-of-point-to-cluster-index, clusters, total-variation-of-fit)
    return (length(viable_centroid_idxs), bcmap, xc[kbest][:, viable_centroid_idxs], ds[kbest])
end

end # End module Cluster


