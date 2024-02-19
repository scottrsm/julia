module Cluster

import Statistics as S
import StatsBase as SB
import Random as R
import DataStructures as DS

include("Metrics.jl")
using .Metrics: L2, LP, LI, KL, CD, JD, raw_confusion_matrix, confusion_matrix

# Export the K-means functions: 
# Base k-means function; K-means function to get information over a range of clusters;
# and function that finds the best K-means cluster.
export kmeans_cluster, find_best_info_for_ks, find_best_cluster

# Export the metric and fit metric functions: 
export L2, LP, LI, KL, CD, JD, raw_confusion_matrix, confusion_matrix

"""
    kmeans_cluster(X, k=3[; dmetric, threshold, W, N, seed])

Groups a set of points into `k` clusters based on the distance metric, `dmetric`.

# Type Constraints
- `T <: Real`
- `F <: Function`

# Arguments
- `X::Matrix{T}`  : (n,m) Matrix representing `m` points of dimension `n`.
- `k::Int64=3`    : The number of clusters to form.

# Keyword Arguments
- `dmetric::F=L2` : The distance metric to use.
- `threshold::Float=1.0e-2`  : The relative error improvement threshold (using total variation)
- `W::Union{Nothing, AbstractMatrix{T}}=nothing` : Optional `(nxn)` weight matrix for metric.
- `N::Int64=1000`    : The maximum number of iterations to try.
- `seed::Int64=0`    : If value > 0, create a random number generator to use for initial clustering.
    
# Input Contract
- ``W = {\\rm nothing} ∨ \\left( ({\\rm typeof}(W) = {\\rm Matrix}\\{T\\}) ∧ W \\in {\\boldsymbol S}_{++}^{n} \\right)``
- ``1 \\le k \\le m``
- `N > 0`
- `threshold > 0.0`
- `dmetric <: Function`

# Return
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
                        threshold::Float64=1.0e-3,
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
    elseif !(1 <= k <= m)
        throw(DomainError(k, "The variable, `k`, is not in the range `[1, m]`."))
    elseif !(N > 0)
        throw(DomainError(N, "The variable, `N`, is less than 1."))
    elseif !(threshold > 0.0)
        throw(DomainError(threshold, "The variable, `threshold`, is <= 0.0."))
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

    # Group the `m` vectors into `k` groups, find each of their means.
    # These will be the initial `k` centers.
    ck = div(m, k)
    idx = 1 .+ (unique(div.(1:m, ck)) * ck)
    idx[end] = min(idx[end], m)

    # Average the vectors in each group to form the group centers.
    # The averaging below double counts some points -- not important
    # as this just a starting point for cluster centers.
    XCS = Array{T,2}(undef, n, k)
    @inbounds  for j in 1:k
        @views XCS[:, j] = S.mean(X[:, idx[j]:idx[j+1]], dims=2)
    end

    # A map of points to centroids using indices: 1:m -> 1:k
    # The map will change as the centroids change.
    cmap = Vector{Int64}(undef, m)

    # Number of points per centroid.
    cntC = Vector{Int64}(undef, m)

    # Variable used to keep track of previous total variation of clusters.
    tmax = typemax(T)
    tv_last = tmax

    adj_metric = dmetric
    if W !== nothing
        adj_metric = (x,y) -> dmetric(x,y; M=W)
    end


    # Now loop until convergence: abs(tv - tv_last) is small -- or max iterations (N): 
    # - Map the `m` values of (`n`-vectors) from X into the nearest cluster.
    # - Form new centers by averaging associated points.
    @inbounds for l in 1:N
        tv = zero(T) # Total variation (sum of distances) of all points to their centers.
        cv = zero(T) # Distance of one point with one center. 
        c_closest = -1 # Closest center (by index) of a point.

        # Loop over the `m` points.
        # For each point, find the nearest cluster (by centroid index).
        # Collect the variation.
        for i in 1:m
            cv_min = tmax
            xv = @view X[:, i]
            @simd for j in 1:k
                xcsv = @view XCS[:, j]
                cv = adj_metric(xv, xcsv)
                if cv < cv_min
                    cv_min    = cv
                    c_closest = j
                end
            end
            tv     += cv_min
            cmap[i] = c_closest
        end

        # IF: No appreciable change based on relative error, return.
        # 1. The mapping dictionary:
        #     (Original point index -> centroid index)
        # 2. The Centroids.
        # 3. The overall total distance from points and their centroids.
        # 4. Unused centroid indices.
        # 5. Number of runs to completion.
        # 6. Did algorithm converge.
        if abs(tv_last - tv) / max(tv, tv_last) < threshold
            return (cmap, XCS, tv, setdiff(1:k, unique(values(cmap))), l, true)
        end

        # ELSE: Update last total distance measure.
        tv_last = tv

        # Compute the new centroids, for each cluster.
        XCS = zeros(T, n, k)
        cntC .= 0 

        # Accumulate vectors in each centroid mapping.
        for mi in 1:m
            ci           = cmap[mi]
            XCS[:, ci] .+= X[:, mi]
            cntC[ci]    += 1
        end

        # Compute new centroids by averaging associated points.
        @inbounds for ci in unique(values(cmap))
            XCS[:, ci] ./= cntC[ci]
        end
    end
    return (cmap, XCS, tv_last, setdiff(1:k, unique(values(cmap))), N, false)
end



"""
    find_best_info_for_ks(X, kRng[; dmetric=L2, threshold=1.0e-3, W, N=1000, num_trials=100, seed=1])

Groups a set of `m` points (`n`-vectors) as an (nxm) matrix, `X`, into `k` clusters where `k` is in the range, `kRng`.
The groupings are determined based on the distance metric, `dmetric`.

# Type Constraints
- `T <: Real`
- `F <: Function`

# Arguments
- `X::Matrix{T}`           : (n,m) Matrix representing `m` points of dimension `n`.
- `kRng::UnitRange{Int64}` : The number of clusters to form.

# Keyword Arguments
- `dmetric::F=L2`          : The distance metric to use.
- `threshold::Float=1.0e-2`: The relative error improvement threshold (using total variation)
- `W::Union{Nothing, AbstractMatrix{T}}=nothing` : Optional Weight matrix for metric.
- `N::Int64=1000`          : The maximum number of kmeans_clustering iterations to try for each cluster number.
- `num_trials::Int64=300`  : The number of times to run kmeans_clustering for a given cluster number. 
- `seed::Int64=1`          : The random seed to use. (Used by kmeans_cluster to do initial clustering.)
    
# Input Contract
- ``W = {\\rm nothing} ∨ \\left( ({\\rm typeof}(W) = {\\rm Matrix}\\{T\\}) ∧ W \\in {\\boldsymbol S}_{++}^{n} \\right)``
- ``N > 0``
- ``∀ i \\in {\\rm kRng}, i \\ge 1``
- `threshold > 0.0`

# Return
A Tuple with entries:
- `OrderedDict{Int64, Float}`         : 1:k -> The Total Variation for each cluster number.
- `OrderedDict{Int64, Vector{Int64}}` : 1:k -> Mapping of index of points (n-vectors in `X`) to centroid indices.
- `OrderedDict{Int64, Matrix{T}}`     : 1:k -> (nxk) Matrix representing `k` `n`-vector centroids.
- `OrderedDict{Int64, Vector{In64}}`  : 1:k -> Vector of unused centroids by index.
"""
function find_best_info_for_ks(X::Matrix{T},
                               kRng::UnitRange{Int64};
                               dmetric::F=L2,
                               threshold::Float64=1.0e-3,
                               W::Union{Nothing,AbstractMatrix{T}}=nothing,
                               N::Int64=1000,
                               num_trials::Int64=300,
                               seed::Int64=1) where {T<:Real,F<:Function}

    tv_by_k   = DS.OrderedDict{Int64,T}()
    cmap_by_k = DS.OrderedDict{Int64,Vector{Int64}}()
    XC_by_k   = DS.OrderedDict{Int64,Matrix{T}}()
    ucnt_by_k = DS.OrderedDict{Int64,Vector{Int64}}()
    tmax = typemax(T)
    cnt = 0
    _, m = size(X)

    # Check input contract -- except the contract for the matrix `W`.
    if N <= 0
        throw(DomainError(N, "The parameter `N` is not in the range: [1, ...)"))
    elseif threshold <= 0.0
        throw(DomainError(threshold, "The parameter `threshold` is not in the range: (0, ...)"))
    elseif length(setdiff(collect(kRng), collect(1:m))) != 0
        throw(DomainError(typeof(kRng), 
            """The variable, `kRng`, has at least one value in its range 
               that is not in the discrete interval [1, m]. Here `m` is the number 
               of points in the data matrix `X`."""))
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
    @inbounds for k in kRng
        tv_by_k[k] = tmax
        for _ in 1:num_trials
            cnt += 1
            cmap, XC, tv, ucnt, N, _ = kmeans_cluster(X, k               ;
                                                      dmetric=dmetric    ,
                                                      threshold=threshold,
                                                      W=W                ,
                                                      N=N                ,
                                                      seed=(seed+cnt)     )
            if tv < tv_by_k[k]
                tv_by_k[k]   = tv
                cmap_by_k[k] = cmap
                XC_by_k[k]   = XC
                ucnt_by_k[k] = ucnt
            end
        end
    end

    return (tv_by_k, cmap_by_k, XC_by_k, ucnt_by_k)

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

# Type Constraints
- `T <: Real`
- `F <: Function`

# Arguments
- `X::Matrix{T}`           : (n,m) Matrix representing `m` points of dimension `n`.
- `kRng::UnitRange{Int64}` : The range of potential cluster values to try.

# Keyword Arguments
- `dmetric::F=L2`          : The distance metric to use.
- `threshold::Float=1.0e-2`: The relative error improvement threshold (using total variation)
- `W::Union{Nothing, AbstractMatrix{T}}=nothing` : Optional Weight matrix for metric.
- `N::Int64=1000`          : The maximum number of kmeans_clustering iterations to try for each cluster number.
- `num_trials::Int64=300`  : The number of times to run kmeans_clustering for a given cluster number. 
- `seed::Int64=1`          : The random seed to use. (Used by kmeans_cluster to do initial clustering.)
- `verbose::Bool=false`    : If `true`, print diagnostic information.
    
# Input Contract
- ``W = {\\rm nothing} ∨ \\left( ({\\rm typeof}(W) = {\\rm Matrix}\\{T\\}) ∧ W \\in {\\boldsymbol S}_{++}^{n} \\right)``
- `N > 0`
- ``∀ i \\in {\\rm kRng}, i \\ge 1``
- `threshold > 0.0`

# Return
A Tuple:
- `Int64`             : The "best" cluster number, `k`.
- `Dict{Int64, Int64}`: Mapping of points (`n`-vectors) indices to centroid indices.
- `Matrix{T}`         : Cluster centroids, represented as an `(n,k)` matrix.
- `Float64`           : The total variation between points and their centroids (using `dmetric`).
"""
function find_best_cluster(X::Matrix{T},
                           kRng::UnitRange{Int64};
                           dmetric::F=L2,
                           threshold::Float64=1.0e-3,
                           W::Union{Nothing,AbstractMatrix{T}}=nothing,
                           N::Int64=1000,
                           num_trials::Int64=300,
                           seed::Int64=1,
                           verbose::Bool=false ) where {T<:Real, F<:Function}

    _, m = size(X)

    # Check input contract -- except the matrix `W`.
    if N <= 0
        throw(DomainError(N, "The parameter `N` is not in the range: [1, ...)"))
    elseif threshold <= 0.0
        throw(DomainError(threshold, "The parameter `threshold` is not in the range: (0, ...)"))
    elseif length(setdiff(collect(kRng), collect(1:m))) != 0
        throw(DomainError(typeof(kRng), 
            """The variable, `kRng`, has at least one value in its range 
               that is not in the discrete interval [1, m]. Here `m` is the number 
               of points in the data matrix `X`."""))
    end

    # Get the info for the best clusters in the range: `kRng`.
    tv, cmap, xc, unct = find_best_info_for_ks(X, kRng              ;
                                               dmetric=dmetric      ,
                                               threshold=threshold  ,
                                               W=W                  ,
                                               N=N                  ,
                                               num_trials=num_trials,
                                               seed=seed             )

    # Get the dimension of the points.
    n, _ = size(X)

    # Used to adjust to cluster variation by data dimension and
    # number of clusters.
    kfact = map(j -> j^(1.0 / n), kRng)

    # Get all of the cluster choices.
    mv = collect(values(kRng))

    # Get the total variation for each cluster number.
    tvv = collect(values(tv))

    # Get the number of unused cluster nodes for each cluster number.
    unctv = length.(collect(values(unct)))

    # Adjust the total variation, `tvv`, by `kfact`.
    # The `kfact` values adjust for the natural tendency for more clusters
    # to give less total variation.
    # Also, penalize the variation by multiplying by a fraction that
    # takes into account unused centroids.
    var_by_k_mod = tvv .* kfact .* (mv .+ unctv) ./ mv

    if verbose
        println("var_by_k     = $tvv")
        println("var_by_k_mod = $var_by_k_mod")
        if length(var_by_k_mod) > 1
            println("rel change of var $(diff(var_by_k_mod) ./ var_by_k_mod[2:end])") 
        end
    end

    # Find the cluster number with the largest relative decrease in 
    # adjusted total variation.
    kbest = kRng.start 
    vlen = length(var_by_k_mod)
    min_idx = Vector{Int64}(undef, vlen)
    mono_var_by_k_mod = Vector{Float64}(undef, vlen)
    @inbounds if vlen > 1
        monvar = var_by_k_mod[1]
        last_min_idx = 1
        for (l,v) in enumerate(var_by_k_mod)
            min_idx[l] = last_min_idx
            v = min(v, monvar)
            mono_var_by_k_mod[l] =  v
            if v < monvar 
                last_min_idx = l 
                monvar  = v
                min_idx[l] = l
            end
        end
        kbest = argmin(diff(mono_var_by_k_mod) ./ mono_var_by_k_mod[2:end] ./ (1.0 .+ diff(min_idx))) + kRng.start 
        if verbose
            println("mono_var_by_mod: $mono_var_by_k_mod")
            println("mono_var_series: $(diff(mono_var_by_k_mod) ./ mono_var_by_k_mod[2:end] ./ (1.0 .+ diff(min_idx)))")
        end
    end
        
    # Number of unused clusters in best cluster.
    unct_len = length(unct[kbest])

    # If no unused centroids, return.
    if unct_len == 0
        return (kbest, cmap[kbest], xc[kbest], tv[kbest])
    end

    # Else we need to remove unused centroids and re-index the used centroids.
    viable_centroid_idxs = setdiff(1:kbest, unct[kbest])
    reindex_centroids = DS.OrderedDict{Int64, Int64}()
    bcmap = DS.OrderedDict{Int64, Int64}()
    cnt = 1
    for i in viable_centroid_idxs
        reindex_centroids[i] = cnt
        cnt += 1
    end

    if verbose
        println("kbest is $kbest; however, there are $unct_len centroids with no associated points -- re-adjusting...")
    end

    # Remap the points to the index of the nearest centroid using the re-index map.
    @inbounds for k in keys(cmap[kbest])
        bcmap[k] = reindex_centroids[cmap[kbest][k]]
    end
    
    # Return (number-of-clusters, map-of-point-to-cluster-index, clusters, total-variation-of-fit)
    return (length(viable_centroid_idxs), bcmap, xc[kbest][:, viable_centroid_idxs], tv[kbest])

end

end # module Cluster

