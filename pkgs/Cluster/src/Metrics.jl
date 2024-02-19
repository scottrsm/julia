module Metrics

# Export metrics metrics: L_2, L_p, L_∞, Kullback-Leibler, Cosine, and Jaccard.
# and fit metrics: raw_confusion_matrix, confusion_matrix
export L2, LP, LI, KL, CD, JD, raw_confusion_matrix, confusion_matrix

import LinearAlgebra as LA


const TOL=1.0e-6

"""
    L2(x,y[; M=nothing])

Computes the ``L_2`` distance between two vectors.
One of the features that may be different from other packages
is the use of weighted metrics in some instances.

# Type Constraints
- `T <: Real`

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.

# Keyword Arguments
- `C::Union{Nothing, Matrix{T}` : Optional Weight matrix.

# Input Contract (Low level function -- Input contract not checked)
- ``|{\\bf x}| = |{\\bf y}|``
- ``M = {\\rm nothing} \\vee \\left( ({\\rm typeof}(M) = {\\rm Matrix}\\{T\\}) \\wedge M \\in {\\boldsymbol S}_{++}^{|{\\bf x}|} \\right)``

# Return
``L_2`` (optionally weighted) distance measure between the two vectors.
"""
function L2(x::AbstractVector{T},
            y::AbstractVector{T};
            M::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:Real}

    d = x .- y
    if M === nothing
        return LA.norm2(d)
    else
        return sqrt(LA.dot(d, M, d))
    end
end

"""
    LP(x,y,p)

Computes the ``L_p`` distance between two vectors.

# Type Constraints
- `T <: Real`

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.
- `p::Int64`     : The power of the norm.

# Input Contract (Low level function -- Input contract not checked)
- ``|{\\bf x}| = |{\\bf y}|``
- `p > 0`

# Return
``L_p`` distance measure between the two vectors.
"""
function LP(x::AbstractVector{T},
            y::AbstractVector{T},
            p::Int64     ) where {T <: Real}

    return LA.norm(x .- y, p)
end

"""
    LI(x,y)

Computes the ``L_\\infty`` distance between two vectors.

# Type Constraints
- `T <: Real`

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.

# Input Contract (Low level function -- Input contract not checked)
- ``|{\\bf x}| = |{\\bf y}|``

# Return
``L_\\infty`` distance measure between the two vectors.
"""
function LI(x::AbstractVector{T},
            y::AbstractVector{T} ) where {T <: Real}

    return max.(abs.(x .- y))
end


"""
    JD(x,y)

Computes the `Jaccard` metric between two vectors of a "discrete" type.
For instance, the vectors could be integers; however, they can 
also be of non-numeric type. The metric can also be used with 
floating point values but, in that case, it may be more useful 
to round/truncate to a particular "block" size.

If both `x` and `y` are vectors of zero length, a distance of ``0`` is returned.

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.

# Return
`Jaccard` distance measure between the two vectors.
"""
function JD(x::AbstractVector{T},
            y::AbstractVector{T} ) where {T <: Real}
    d = length(symdiff(x,y))
    u = length(union(x,y)) 

    return length(u) == 0 ? 0.0 : d / u
end


"""
    KL(x,y)

Computes the ``Kullback-Leibler`` distance between two vectors.

# Type Constraints
- `T <: Real`

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.

# Input Contract (Low level function -- Input contract not checked)
Let ``N = |{\\bf x}|``.
- ``|{\\bf x}| = |{\\bf y}|``
- ``\\forall i \\in [1, N]: x_i \\ge 0``
- ``\\forall i \\in [1, N]: y_i \\ge 0``
- ``\\sum_{i=1}^N x_i = 1``
- ``\\sum_{i=1}^N y_i = 1``

# Return
`KL` distance measure between the two vectors.
"""
function KL(x::AbstractVector{T},
            y::AbstractVector{T} ) where {T <: Real}

    z = zero(T)
    d1 = map((a, b) -> a == z ? z : a * log(a / b), x, y)
    d2 = map((a, b) -> b == z ? z : b * log(b / a), x, y)

    return sum(d1 .+ d2)
end


"""
    CD(x,y[; M=nothing])

Computes the "cosine" distance between two vectors.

# Type Constraints
- `T <: Real`

# Arguments
- `x::AbstractVector{T}` : A numeric vector.
- `y::AbstractVector{T}` : A numeric vector.

# Keyword Arguments
- `M::Union{Nothing, Matrix{T}` : Optional Weight matrix.

# Input Contract (Low level function -- Input contract not checked)
- ``|{\\bf x}| = |{\\bf y}|``
- ``M = {\\rm nothing} \\vee \\left( ({\\rm typeof}(M) = {\\rm Matrix}\\{T\\}) \\wedge M \\in {\\boldsymbol S}_{++}^{|{\\bf x}|} \\right)``

# Return
Cosine distance measure between the two vectors.

"""
function CD(x::AbstractVector{T},
            y::AbstractVector{T};
            M::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:Real}
    z = zero(T)
    o = one(T)
    tol = T(TOL)

    if all(abs.(x .- y) / (2.0 .* (abs.(x) .+ abs.(y))) .< tol)
        return o
    elseif all(abs.(x) .< tol)
        return o
    elseif all(abs.(y) .< tol)
        return o
    elseif M === nothing
        return o - LA.dot(x, y) / sqrt(LA.dot(x, x) * LA.dot(y, y))
    end

    return o - LA.dot(x, M, y) / sqrt(LA.dot(x, M, x) * LA.dot(y, M, y))
end


"""
    raw_confusion_matrix(act, pred)

Computes a confusion matrix of the discrete variables, `act` and `pred`.
There are no row or column labels for this matrix.

# Type Constraints:
- Expects types, `A` and `P` to have discrete values.

# Arguments
- `act ::AbstractVector{A}` : A vector of the actual target values.
- `pred::AbstractVector{P}` : A vector of the predicted target values.

# Input Contract:
- |act| = |pred|

# Return
Matrix{Int64}: A 3-tuple consisting of:
- Vector of unique values of `act`.  (Sorted from lowest to highest, otherwise the order returned from unique.)
- Vector of unique values of `pred`. (Sorted from lowest to highest, otherwise the order returned from unique.)
- A matrix of counts for all pairings of discrete values of `act` with `pred`.
"""
function raw_confusion_matrix(act::AbstractVector{A}, pred::AbstractVector{P}) where {A, P}
    N = length(act) 

    # Check Input Contract:
    if length(pred) != N
        throw(DomainError(N, "confusion_matrix: Vector inputs, `act` and `pred` do NOT have the same length"))
    end
    # Get unique values of actual values and their associated length.
    a_vals = unique(act)
    try 
        if ! ( a_vals[1] < a_vals[1] )
            sort!(a_vals)
        end
    catch 
    end

    a_N = length(a_vals)

    # Get unique values of predicted values and their associated length.
    p_vals = unique(pred)
    try 
        if ! ( p_vals[1] < p_vals[1] )
            sort!(p_vals)
        end
    catch 
    end

    p_N = length(p_vals)

    # Confusion Matrix -- to be filled in.
    CM = fill(0, a_N, p_N)
    da = Dict{A, Int64}()
    dp = Dict{P, Int64}()

    # Map the actual values to index order as assigned by either sort;
    # or, in case the values are not sortable, the function unique.
    @inbounds for i in 1:a_N
        da[a_vals[i]] = i
    end
    
    # Map the predicted values to index order as assigned by either sort;
    # or, in case the values are not sortable, the function unique.
    @inbounds for i in 1:p_N
        dp[p_vals[i]] = i
    end

    # Fill in the non-zero entries of the confusion matrix
    # as the number of counts for each pair of (actual, predicted) pairings
    # as encoded by the actual and predicted index values.
    @inbounds for i in 1:N
        CM[da[act[i]], dp[pred[i]]] += 1
    end

    # Return the confusion matrix along with the associated 
    # ordered actual and predicted values.
    return (a_vals, p_vals, CM)    
end



"""
    confusion_matrix(act, pred)

Computes the confusion matrix of the *discrete* variables, `act` and `pred`.

# Type Constraints:
- Expects types, `A` and `P` to have discrete values.

# Arguments
- `act ::AbstractVector{A}` : A vector of the actual target values.
- `pred::AbstractVector{P}` : A prediction vector for the target.

# Input Contract:
- |act| = |pred|

# Return
Matrix{Any}:
- The raw confusion matrix augmented by a column on the left listing 
  all *actual* values (in sorted order if sortable) and 
  augmented on top with a row listing 
  all *predicted* values (in sorted order if sortable).
"""
function confusion_matrix(act::AbstractVector{A}, pred::AbstractVector{P}) where {A, P}
    res = confusion_matrix(act, pred)
    N, M = size(res[3])
    PM = Matrix(undef, N+1, M+1)

    PM[2:N+1, 2:M+1] = copy(res[3]) # Fill in confusion matrix in lower right of `PM`.
    PM[2:N+1, 1    ] = res[1]       # Fill in actual values on the left of `PM`.
    PM[1    , 2:M+1] = res[2]       # Fill in predicted values on the top of `PM`.

    # Upper left hand label: Act versus Pred.
    PM[1, 1] = "ACT\\PRED"

    return PM
end


end # module Metrics
