module DNNS

export AD, PWL, DLayer, DNN, sigmoid1, sigmoid2, sigmoid3, relu, relur, loss, fit

import Base: promote_rule, show
import Base: +, -, *, /, ^, exp, log
import Base: sin, cos, tan, csc, sec, cot
import Base: sinh, cosh, tanh, csch, sech, coth
import Base: asin, acos, atan, acsc, asec, acot

import LinearAlgebra as LA
import StatsBase: sample

# Module constants
const X_REL_TOL = 1.0e-10

"""
    AD{T}

Automatic differentiation structure.

Fields
- v :: T -- The value of this structure.
- d :: T -- The derivative at this value.

"""
mutable struct AD{T<:Number} <: Number
    v::T
    d::T

    # Inner Constructors.
    AD{T}(nv::T, nd::T) where {T<:Number} = new{T}(nv, nd)

    AD{T}(nv::T) where {T<:Number} = new{T}(nv, zero(T))
    function AD{T}(ad::AD{S}) where {T<:Number,S<:Number}
        W = promote_type(S, T)
        return new{W}(convert(W, ad.v), convert(W, ad.d))
    end

    function AD{T}(nv::S) where {T<:Number,S<:Number}
        W = promote_type(S, T)
        return AD{W}(convert(W, nv), zero(W))
    end
end

# Outer Constructors.
AD(nv::T; var::Bool=false) where {T<:Number} = var ? AD{T}(nv, one(T)) : AD{T}(nv, zero(T))
AD(nv::T, nd::T) where {T<:Number} = AD{T}(nv, nd)
AD(nv::T, nd::S) where {T<:Number,S<:Number} = AD(Base.promote(nv, nd)...)

# Show values of AD.
Base.show(io::IO, x::AD{T}) where {T<:Number} = print(io, "($(x.v), $(x.d))")


Base.promote_rule(::Type{AD{T}}, ::Type{T}) where {T<:Number} = AD{T}
Base.promote_rule(::Type{AD{T}}, ::Type{S}) where {T<:Number,S<:Number} = AD{Base.promote_type(T, S)}
Base.promote_rule(::Type{AD{T}}, ::Type{AD{S}}) where {T<:Number,S<:Number} = AD{Base.promote_type(T, S)}


"""
    PWL{T}

A structure representing a piece-wise linear function.

In practice, one uses one of two outer constructors to create a `PWL` struct.

## Type Constraints
- T <: Number
- The type T must have a total ordering.

## Fields
- `xs :: Vector{T}`   -- The "x" values.
- `ys :: Vector{T}`   -- The "y" values.
- `ds :: Vector{T}`   -- The "slopes" of the segments.
- `n  :: Int64`       -- The number of "x,y" values.
                         

## Public Constructors
`PWL(xs::Vector{T}, y::T, ls::Vector{T})` 
- `xs` -- The `x` coordinates in ascending order -- no duplicates.
- `y`  -- The value of `y` corresponding to the first entry in `xs`.
- `ls` -- The slope values before the first and last `xs` value.
`PWL(xs::Vector{T}, nys::Vector{T}, nls::Vector{T})` 
- `xs` -- The `x` coordinates in ascending order -- no duplicates.
- `ys` -- The `y` coordinates corresponding to each `x` value.

## Examples
```jdoctest
julia> # Create the same (in behavior) Piecewise linear functions in two ways:
julia> pw1 = PWL([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [0.0, 5.0])

julia> pw2 = PWL([1.0, 2.0, 3.0], 2.0, [0.0, 1.0, 1.0, 5.0])

julia> pw1(2.5)
3.5

julia> pw2(2.5)
3.5
```
"""
struct PWL{T<:Number}
    xs::Vector{T}
    ys::Vector{T}
    ls::Vector{T}
    n::Int64

    # Inner Constructor.
    function PWL{T}(nxs::Vector{T}, nys::Vector{T}, nls::Vector{T}) where {T<:Number}
        try
            zero(T) < one(T)
        catch
            throw(DomainError("PWD{T}: (Inner Constructor) Type $T does not have a total ordering"))
        end

        xmin, xmax = extrema(nxs)
        tol = X_REL_TOL * max(abs(xmin), abs(xmax))

        n = length(nxs)
        if any(diff(nxs) .- tol .<= zero(T))
            throw(DomainError("PWD{T}: (Inner Constructor) `nxs` is not sorted or has duplicates"))
        end

        if length(nls) != 2
            throw(DomainError("PWD{T}: (Inner Constructor) `nls` vector must have a length of 2"))
        end

        n = length(nxs)
        if n != length(nys)
            throw(DomainError("PWD{T}: (Inner Constructor) `nxs` and `nys` vectors must have the same length"))
        end

        # Compute the interior slopes.
        dxs = diff(nys) ./ diff(nxs)

        W = eltype(dxs)
        if W != T
            nxs = convert.(W, nxs)
            nys = convert.(W, nys)
            dxs = convert.(W, dxs)
            nls = convert.(W, nls)
        end

        ls = Vector{W}(undef, n + 1)
        ls[1] = nls[1]
        ls[n+1] = nls[2]
        ls[2:n] = dxs

        return new{W}(copy(nxs), copy(nys), ls, n)

    end

    # Inner Constructor.
    function PWL{T}(nxs::Vector{T}, ny::T, nls::Vector{T}) where {T<:Number}
        n = length(nxs)
        xmin, xmax = extrema(nxs)
        tol = X_REL_TOL * max(abs(xmin), abs(xmax))

        if any(diff(nxs) .- tol .<= zero(T))
            throw(DomainError("PWD{T}: (Inner Constructor) `nxs` is not sorted or has duplicates."))
        end

        if (length(nls) - n) != 1
            throw(DomainError("PWD{T}: (Inner Constructor) `nls` vector length must be 1 larger than `nxs` vector length."))
        end

        nys = zeros(T, length(nxs))
        lasty = ny
        nys[1] = lasty

        if isbitstype(T)
            for i in 2:n
                lasty += (nxs[i] - nxs[i-1]) * nls[i]
                nys[i] = lasty
            end
        else
            for i in 2:n
                lasty += (nxs[i] - nxs[i-1]) * nls[i]
                nys[i] = deepcopy(lasty)
            end
        end

        new(copy(nxs), nys, copy(nls), n)
    end
end

# Outer Constructors.
PWL(nxs::Vector{T}, ny::T, nls::Vector{T}) where {T<:Number} = PWL{T}(nxs, ny, nls)

# Outer constructor.
PWL(nxs::Vector{T}, nys::Vector{T}, nls::Vector{T}) where {T<:Number} = PWL{T}(nxs, nys, nls)



"""
    (p)(x)

Uses the structure `PWL` as a piece-wise linear function. 

# Type Constraints
- `T <: Number`

# Arguments
- `x :: T`  -- An input value.

# Return
`:: T`
"""
function (p::PWL{T})(x::T) where {T<:Number}

    idx = searchsorted(p.xs, x)
    u = first(idx)
    l = last(idx)

    l += l == 0 ? 1 : 0

    return p.ys[l] + p.ls[u] * (x - p.xs[l])
end


struct DLayer{T<:Number}
    M::Matrix{AD{T}}
    b::Vector{AD{T}}
    op::Function
    dims::Tuple{Int64,Int64}

    function DLayer{T}(Mn::Matrix{T}, bn::Vector{T}, opn::Function) where {T<:Number}
        n, m = size(Mn)
        length(bn) == n || error("DLayer (Inner Constructor): Matrix, `Mn`, and vector, `bn`, are incompatible.")

        return new{T}(AD{T}.(Mn), AD{T}.(bn), opn, (n, m))
    end
end

# Outer constructor
DLayer(Mn::Matrix{T}, bn::Vector{T}, opn) where {T<:Number} = DLayer{T}(Mn, bn, opn)


"""
    (L)(x)

Treats the structure `DLayer` as a function: ``{\\cal R}^m \\mapsto {\\cal R}^n``
Takes input `x` and passes it through the layer.

# Type Constraints
- `T <: Number`

# Arguments
- `x :: AD{T}`  -- An input value.

# Return
`::Vector{AD{T}}` of dimension ``dims[1]`
"""
function (L::DLayer{T})(x::AD{T}) where {T<:Number}
    length(x) == L.dims[1] || error("DLayer (As Function): Vector `x` is incompatible with layer dimensions.")

    return L.op.(L.M * x .+ L.b)
end

struct DNN{T<:Number}
    layers::Vector{DLayer{T}}

    function DNN{T}(ls::Vector{DLayer{T}}) where {T<:Number}
        length(ls) != 0 || error("DNN (Inner Constructor): Length of ls is 0.")
        for i in eachindex(ls[1:end-1])
            ls[i].dims[1] == ls[i+1].dims[2] || error("DNN (Inner Constructor): DLayer incompatibility between layers $i and $(i+1).")
        end
        return new{T}(ls)
    end
end

# Outer Contructor
DNN(ls::Vector{DLayer{T}}) where {T<:Number} = DNN{T}(ls)

"""
    (DNN)(x)

Treats the structure `DNN` as a function: ``{\\cal R}^m \\mapsto {\\cal R}^n``
Takes input `x` and passes it through each of the layers of DNN.

# Type Constraints
- `T <: Number`

# Arguments
- `x :: T`  -- An input value.

# Return
`::Vector{AD{T}}` of dimension
"""
function (dnn::DNN{T})(x::AbstractVector{T}) where {T<:Number}

    _, n = size(dnn.layers[1].M)
    length(x) == n || error("DNN (as function): Matrix from first layer is incompatible with `x`.")

    for i in eachindex(dnn.layers)
        x = dnn.layers[i].op.(dnn.layers[i].M * x .+ dnn.layers[i].b)
    end

    return x
end


"""
    make_const!(l)

Makes the parameters in the layers constants -- for the purpose of differentiation.

# Type Constraints
- T <: Number

# Arguments
- l :: DLayer{T,F}

# Return
nothing
"""
function make_const!(l::DLayer{T}) where {T<:Number}
    t0 = zero(T)
    n, m = l.dims
    @inbounds for i in 1:n
        l.b[i].d = t0
    end

    @inbounds for i in 1:n
        for j in 1:m
            l.M[i, j].d = t0
        end
    end

    return nothing
end

"""
    set_bd_pd!l, k, d)

Sets the derivative of one of the elements of the bias vector in the layer.

# Type Constraints
- T <: Number

# Arguments
- l :: DLayer{T,F} -- A DNN layer
- k :: Int64{T,F}  -- The index to access the layer biases vector.
- d :: T           -- The value of the derivative to set at index `k`.

# Return
nothing
"""
function set_bd_pd!(l::DLayer{T}, k::Int64, d::T) where {T<:Number}
    l.b[k].d = d

    return nothing
end


"""
    set_md_pd!l, k, d)

Sets the derivative of one of the elements of the matrix in the layer.

# Type Constraints
- T <: Number

# Arguments
- l :: DLayer{T,F} -- A DNN layer.
- k :: Int64{T,F}  -- The index to access the layer matrix.
- d :: T           -- The value of the derivative to set at index `k`.

# Return
nothing
"""
function set_md_pd!(l::DLayer{T}, k::Int64, d::T) where {T<:Number}
    l.M[k].d = d

    return nothing
end


"""
    loss(dnn, X, Y)

Computes the loss of the neural network given inputs, `X`, and outputs `Y`.

# Type Constraints
- T <: Number

# Arguments
- dnn :: DNN{T,F}   -- A DNN layer.
- X   :: Matrix{T}  -- The matrix of input values.
- Y   :: Matrix{T}  -- The matrix of output values.

# Return
::AD{T} -- The loss of the network
"""
function loss(dnn::DNN{T}, X::Matrix{T}, Y::Matrix{T}) where {T<:Number,F<:Function}
    _, m = size(X)
    _, my = size(Y)
    m == my || error("fit: Dimensions of `X` and `Y` are incompatible.")

    s = zero(AD{T})
    @inbounds for i in 1:m
        df = dnn(@view X[:, i]) .- (@view Y[:, i])
        s += LA.dot(df, df)
    end

    return s
end


"""
    fit(dnn, X, Y)

Adjusts the parameters of neural network, `dnn`, to get the best fit of 
the data: `X`, `Y`. The parameters of the network are all paris of 
matrices and biases for each layer in the network.

# Type Constraints
- T <: Number

# Arguments
- dnn :: DNN{T,F}   -- A DNN layer.
- X   :: Matrix{T}  -- The matrix of input values.
- Y   :: Matrix{T}  -- The matrix of output values.

# Return
::nothing
"""
function fit(dnn::DNN{T}, X::Matrix{T}, Y::Matrix{T};
    N=1000, relerr=1.0e-6, μ=1.0e-3, verbose=false) where {T<:Number}

    _, m = size(X)
    _, my = size(Y)

    m == my || error("fit: Arrays, `X`, and `Y`, are incompatible.")

    lss_last = Inf
    lss = Inf
    finished_early = false
    rel_err = Inf
    num_iterates = N
    mu = μ
    for j in 1:N
        if N % 1000 == 0
            mu = μ
        end
        μ *= 0.999
        rel_err = abs((lss - lss_last) / lss_last)
        if !isnan(rel_err) && j > 20 && rel_err <= relerr
            finished_early = true
            num_iterates = j
            break
        end
        verbose && println("Iteration $(j): loss = $lss")
        lss_last = lss
        # Walk over each layer...
        for i in eachindex(dnn.layers)
            brat = length(dnn.layers[i].M) / length(dnn.layers[i].b)
            # Treat the M and b parameters for this layer as constants.
            make_const!(dnn.layers[i])

            # Selectively treat the kth element of M as a variable so that
            # we may take the partial derivative with respect to M[k].
            for k in eachindex(dnn.layers[i].M)
                set_md_pd!(dnn.layers[i], k, one(T))
                ls = loss(dnn, X, Y)
                set_md_pd!(dnn.layers[i], k, zero(T))
                dnn.layers[i].M[k].v -= ls.d * mu
            end

            # Selectively treat the kth element of b as a variable so that
            # we may take the partial derivative with respect to b[k].
            for k in eachindex(dnn.layers[i].b)
                set_bd_pd!(dnn.layers[i], k, one(T))
                ls = loss(dnn, X, Y)
                set_bd_pd!(dnn.layers[i], k, zero(T))
                lss = ls.v
                dnn.layers[i].b[k].v -= ls.d * brat * mu
            end
        end
    end
    if finished_early
        println("Total number of iterates tried = $num_iterates from a max of $N.")
    else
        println("Used the maximum nunmber of iterates ($N).")
        println("The relerr = $rel_err.")
    end
end




# --------------------------------------------------------------------
# ------------  Overload Math Functions for AD  ----------------------
# --------------------------------------------------------------------
# Binary operators below are defined on two potentially different
# AD types: AD{T}, AD{S}.
# Note: Given the promote_type rules above, we can then do:
#       (operator)(x::AD{T}, y::S})
# The y variable can be promoted to AD{S}, then we have a method match.
# The method (+)(x::AD{T}, AD(y)::AD{S}) gets called.
# The first thing this function does is promote x,y to a promoted_type, W
# (which is invisible in the code) and then a value of type AD{W} is returned.
# --------------------------------------------------------------------


#-----------------------------------------------------------------
# -----    Standard Scalar Functions/Operators      --------------
#-----------------------------------------------------------------

# -----  Standard Binary Functions  --------------------------
# Operators: +. -, *, /, ^
function Base.:(+)(x::AD{T}, y::AD{S}) where {T<:Number,S<:Number}
    xp, yp = promote(x, y)
    AD(xp.v + yp.v, xp.d + yp.d)
end

function Base.:(-)(x::AD{T}, y::AD{S}) where {T<:Number,S<:Number}
    xp, yp = promote(x, y)
    AD(xp.v - yp.v, xp.d - yp.d)
end

function Base.:(*)(x::AD{T}, y::AD{S}) where {T<:Number,S<:Number}
    xp, yp = promote(x, y)
    AD(xp.v * yp.v, xp.v * yp.d + xp.d * yp.v)
end

function Base.:(/)(x::AD{T}, y::AD{S}) where {T<:Number,S<:Number}
    xp, yp = promote(x, y)
    yp == zero(eltype(yp.v)) && error("AD: 'y.v' == 0 is not a valid value for x.v / y.v.")
    AD(xp.v / yp.v, xp.d / yp.v - (xp.v * yp.d) / (yp.v * yp.v))
end

function Base.:(^)(x::AD{T}, y::AD{S}) where {T<:Number,S<:Number}
    xp, yp = promote(x, y)
    x.v == zero(eltype(xp)) && error("AD: 'x.v' == 0 is not a valid value for x.v^y.v.")
    t = xp.v^yp.v
    AD(t, t * (yp.d * log(xp.v) + (yp.v * xp.d) / xp.v))
end


# ----- Standard Unary Scalar Functions  --------------------------
# Unary minus
function Base.:(-)(x::AD{T}) where {T<:Number}
    AD(-x.v, -x.d)
end

# exp, log
function Base.exp(x::AD{T}) where {T<:Number}
    et = exp(x.v)
    AD(et, et * x.d)
end

function Base.log(x::AD{T}) where {T<:Number}
    x.v == zero(T) && error("AD: 'x.v' == 0 is not a valid value for log(x.v).")
    AD(log(x.v), x.d / x.v)
end

# sin, cos, tan
Base.sin(x::AD{T}) where {T<:Number} = AD(sin(x.v), cos(x.v) * x.d)

Base.cos(x::AD{T}) where {T<:Number} = AD(cos(x.v), -sin(x.v) * x.d)

function Base.tan(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi / 2))
    mx == zero(T) && error("AD: 'x.v' mod π / 2 == 0 is not a valid value for tan(x.v).")
    s = sec(mx)
    t = tan(mx)
    AD(t, s * s * x.d)
end

# csc, sec, cot
function Base.csc(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi))
    mx == zero(T) && error("AD: 'x.v' mod π == 0 is not a valid value for csc(x.v).")
    ct = cot(mx)
    c = csc(mx)
    AD(c, -c * ct * x.d)
end

function Base.sec(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi / 2))
    mx == zero(T) && error("AD: 'x.v' mod π / 2 == 0 is not a valid value for sec(x.v).")
    s = sec(mx)
    t = tan(mx)
    AD(s, s * t * x.d)
end

function Base.cot(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi))
    mx == zero(T) && error("AD: 'x.v' mod π == 0 is not a valid value for cot(x.v).")
    c = csc(mx)
    ct = cot(mx)
    AD(ct, -c * c * x.d)
end

# sinh, cosh, tanh
function Base.sinh(x::AD{T}) where {T<:Number}
    ch = cosh(x.v)
    sh = sinh(x.v)
    AD(sh, ch * x.d)
end

function Base.cosh(x::AD{T}) where {T<:Number}
    ch = cosh(x.v)
    sh = sinh(x.v)
    AD(ch, sh * x.d)
end

function Base.tanh(x::AD{T}) where {T<:Number}
    sh = sech(x.v)
    th = tanh(x.v)
    AD(th, sh * sh * x.d)
end

# csch, sech, coth
function Base.csch(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi))
    mx == zero(T) && error("AD: 'x.v' mod π == 0 is not a valid value for csch(x.v).")
    ch = csch(mx)
    ct = coth(mx)
    AD(ch, -ch * ct * x.d)
end

function Base.sech(x::AD{T}) where {T<:Number}
    sh = sech(x.v)
    th = tanh(x.v)
    AD(sh, -sh * th * x.d)
end

function Base.coth(x::AD{T}) where {T<:Number}
    mx = mod(x.v, T(pi))
    mx == zero(T) && error("AD: 'x.v' mod π == 0 is not a valid value for coth(x.v).")
    ch = csch(mx)
    ct = coth(mx)
    AD(ct, -ch * ch * x.d)
end

# Inverse trig functions: asin, acos, atan.
Base.asin(x::AD{T}) where {T<:Number} = AD(asin(x.v), one(T) / sqrt(one(T) - x.v * x.v))
Base.acos(x::AD{T}) where {T<:Number} = AD(acos(x.v), -one(T) / sqrt(one(T) - x.v * x.v))
Base.atan(x::AD{T}) where {T<:Number} = AD(atan(x.v), one(T) / (one(T) + x.v * x.v))

# Inverse trig functions: acsc, asec, acot.
function Base.acsc(x::AD{T}) where {T<:Number}
    abs(x.v) < one(T) && error("AD: '|x.v|' < 1 is not a valid value for asec(x.v).")
    AD(acsc(x.v), -one(T) / (x.v * sqrt(x.v * x.v - one(T))))
end

function Base.asec(x::AD{T}) where {T<:Number}
    abs(x.v) < one(T) && error("AD: '|x.v|' < 1 is not a valid value for asec(x.v).")
    AD(asec(x.v), one(T) / (abs(x.v) * sqrt(x.v * x.v - one(T))))
end

Base.acot(x::AD{T}) where {T<:Number} = AD(acot(x.v), -one(T) / (one(T) + x.v * x.v))


#-------------------------------------------------------------------
# ---------  Non Standard and Thresholding Functions  --------------
#-------------------------------------------------------------------

"""
    sigmoid1(x)

Implements an `AD` version of the standard "exponential" sigmoid function.
j
# Type Constraints
- T <: Number

# Arguments
- x   :: AD{T}  -- The `AD` input value.

# Return
::AD{T} -- The output AD value/derivative.
"""
function sigmoid1(x::AD{T}) where {T<:Number}
    on = one(T)
    v = on / (on + exp(-x.v))
    d = x.d * v * (on - v)

    return AD(v, d)
end


"""
    sigmoid2(x)

Implements an `AD` version of the standard "tanh" sigmoid function.

# Type Constraints
- T <: Number

# Arguments
- x   :: AD{T}  -- The `AD` input value.

# Return
::AD{T} -- The output AD value/derivative.
"""
function sigmoid2(x::AD{T}) where {T<:Number}
    v = tanh(x.v)

    return AD(v, x.d * (one(T) - v * v))
end


"""
    sigmoid3(x)

Implements an `AD` version of the standard "arctan" sigmoid function.

# Type Constraints
- T <: Number

# Arguments
- x   :: AD{T}  -- The `AD` input value.

# Return
::AD{T} -- The output AD value/derivative.
"""
function sigmoid3(x::AD{T}) where {T<:Number}
    t1 = one(T)
    v = x.v

    return AD(atan(v), x.d * (t1 / (t1 + v * v)))
end


"""
    relu(x)

Implements an `AD` version of the standard relu function.

# Type Constraints
- T <: Number

# Arguments
- x   :: AD{T}  -- The `AD` input value.

# Return
::AD{T} -- The output AD value/derivative.
"""
function relu(x::AD{T}) where {T<:Number}
    d = x.v <= 0 ? zero(T) : one(T)
    AD(x.v, d * x.d)
end


"""
    relur(x)

Implements an `AD` version of a nodified version of the relu function.
The modification is that while the value of the `relur` is the same as `relu`,
its derivative is not. The value of the derivative is `0` or `1`, however
the boundary moves randomly around the natural input boundary of `0`,
its derivative is not. The value of the derivative is `0` or `1`, however
the boundary moves randomly around the natural input boundary of `0`.

# Type Constraints
- T <: Number

# Arguments
- x   :: AD{T}  -- The `AD` input value.

# Return
::AD{T} -- The output AD value/derivative.
"""
# Thresholding function: Relur
function relur(x::AD{T}) where {T<:Number}
    d = x.v <= rand([-0.25, -0.1, -0.025, -0.01, 0.0, 0.01, 0.025, 0.1, 0.25]) ? zero(T) : one(T)
    d = x.v <= 0 ? zero(T) : one(T)
    AD(x.v, d * x.d)
end


"""
    (PWL)(x)

Uses the structure `PWL` as a piece-wise linear function. 

# Type Constraints
- `T <: Number`

# Arguments
- `x :: AD{T}`  -- An AutoDiff value.

# Return
`:: AD{T}`
"""
function (p::PWL{T})(x::AD{T}) where {T<:Number}

    Idx = searchsorted(p.xs, x.v)
    u = first(Idx)
    l = last(Idx)

    l += l == 0 ? 1 : 0

    return AD(p.ys[l] + p.ls[u] * (x.v - p.xs[l]), p.ls[u] * x.d)
end


#-------------------------------------------------------------------
# ---------       Matrix/Vector Functions               ------------
#-------------------------------------------------------------------

Base.zero(::Type{AD{T}}) where {T<:Number} = AD(zero(T), zero(T))
Base.zeros(::Type{AD{T}}, n::Int64) where {T<:Number} = fill(AD(zero(T), zero(T), n))
Base.one(::Type{AD{T}}) where {T<:Number} = AD(one(T), zero(T))
Base.ones(::Type{AD{T}}, n::Int64) where {T<:Number} = fill(AD(one(T), zero(T), n))


function LA.dot(x::Vector{AD{T}}, y::Vector{AD{T}}) where {T<:Number}
    n = length(x)
    if length(y) != n
        error("dot: Vector lengths are not the same.")
    end
    s = zero(AD{T})
    @inbounds @simd for i in 1:n
        s += x[i] * y[i]
    end
    return s
end

function LA.dot(x::Vector{AD{T}}, y::Vector{T}) where {T<:Number}
    n = length(x)
    if length(y) != n
        error("dot: Vector lengths are not the same.")
    end
    s = zero(AD{T})
    @inbounds @simd for i in 1:n
        s += x[i] * y[i]
    end
    return s
end

function LA.dot(x::Vector{T}, y::Vector{AD{T}}) where {T<:Number}
    return LA.dot(y, x)
end

function Base.:(*)(A::Matrix{AD{T}}, v::Vector{AD{T}}) where {T<:Number}
    n, m = size(A)
    if m != length(v)
        error("*: Matrix A and vector v have incompatible sizes.")
    end

    res = Vector{AD{T}}(undef, n)

    @inbounds for i in 1:n
        s = zero(AD{T})
        @simd for j in 1:m
            s += A[i, j] * v[j]
        end
        res[i] = s
    end

    return res
end

function Base.:(*)(A::Matrix{AD{T}}, v::Vector{T}) where {T<:Number}
    n, m = size(A)
    if m != length(v)
        error("*: Matrix A and vector v have incompatible sizes.")
    end

    res = Vector{AD{T}}(undef, n)

    @inbounds for i in 1:n
        s = zero(AD{T})
        @simd for j in 1:m
            s += A[i, j] * v[j]
        end
        res[i] = s
    end

    return res
end

end # DNNS module

