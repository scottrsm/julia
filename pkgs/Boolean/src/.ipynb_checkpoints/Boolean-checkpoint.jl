module Boolean

global vars       = nothing
global logic_size = nothing
global opMap = Dict( :* => :.&, :+ => :.|, :⊕ => :.⊻, :~ => :.~)

import Base
export Blogic, logicCount, nonZero, count_non_zero, bool_var_rep, init_logic, modifyLogicExpr!, simplifyLogic, create_bool_rep


## Define an operations type. Meant for the operators :+, :*, :⊕ so that we can dispatch on them as types.
struct Op{T} end

"""
    Structure used to represent a boolean formula involving variables given by a single base string followed by a number.
      Example: (z1 + z2) * z3. Is the boolean formula that takes z1 and z2 and ORs then and then ANDs that with z3.
    The field, val, is a bit vector representing the formula. It essentially expresses the values of all possible inputs.  
"""
struct Blogic
    formula::String
    var    ::String
    size   ::Tuple{Int64}
    val    ::BitVector
    Blogic(form::String, v::String, sz::Int64, value::BitVector) = new(form, v, map(x -> Int64(log2(x)), size(value)), value)
end

"""
    Show the Blogic structure.
    Params:
    io: IO handle.
    x : Blogic structure.
"""
function Base.show(io::IO, x::Blogic)
    println(io, "Formula    = ", x.formula)
    println(io, "Variable   = ", x.var    )
    println(io, "Size       = ", x.size   )
    println(io, "Bit vector = ", x.val    )
end

"""
    Show a BitMatrix.
"""
function Base.show(io::IO, z::BitMatrix)
    n, m = size(z)
    if n == 0
        println("N/A")
    else
        for i in 1:n
            println(io, Tuple(map(x -> Int64(x), z[i, :])))
        end
    end
end


function Base.isless(x::Int64, y::Symbol)
    true
end

function Base.isless(x::Symbol, y::Int64)
    false
end

function Base.isless(x::Symbol, y::Expr)
    true
end

function Base.isless(x::Expr, y::Symbol)
    false
end

"""
    Count the number of true values possible in a given formula.
"""
logicCount(l::Blogic) = count(l.val)


"""
    Get upto `head` inputs that generate true values for a logic function, l.
    `l` A Blogic structure.
    `head` The maximum number of inputs to consider.
    
    return: A list of up to `head` input values that will give the logic function, `l`, a value of `true`
"""
function nonZero(l::Blogic; head=1)
    n = logicCount(l)
    count_non_zero(l.val, l.size[1], num=min(n,head))
end

"""
    Get `num` inputs that generate true values for a logic function.
    `v` is a boolean vector that indicates which elements of the truth table
     yield a value of `true`.
"""
function count_non_zero(v, n; num=1)
    idx = collect(1:2^n)[v]
    return(vars[idx[1:num], :])
end


"""
    Generate the boolean bit vectors necessary to represent a logic formula of `n` variables.
"""
function bool_var_rep(n::Int64)
    if n > 30
        error("Can't represent more than 30 variables.")
    elseif n < 2
        error("Can't represent more than 2 variables.")
    else
        let nn::UInt64 = UInt64(n)
            # BitArray([div(i-1, 2^(j-1)) % 2 != 0  for i in 1:2^n, j in 1:n])
            # This is a bit matrix of shape (2^n, n), where column 1 represents x1, column 2 represents x2, ... up to xn.
            BitArray([((i-1) >> (j-1)) & 1  for i in 1:2^nn, j in 1:nn])
        end
    end
end

"""
    This sets two global variables, the size of the boolean vectors and the other 
    the Bitarray represenations of the variables.
"""
function init_logic(n::Int64)
    global vars = bool_var_rep(n)
    global logic_size = n
end

"""
    Walk an expression tree, converting variable names and operators
    to Julia operators and variables with Bitvector representations.
"""
function modifyLogicExpr!(e::Expr)
    ary = []
    for (i, arg) in enumerate(e.args)
        push!(ary, modifyLogicExpr!(arg))
    end
    e.args = ary
    return(e)
end

"""
    The default rule for modifying a logic expression is to do nothing.
"""
function modifyLogicExpr!(e::T) where {T}
    return(e)
end

"""
    If e is a Symbol, it should be a variable of the form r"[a-zA-Z]+[0-9]+".
    The code splits the name off and uses the number to look up the bitvector representation.
    Otherwise, it is assumed to be an operator symbol and it is then mapped to the appropriate 
    Julia operator.
    NOTE: This will work even if one makes a mistake and uses x3, or y3, the bit vector for the 
          third "variable" will be used.
"""
function modifyLogicExpr!(e::Symbol)
    global vars
    global opMap
    
    ## If this is a variable get the corresponding Bitvector.
    if match(r"[a-zA-Z]+", String(e)) != nothing
        vn = parse(Int64, (split(String(e), r"[a-zA-Z]+"))[2])
        return(vars[:, vn])
    end

    ## If this is an operator symbol, get the corresponding Julia operator.
    return(get(opMap, e, e))
end


"""
    Performs an RLE on an array, grouping like values into arrays.
    Assumes that <xs> is sorted.
    Params:
    xs: An array of Any

    Return: A vector of pairs (x, count), each item counting the number of times a given value occurred in xs.
"""
function rle(xs::Vector{T}) where T
    lastx = xs[1]
    rle = []
    cnt = 1
    for x in xs[2:end]
        if x == lastx
            cnt += 1
        else
            push!(rle, (lastx, cnt))
            lastx = x
            cnt = 1 
        end
    end
    push!(rle, (lastx, cnt))
    return(rle)
end

"""
    Reduce a pair consisting of an expression and its count to just an expression.
    The default case is to just return the expression.
"""
function redux(::Op{T}, pair) where T
    return(pair[1])
end

"""
    Reduce a pair consisting of an expression and its count to just an expression.
    For an XOR expression, we know that only the expression survies or the value is 0.
"""
function redux(::Op{:⊕}, pair)
    if pair[2] % 2 == 0
        return(0)
    else
        return(pair[1])
    end
end


"""
    Operator is NOT.
"""
function simplifyLogic(::Op{:~}, xarg::Any)
    xargs = map(arg -> simplifyLogic(arg), xargs)
    xargs = map(x -> redux(Op{:~}(), x), rle(sort(xargs)))
    
    if xarg == 1
        return(0)
    end
    if xargs == 0
        return(1)
    end
    return(Expr(:call, :~, xarg))
end


"""
    Operator is OR.
"""
function simplifyLogic(::Op{:+}, xargs::Vector{Any})
    xargs = map(arg -> simplifyLogic(arg), xargs)
    xargs = map(x -> redux(Op{:+}(), x), rle(sort(xargs)))
    
    if any(x -> x == 1, xargs)
        return(1)
    end
    xargs = filter(x -> x != 0, xargs)
    if length(xargs) == 0
        return(0)
    elseif length(xargs) == 1
        if xargs[1] isa Vector{Any}
            return(Expr(xargs[1]...))
        else
            return(xargs[1])
        end
    else
        return(Expr(:call, :+, xargs...))
    end
end

"""
    Operator is AND.
"""
function simplifyLogic(::Op{:*}, xargs::Vector{Any})
    xargs = map(arg -> simplifyLogic(arg), xargs)
    xargs = map(x -> redux(Op{:*}(), x), rle(sort(xargs)))
    
    if any(x -> x == 0, xargs)
        return(0)
    end
    xargs = filter(x -> x != 1, xargs)
    if length(xargs) == 0
        return(1)
    elseif length(xargs) == 1
        if xargs[1] isa Vector{Any}
            return(Expr(xargs[1]...))
        else
            return(xargs[1])
        end
    else
        return(Expr(:call, :*, xargs...))
    end
end

"""
    Operator is XOR.
"""
function simplifyLogic(::Op{:⊕}, xargs::Vector{Any})
    xargs = map(arg -> simplifyLogic(arg), xargs)
    xargs = map(x -> redux(Op{:⊕}(), x), rle(sort(xargs)))
    
    iargs = filter(arg -> typeof(arg) == Int64, xargs)
    xargs = filter(arg -> typeof(arg) != Int64, xargs)
    
    ## If there are no simple booleans (0 or 1s), return the xor expression with the xargs.
    if length(iargs) == 0
        return(Expr(:call, :⊕, xargs...))
    end
    
    ## If there are no complex boolean expressions, return the xor value of the simple booleans.
    if length(xargs) == 0
        return(sum(iargs) % 2)
        ## else if there is one complex boolean expression, return the expression that is 
        ## the xor of the resulting simple boolean xors with the complex boolean expression.
    elseif length(xargs) == 1
        if (sum(iargs) % 2) == 1
            return(Expr(:call, :~, xargs[1]))
        else
            if xargs[1] isa Vector{Any}
                return(Expr(xargs[1]...))
            else
                return(xargs[1])
            end
        end
    end
    
    ## Otherwise, there is a simple component, find its xor value 
    ## and then return an expression of the xor with the complex expressions.
    if (sum(iargs) % 2) == 1
        return(Expr(:call, :~, Expr(:call, :⊕, xargs...)))
    else
        return(Expr(:call, :⊕, xargs...))
    end
    println("Not here I hope")
end

"""
    The default rule for simplyfying a logic expression is to do nothing.
"""
function simplifyLogic(e::Union{Int64, Symbol})
    return e
end


"""
    Arbitrary expression (parsed) logic expression.
"""
function simplifyLogic(e::Expr)
    if length(e.args) >= 3
        op = e.args[1]
        return(simplifyLogic(Op{op}(), e.args[2:end]))
    end
    ## If this has the form: ~ expr...
    if length(e.args) == 2 && e.args[1] == :~
        arg = simplifyLogic(e.args[2])
        if typeof(arg) == Int64
            return((1 + arg) % 2)
        else
            return(Expr(:call, :~, arg))
        end
    end
    
    return(e)
end

"""
    Turn boolean formula into a bitvector representation, Blogic.
"""
function create_bool_rep(s::String, simplify=false)
    global logic_size
    
    ## Check that the variables used have the same name:
    ## Looking for x1, x2, x3. Not x1, y2, z3.
    ## Get the array of unique variables names.
    ar = []
    for m in eachmatch(r"[a-zA-Z]+([0-9]+)", s)
        push!(ar, split(m.match, r"[0-9]+")[1])
    end
    
    ## If there are more than one, error.
    ar = unique(ar)
    if length(ar) > 1
        error("Logic string uses more than one variable: ", map(x -> String(x), ar))
    end
    if simplify
        val = eval(modifyLogicExpr!(simplifyLogic(Meta.parse(s))))
    else
        val = eval(modifyLogicExpr!(Meta.parse(s)))
    end
    Blogic(s, String(ar[1]), logic_size, val)
end


end # module
