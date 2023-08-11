module Boolean

global vars       = nothing
global logic_size = nothing
global opMap = Dict( :* => :.&, :+ => :.|, :⊕ => :.⊻, :~ => :.~)

import Base
export Blogic, logicCount, nonZero, get_non_zero_inputs, bool_var_rep
export init_logic, modifyLogicExpr!, simplifyLogic, create_bool_rep
export isEquiv


"""
Define an operations type. Meant for the operators :
`+`, `:*`, `:⊕ `, `:~` so that we can dispatch on them as types.
"""
struct Op{T} end

"""
Structure used to represent a boolean formula involving variables 
        given by a single base string followed by a number.

In practice, one uses a higher level constructor (`create_bool_rep`) 
rather than the one provided here.
## Fields
- `formula :: String` -- The string representation of the formula.
- `var :: String` -- The base name of the logical variables.
- `size :: Int64` -- The number of variables in the formula.
- `val :: BitVector` -- The bit vector representing the formula. 
                      It essentially expresses the values of all possible inputs.  
## Constructors
`Blogic(form::String, v::String, value::BitVector)`

## Examples
- `Blogic("(z1 + z2) * z3", BitVector(Bool[0, 0, 0, 0, 0, 1, 1, 1]))` 

This is the logic (boolean) formula that ORs `z1` and `z2`, 
        then ANDs that with `z3`.
"""
struct Blogic
    formula::String
    var    ::String
    size   ::Int64
    val    ::BitVector
    Blogic(form::String, v::String, value::BitVector) = 
    new(form, v, map(x -> Int64(log2(x)), (size(value))[1]), value)
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
    n, _ = size(z)
    if n == 0
        println("N/A")
    else
        for i in 1:n
            println(io, Tuple(map(x -> Int64(x), z[i, :])))
        end
    end
end


Base.isless(x::Int64 , y::Symbol) = true
Base.isless(x::Symbol, y::Int64 ) = false
Base.isless(x::Symbol, y::Expr  ) = true
Base.isless(x::Expr  , y::Symbol) = false

function Base.:(==)(b1::Blogic, b2::Blogic) 
    (b1.formula == b2.formula) &&
    (b1.var     == b2.var    ) &&    
    (b1.size    == b2.size   ) &&    
    (b1.val     == b2.val    ) 
end

"""
    logicCount(l)

Count the number of true values possible in a given formula.

## Arguments 
- l :: Blogic -- A logic formula

## Return
The number of true values that are possible with this formula.

"""
logicCount(l::Blogic) = count(l.val)


"""
    nonZero(l[,head])

Get up to `head` inputs that generate true values for a logic function, `l`.

## Arguments
- `l :: Blogic`   -- A logic formula.
- `head :: Int64` --  The maximum number of inputs to consider.
    
## Return
A list of up to `head` input values that will give the 
logic function, `l`, a value of `true`.
"""
function nonZero(l::Blogic; head=1)
    n = logicCount(l)
    get_non_zero_inputs(l.val, l.size[1], num=min(n,head))
end


"""
    get_non_zero_inputs

Get `num` inputs that generate true values for a logic function.
`v` is a boolean vector that indicates which elements of the truth table
yield a value of `true`.

## Arguments
- `v ::BitVector` -- A bit vector representing `true` and `false` values.
- `n :: Int64`  -- Describes the length of the truth table column: ``2^n``.
- `num :: Int64` -- The desired number of inputs that generate truth values.

## Returns
`::BitMatrix` -- Input values that generate truth values for the current function.
"""
function get_non_zero_inputs(v::BitVector, n::Int64; num::Int64=1)
    idx = collect(1:2^n)[v]
    return(vars[idx[1:num], :])
end


"""
    bool_var_rep(n)

Generate the boolean bit vectors necessary to represent a logic 
formula of `n` variables. Essentially, generate the truth table 
of each of the variables collectively as a `BitArray`.

## Arguments
- `n` : Number of logical variables.

## Return
`::BitArray` -- The bit representation of all of the logical variables.
"""
function bool_var_rep(n::Int64)
    if n > 30
        error("Can't represent more than 30 variables.")
    elseif n < 2
        error("Can't represent more than 2 variables.")
    else
        let nn::UInt64 = UInt64(n)
            # `BitArray([div(i-1, 2^(j-1)) % 2 != 0  for i in 1:2^n, j in 1:n])`
            # This is a bit matrix of shape (2^n, n), where column 1 
            # represents `x1`, column 2 represents `x2`, etc.
            BitArray([((i-1) >> (j-1)) & 1  for i in 1:2^nn, j in 1:nn])
        end
    end
end

"""
    init_logic

This sets two global variables, the size of the boolean vectors and 
the other the `Bitarray` representations of the variables.

## Arguments
- `n :: Int64` -- The number of boolean variables used in the formulas
this module will consider.

## Return
Nothing

"""
function init_logic(n::Int64)
    global vars = bool_var_rep(n)
    global logic_size = n
end


"""
    rle(xs)

Performs a R(un) L(ength) E(ncoding) on an array, 
grouping like values into arrays.
The values are **assumed** to be sorted.
    
## Arguments
- `xs :: Vector{T}` -- An array that is sortable.

## Return
`::Vector{Tuple{T, Int64}}` -- A Vector of pairs of the form: `(T, Int64)`
representing values from `xs` and the number of
their occurrences.

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
    modifyLogicExpr!(e)

The default rule for modifying a logic expression is to do nothing.
"""
function modifyLogicExpr!(e::T) where {T}
    return(e)
end

"""
    modifyLogicExpr!(e::Expr)

Walk an expression tree, converting variable names and operators
to Julia operators and variables into `BitVector` representations.

## Arguments
- `e :: Expr` -- An expression.

## Return
A logic expression.
"""
function modifyLogicExpr!(e::Expr)
    ary = []
    for (_, arg) in enumerate(e.args)
        push!(ary, modifyLogicExpr!(arg))
    end
    e.args = ary
    return(e)
end


"""
    modifyLogicExpr!(e::Symbol)

If `e` is a Symbol, it should be a variable of the form `r"[a-zA-Z]+[0-9]+"`.
The code splits the name off and uses the number to look up the 
    `BitVector` representation.
    Otherwise, it is assumed to be an operator symbol and it is then 
    mapped to the appropriate Julia operator.

  **NOTE:** This will work even if one makes a mistake and uses 
            `x3`, or `y3`, the bit vector for the 
            third "variable" will be used.

## Arguments
- `e :: Symbol` -- An variable or operator.

## Return
A logic expression.
"""
function modifyLogicExpr!(e::Symbol)
    global vars
    global opMap
    
    ## If this is a variable get the corresponding `BitVector`.
    if match(r"[a-zA-Z]+", String(e)) !== nothing
        vn = parse(Int64, (split(String(e), r"[a-zA-Z]+"))[2])
        return(vars[:, vn])
    end

    ## If this is an operator symbol, get the corresponding Julia operator.
    return(get(opMap, e, e))
end



"""
    redux(::Op{T}, Tuple{S, Int64})

Reduce a pair consisting of an expression and its count to just 
an expression. The default case is to just return the expression.

## Arguments
- `::Op{T}` -- An operator type.
- `pair :: Tuple{Expr, Int64}` -- Expression and its count.

## Return
``::Expr`` -- Expression.

"""
function redux(::Op{T}, pair::Tuple{S, Int64}) where {S, T}
    return(pair[1])
end

"""
    redux(::Opt{:⊕}, pair::Tuple{Expr, Int64})

Reduce a pair consisting of an expression and its count to just 
an expression. For an XOR expression, we know that only the expression 
remains or the value is 0.
"""
function redux(::Op{:⊕}, pair::Tuple{Expr, Int64})
    if pair[2] % 2 == 0
        return(0)
    else
        return(pair[1])
    end
end

"""
    simplifyLogic(e)

Simplify a logical expression.
This function calls a number of specialized variations of this function 
to deal with different logical operators.

## Arguments
- `e :: Expr` -- Logic expression.

## Return
`::Expr` -- Simplified logic expression.

"""
function simplifyLogic(e::Expr)
    if length(e.args) >= 3
        op = e.args[1]
        return(simplifyLogic(Op{op}(), e.args[2:end]))
    end
    ## If this has the form: `~ expr...`
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
    simplifyLogic(::Op{:~}, xargs::Any)

`simplifyLogic` for the NOT operator.
"""
function simplifyLogic(::Op{:~}, xargs::Any)
    xargs = map(arg -> simplifyLogic(arg), xargs)
    xargs = map(x -> redux(Op{:~}(), x), rle(sort(xargs)))
    
    if xargs == 1
        return(0)
    end
    if xargs == 0
        return(1)
    end
    return(Expr(:call, :~, xargs))
end


"""
    simplifyLogic(::Op{:+}, xargs::Vector{Any})

`simplifLogic` for the OR operator.
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
    simplifyLogic(::Op{:*}, xargs::Vector{Any})

`simplifyLogic` for the AND operator.
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
    simplifyLogic(::Op{:⊕}, xargs::Vector{Any})

`simplifyLogic` for the XOR operator.
"""
function simplifyLogic(::Op{:⊕}, xargs::Vector{Any})
    xargs = map(arg -> simplifyLogic(arg), xargs)
    xargs = map(x -> redux(Op{:⊕}(), x), rle(sort(xargs)))
    
    iargs = filter(arg -> typeof(arg) == Int64, xargs)
    xargs = filter(arg -> typeof(arg) != Int64, xargs)
    
    ## If there are no simple booleans (0 or 1s), return the xor expression 
    ##      with the xargs.
    if length(iargs) == 0
        return(Expr(:call, :⊕, xargs...))
    end
    
    ## If there are no complex boolean expressions, return the xor 
    ##      value of the simple booleans.
    if length(xargs) == 0
        return(sum(iargs) % 2)
        ## else if there is one complex boolean expression, return the 
        ## expression that is the xor of the resulting simple boolean XORS 
        ## with the complex boolean expression.
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
    simplifyLogic(e::Union{Int64, Symbol})

`simplifyLogic` for the irreducible cases: A number or a symbol.
"""
function simplifyLogic(e::Union{Int64, Symbol})
    return e
end



"""
    create_bool_rep(s[, simplify])

Turn boolean formula into a `BitVector` representation, `Blogic`.
- Determine the underlying base variable used in the formula.
- Parse the formula into an expression, `Expr`.
- Optionally simplify the logical expression.
- Walk the expression tree creating a new tree with Julia 
    mathematical operators substituted for user operators.
- Evaluate the expression to create a `BitVector`.

## Arguments 
- `s :: String`      -- A logical string.
- `simplify :: Bool` -- If `true` simplify the logical expression before 
                        creating the `BitVector`.
## Examples
- `create_bool_rep("(z1 + z2) * z3")`

## Return
`BitVector` -- representing the logical expression.
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
        error("Logic string uses more than one variable: ", 
                map(x -> String(x), ar))
    end
    if simplify
        val = eval(modifyLogicExpr!(simplifyLogic(Meta.parse(s))))
    else
        val = eval(modifyLogicExpr!(Meta.parse(s)))
    end
    Blogic(s, String(ar[1]), val)
end

"""
    isEquiv(f1::String, f2::String)

Determines if two logical functions are equivalent.

## Arguments
- `f1 :: String` -- Formula 1.
- `f2 :: String` -- Formula 2.

## Return
`Bool` -- `true` if the formulas are equivalent; `false` otherwise.

"""
function isEquiv(f1::String, f2::String)
    b = create_bool_rep( "( " * f1 * ") ⊕ " * " ( " * f2 * " )" )
    lc = logicCount(b)
    return( lc == 0 ? true : false )
end

    
end # module Boolean

