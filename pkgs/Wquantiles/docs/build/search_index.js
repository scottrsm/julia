var documenterSearchIndex = {"docs":
[{"location":"#Wquantiles.jl-Documentation","page":"Wquantiles.jl Documentation","title":"Wquantiles.jl Documentation","text":"","category":"section"},{"location":"","page":"Wquantiles.jl Documentation","title":"Wquantiles.jl Documentation","text":"CurrentModule = Wquantiles","category":"page"},{"location":"#Functions","page":"Wquantiles.jl Documentation","title":"Functions","text":"","category":"section"},{"location":"","page":"Wquantiles.jl Documentation","title":"Wquantiles.jl Documentation","text":"wquantile(::Vector{T}, ::Vector{S}, ::Vector{V}; ::Bool = true, ::Bool = true, ::Bool = true) where {T, S <: Real, V <: Real}","category":"page"},{"location":"#Wquantiles.wquantile-Union{Tuple{V}, Tuple{S}, Tuple{T}, Tuple{Vector{T}, Vector{S}, Vector{V}}} where {T, S<:Real, V<:Real}","page":"Wquantiles.jl Documentation","title":"Wquantiles.wquantile","text":"wquantile(x, w, q[; chk=true, norm_wgt=true, sort_q=true])\n\nFinds the q weighted quantile values from the vector x.\n\nType Constraints\n\nS <: Real\nV <: Real\n\nArguments\n\nx  ::Vector{T}: Vector(n) of values from which to find quantiles.\nw  ::Vector{S}: Vector(n) of weights to use.\nq  ::Vector{V}: Vector(l) of quantile values.\n\nKeyword Args\n\nchk     ::Bool: If true, check the input contract described below.\nnorm_wgt::Bool: If true, normalize the weights.\nNOTE: If norm_wgt is false, it is ASSUMED that w is already normalized.\nsort_q  ::Bool: If true, sort the quantile vector, q.\nNOTE: If sort_q is false, it ASSUMED that q is already sorted.\n\nInput Contract\n\nThe type of x implements sortable.\n|x|  = |w|     – Length of x matches length of weights.\n∀i,  w[i] >= 0    – The weights are non-negative.\nΣ w[i] >  0    – The sum of the weights is positive.\n∀i,  q[i] <= 1    – The quantile values are in 01.\n∀i,  q[i] >= 0\n\nReturn\n\nThe vector(l) of weighted quantile values from x. Letting qs be the sorted quantiles of q. The entry i is the i^rm th quantile (in qs) of x.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Wquantiles.jl Documentation","title":"Wquantiles.jl Documentation","text":"Wquantile","category":"page"},{"location":"#Wquantiles.Wquantile","page":"Wquantiles.jl Documentation","title":"Wquantiles.Wquantile","text":"Wquantile(X, w, q[; chk=true, norm_wgt=true, sort_q=true])\n\nFinds the q weighted quantile values from the columns of the matrix X.\n\nType Constraints\n\nS <: Real\nV <: Real\n\nArguments\n\nX  ::Matrix{T}: Matrix(n,m) of values from which to find quantiles.\nw  ::Vector{S}: Vector(n) of weights to use.\nq  ::Vector{V}: Vector(l) of quantile values.\n\nKeyword Args\n\nchk     ::Bool: If true, check the input contract described below.\nnorm_wgt::Bool: If true, normalize the weights.\nNOTE: If norm_wgt is false, it is ASSUMED that w is already normalized.\nsort_q  ::Bool: If true, sort the quantile vector, q.\nNOTE: If sort_q is false, it is ASSUMED that q is already sorted.\n\nInput Contract\n\nThe type of X implements sortable.\n∀i, |X[:, i]|   = |w| – Length of each column of X matches length of weights.\n∀i,      w[i]  >= 0   – Weights are non-negative.\nΣ w[i]  >  0          – The sum of the weights is positive.\n∀i,      q[i]  <= 1   – The quantiles values are in 01.\n∀i,      q[i]  >= 0\n\nReturn\n\nThe (l,m) matrix of weighted quantile values from X. Letting qs be the sorted quantiles of q. The entry (i,j) is the i^rm th quantile (in qs) from the j^rm th column of X.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Wquantiles.jl Documentation","title":"Wquantiles.jl Documentation","text":"WquantileM","category":"page"},{"location":"#Wquantiles.WquantileM","page":"Wquantiles.jl Documentation","title":"Wquantiles.WquantileM","text":"WquantileM(X, w, q[; chk=true])\n\nFinds the q weighted quantile values from the columns of the matrix X.\n\nType Constraints\n\nS <: Real\nV <: Real\n\nArguments\n\nX  ::Matrix{T}: Matrix(n,m) of values from which to find quantiles.\nw  ::Vector{S}: Vector(n) of weights to use.\nq  ::Vector{V}: Vector(l) of quantile values.\n\nKeyword Args\n\nchk::Bool     : If true, check the input contract described below.\n\nInput Contract\n\nThe type of X is sortable.\n∀i, |X[:, i]|  = |w| – Length of each column of X matches length of weights.\n∀i,      w[i] >= 0   – Weights are non-negative.\nΣ w[i] >  0          – The sum of the weights is positive.\n∀i, 0 <= q[i] <= 1   – The quantiles values are in 01.\n\nReturn\n\nThe (l,m) matrix of weighted quantile values from X. Letting qs be the sorted quantiles of q. The entry (i,j) is the i^rm th quantile (in qs) from the j^rm th column of X.\n\n\n\n\n\n","category":"function"},{"location":"#Index","page":"Wquantiles.jl Documentation","title":"Index","text":"","category":"section"},{"location":"","page":"Wquantiles.jl Documentation","title":"Wquantiles.jl Documentation","text":"","category":"page"}]
}
