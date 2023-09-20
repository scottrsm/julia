# Boolean.jl Documentation

```@meta
CurrentModule = Boolean
```
# Overview
This module contains functions to compare Boolean functions.
It does this by using a bit-vector representation and 
comparing bits. The drawback with this representation is that it 
grows exponentially with the number of variables in a boolean expression.

There is an associated Jupyter notebook at src/BoolTest.ipynb.

## Types

```@docs
Op
```

```@docs
Blogic
```

## Functions

```@docs
logicCount
```

```@docs
nonZero
```

```@docs
get_non_zero_inputs
```

```@docs
bool_var_rep
```

```@docs
init_logic
```

```@docs
modifyLogicExpr!
```

```@docs
simplifyLogic
```

```@docs
create_bool_rep
```

```@docs
rle
```

```@docs
redux
```

```@docs
isEquiv
```

## Index

```@index
```

