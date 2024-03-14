# DNNS.jl Documentation

```@meta
CurrentModule = DNNS
```

# Overview
Below is a collection of structures and functions which are useful to create (D)epp (N)eural (N)etworks.
One of the features of this package is the ability to do *Automatic Differentiation*.
This is accomplished via the structure, `AD`. The basic arithematic functions as well
as the power, log, exponential, and all of the trigonometric functions from the base 
package are overloaded by this package to work with `AD` "numbers".

Additionaly, this package adds a notion of a piece-wise linear function via the struct, `PWL`.
This structure, as well as are the structures `DLayer` and `DNN`, has been instrumented so 
that it can also serve as a function.

- Structures:
    - `PWL:` A structure representing a piece-wise linear function.
    - `AD:` A structure representing the value and its associated derivative. 
            Such a structure is used to compute "Automatic Differentiation".
    - `DLayer:` A structure representing a single layer of a DNN.
    - `DNN:` A structure representing a Deep Neural Net.
- Functions
    - Non-linear Activation Functions
        - `sigmoid1`
        - `sigmoid2`
        - `sigmoid3`
        - `relu`
        - `relur`
    - Associated Linearing Functions
        - `loss`
        - `fit`

## Data Structures


```@docs
PWL
```

```@docs
AD
```

```@docs
DLayer
```

```@docs
DNN
```

## Non-linear Activation Functions

```@docs
sigmoid1
```

```@docs
sigmoid2
```

```@docs
sigmoid3
```

```@docs
relu
```

```@docs
relur
```

## Functions

```@docs
loss
```

```@docs
fit
```


## Index

```@index
```

