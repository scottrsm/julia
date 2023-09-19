# Cards.jl Documentation

```@meta
CurrentModule = Cards
```

## Enums
```@docs
Suit
```

```@docs
Rank
```

```@docs
PokerType
```

## Structs
```@docs
Card
```

```@docs
PokerHand
```

```@docs
Deck
```

## Functions

```@docs
shuffle_deck!
```

```@docs
restore_deck!
```

```@docs
num_cards_left_in_deck
```

```@docs
deal_hand!
```

```@docs
draw_cards!
```

```@docs
make_secondary_draw!
```

```@docs
play_poker!
```

## Lower Level Functions
The Base function `isless` has been overloaded for
the Structs: `Card`, and `PokerHand`.
- This means that a vector of `Card` is sorted first by `Suit` and then `Rank`.
- This means that a vector of `PokerHand` is sorted by the strength of the 
  hand as measured by the game Poker.

The Base function `(==)` has been overloaded for struct `PokerHand`.


```@docs
classify_hand
```

```@docs
grouped_rank_rep
```

```@docs
poker_hand_contract
```

## Index

```@index
```

