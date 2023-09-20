# Cards.jl Documentation

```@meta
CurrentModule = Cards
```

# Overview
This module describes the **enums/structs/functions** to play a simple version of Poker.

**Enums:**
- `Suit` -- The 4 suits of the cards.
- `Rank` -- The 13 different ranks.
- `PokerType` -- The different poker hands one may have.

**Structs:**
- `Card` -- A description of a given card as a combination of `Suit` and `Rank`.
- `PokerHand` -- A data structure describing a poker hand including its classification;
               i.e., its `PokerType`.
- `Deck` -- A description of a deck of cards.

**Functions:**
- Deck Functions:
    - `Deck()` -- Create a deck of cards. 
    - `shuffle_deck!` -- Shuffle the deck of cards.
    - `restore_deck!` -- Return all "dealt" cards back to the place in the deck where
                         they were drawn.
    - `num_cards_left_in_deck` -- The number of cards left in the deck.
- Card Play
    - `draw_cards!` -- Draw some number of cards from a deck.
    - `deal_hand!`   -- Deal a poker hand to a player.
    - `make_secondary_draw!` -- Execute a second draw of a simple poker game.
    - `play_poker!` -- Simulate a simple version of a poker game with two players.

There is an associated Jupyter notebook at src/CardTest.ipynb.


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

