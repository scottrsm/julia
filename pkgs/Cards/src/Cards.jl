"""
    Module Cards:
    Functions to define cards, a card deck, shuffle the deck, deal from 
    the deck, poker card comparison, and poker hand representations.
    Can reset the deck, pulling in all cards from a previous game.
    Additionally, can play a round of vanilla poker with two players.
"""
module Cards

export PHrep, SPHrep, Card, Deck, show, deal!, reset!, pokerHand
export pokerHandCmp, pokerTradeInCards!, playPoker2!
export pokerHandGroupLength, pokerHandSubRep
export get_card_rank_and_top_suit, getOrderedCardsFromRep
export ♣, ♠, ♦, ♥
export Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Jack, Queen, King, Ace
export Singles, Pair, TwoPair, ThreeOfAKind, FullHouse, FourOfAKind, Flush, Straight, StraightFlush

import Random
import LinearAlgebra
import Printf


## Enums for card ranking and for Suits. The ranking is based on poker.
@enum Suit ♠ ♦ ♣ ♥
@enum Rank Two Three Four Five Six Seven Eight Nine Ten Jack Queen King Ace

## Enum for the ranking of poker hands.
@enum PokerType Singles Pair TwoPair ThreeOfAKind FullHouse FourOfAKind Flush Straight StraightFlush

const SUITSIZE = 4
const RANKSIZE = 13
const DECKSIZE = RANKSIZE * SUITSIZE
const DIFFMAT  = LinearAlgebra.Bidiagonal([1,1,1,1,1], [-1, -1, -1, -1], :U)


### -------------------------------------------------------------
### ---- User Defined Structures  -------------------------------
### -------------------------------------------------------------

"""
    DATA STRUCTURE: Representation of a Card.
"""
struct Card
    suit::Suit
    rank::Rank
    Card(s::Suit, r::Rank) = new(s, r)
end


"""
    DATA STRUCTURE: Representation of a Desk of Cards.
"""
mutable struct Deck
    cards::Vector{Card}
    left::UInt8
    function Deck()
        new(reshape([Card(Suit(suit), Rank(num)) for num in 0:(RANKSIZE-1), suit in 0:(SUITSIZE-1)], (DECKSIZE,)), UInt8(1))
    end
end

"""
   Comparison function used for slot.
   Assumes that the vector of suits are sorted from highest to lowest.
"""
function isPSRLess(cr1::Tuple{Rank, Vector{Suit}}, cr2::Tuple{Rank, Vector{Suit}})
    if length(cr1[2]) < length(cr2[2])      # Length of the vector1 is less than vector2 (second hand-piece has more-of-a-kind)
        return(true)
    elseif length(cr1[2]) > length(cr2[2])  # Or not...
        return(false)
    # Otherwise, hand-piece has same number of-a-kind...
    elseif cr1[1] < cr2[1]                  # Check that rank1 < rank2
        return(true)
    elseif cr1[1] > cr2[1]                  # Or, not...
        return(false)                
    # Now, h1 piece has the rank and same number of cards with this rank with h2 piece.
    elseif cr1[2][1] < cr2[2][1]            # Is the largest suit (vector is ordered) hand-piece1 less than the largest suit of hand piece2 
        return(true)
    elseif cr1[2][1] > cr2[2][1]            # Or not...
        return(false)
    else
        return(false)                       # Default will be to say first piece is not less than second piece -- should not happen.
    end
end

"""
    DATA STRUCTURE: Representation of a poker hand. 
        A rep of the form: [(rank, [Suits])].
        These are the "ordered Card Pairings".
        This is a natural way that a player would organize their hand: 
            grouping by card ranking.
        This structure will be used to place a hand in a canonical way 
            by ordering this structure in the following way: 
            First by pairings (rank, [suits]) will be ranked by rank. 
        Within the pair the suits will be ranked in natural order.
    - Step 0: First sort second arg.
    - Step 1: Use the comparison function above to do the sortgin.
"""
struct SPHrep
    orderedCP::Vector{Tuple{Rank, Vector{Suit}}}
    function SPHrep(ocp::Vector{Tuple{Rank, Vector{Suit}}})
        x = copy(ocp)
        map(z -> sort!(z[2], rev=true), x)
        sort!(x, lt = isPSRLess, rev=true)
        new(x)
    end
end

"""
    DATA STRUCTURE: Representation of a poker hand. 
        A rep of the form: (PokerType, SPHrep)
        The SPHrep is a poker hand in canonical order -- see above.
        This structure will allow us to compare poker hands.
"""
struct PHrep
    pt::PokerType
    subr::SPHrep
    function PHrep(pt::PokerType, sr::SPHrep)
        new(pt, sr)
    end
end


### ----------- AUGMENT JULIA METHODS TO WORK WITH OUR DATA STRUCTURES ------------------------
## Define how to show a Card and a Deck.
Base.show(io::Base.IO, c::Card)         = print(io, c.suit, "/", c.rank)
Base.show(io::Base.IO, d::Deck)         = ([mod(i, 7) == 0 ? println(io, "") : print(io, card, " ") for (i, card) in enumerate(d.cards[d.left:end])]; println(""))
Base.show(io::Base.IO, h::Vector{Card}) = ([print(io, c, ", ") for c in h[1:end-1]]; print(io, h[end]);)
Base.show(io::Base.IO, h::Vector{Suit}) = ([print(io, s, ", ") for s in h[1:end-1]]; print(io, h[end]);)
Base.show(io::Base.IO, h::PHrep)        = print(io, "\n", h.pt, " => ", h.subr);
Base.show(io::Base.IO, h::SPHrep)       = ([Printf.@printf(io, "%s %-5s -- [%s]", "\n\t", r, cs) for (r, cs) in h.orderedCP[1:end-1]];
                                           Printf.@printf(io, "%s %-5s -- [%s]\n", "\n\t", h.orderedCP[end][1], h.orderedCP[end][2]));

## Define the length of a Deck.
Base.length(d::Deck) = length(d.cards) + 1 - d.left

## Define the length of a Poker Hand Sub-representation.
Base.length(phr::SPHrep) = length(phr.orderedCP)

## Define the length of a Poker Hand.
Base.length(phr::PHrep) = length(phr.subr)

## Define how to do a random shuffle of the Deck.
Random.shuffle!(d::Deck) = d.cards = d.cards[Random.shuffle(d.left:DECKSIZE)];


"""
    Deal a hand of `n` cards from a Deck.
    Note: Will mutate the deck, d.
        
    param d A deck of Cards.
    param n The number of Cards to deal.

    return A vector of `n` cards.
"""
function deal!(d::Deck, n::Int64)
    ## Check to make sure we have enough cards to deal out.
    if n > (DECKSIZE+1 - d.left)
        throw("deal!: Not enough cards to deal.")
    end

    ## Get the next cards from where we left off dealing.
    cards = d.cards[d.left:(d.left + n-1)]

    ## Change our internal marker.
    d.left = d.left + n

    ## Return the dealt cards.
    return(cards)
end


"""
    Set/reset the deck to be full.
    That is, gather any outstanding cards from any games and place 
        them back in the deck.

    param d A deck of Cards.

    return Nothing
"""
function reset!(d::Deck)
    d.left = 1
end



"""
    Gets a card ranking and its top suit for the nth element of a 
        poker hand representation.
    That is, take the nth grouping based on the 
        PHrep(ordered as described in the PHrep doc)
        and return the rank and the top suit in that grouping.

    param handRep A poker hand representation.
    param n The index into the handRep array.

    return The card ranking and top suit of the nth poker hand group.
"""
function get_card_rank_and_top_suit(handRep::PHrep, n::Int64)
    ocp = handRep.subr.orderedCP
    if length(ocp) < n
        throw("get_card_rank_and_top_suit: Element \"$n\", is longer than the length of the poker hand representation: $ocp")
    end
    return(ocp[n][1], ocp[n][2][1])
end




"""
    Takes a poker hand sorted by rank from highest to lowest and converts 
        it to a sub-representation, SPHrep.
    Currently, this has the form: [(card-rank, [suits])]
    Here, these paired elements are sorted from highest to lowest based 
        on the number of duplicate cards by rank.
    The "suits" array keeps track of the suits of the duplicate cards 
        and are ordered from highest to lowest.
    If a tie appears (2 pairs, singles) sort by the larger of the 
        card rankings of the paired elements.
    Example 1: The raw (sorted by rank) hand): 
        [ Card(Spade, 10), Card(Club, 10), Card(Diamond, 2), Card(Club, 2), Card(Heart, 2) ]
               becomes: [(2, [Heart, Club, Diamond]), (10, [Club, Spade])]
    Example 2: The raw (sorted by rank) hand): 
        [ Card(Spade, 10), Card(Club, 10), Card(Diamond, Ace), Card(Heart, Ace), Card(Club, Jack) ]
               becomes: [ (Ace, [Heart, Diamond]), (10, [Club, Spade]), (Jack, [Club]) ]
   
    ## Heading
    param vs A rank-sorted Card vector.

    return A SPHrep -- with ordering as described above.
"""
function pokerHandSubRep(vs::Vector{Card})

    ## Temp vars used below.
    local lrank::Rank
    local csuits::Vector{Suit}

    ## Create a structure of the form: [(rank, [suits])]
    ## NOTE: If we don't specify the type here, the constructor
    ##       SPHrep will not know what to do with a raw list.
    local hsRep::Vector{Tuple{Rank, Vector{Suit}}} = []

    ## Loop over the vector collecting run-lengths for the unique strings.
    ## Start by taking the first card.
    lrank  = vs[1].rank
    csuits = [vs[1].suit]

    ## Now group by card "runs" by card rank.
    for v in vs[2:end]

        ## Get the rank and suit of the current card.
        crank = v.rank
        csuit = v.suit
        
        ## If current and last rank differ, we have finished a "run".
        ## Push the last rank and corresponding suits to hsRep.
        ## Start the "run" over with the current card suit.
        if crank != lrank
            push!(hsRep, (lrank, csuits))
            csuits = [csuit]
        else  ## Otherwise, gather more suits for the current card "run".
            push!(csuits, csuit)
        end
        
        ## Update the last rank.
        lrank = crank
    end

    ## We processed all but the last sequence, do that now.
    push!(hsRep, (lrank, csuits))

    ## Give this raw list of pairs, [ (rank, [suits]) ] to the SPHrep constructor to 
    ## put it into canonical order.
    return(SPHrep(hsRep))
end

"""
    Get the length of the nth ordered group in a poker hand.
    Example: Given that we have a poker hand sub rep, hsRep, 
                representing a full house, 
             This function would return 3 for the call pokerHandGroupLength(hsRep, 1)
             and return 2 for the call pokerHandgroupLength(hsRep, 2).

    param hsRep A poker sub-hand representation.
    param n     The group to examine.

    return The number of Cards in the nth group of the sub-representation.
"""
function pokerHandGroupLength(hsRep::SPHrep, n::Integer)
    ocp = hsRep.orderedCP
    return(length(ocp[n][2]))
end

"""
    Takes a poker hand and converts it to a representation of the form:
    (hand-type, poker-hand-sub-rep)
    See the description of poker-hand-sub-rep in the function pokerHandSubRep.
   
    param v A vector of Cards.

    return A pairing of hand-type with a vector with a sub-representation, 
            currently of the form: (rank, [suits]).
           The sub-representation is ordered as described in the 
            function pokerHandSubRep.
           Specifically, the return has the form: 
            (hand-type, [(card-rank, [card-suits])])
"""
function pokerHand(v::Vector{Card})
    if length(v) != 5
        throw("pokerHand: Poker hand must have 5 cards.")
    end

    ## Sort the vector by card rank from highest to lowest.
    local vs = sort(v, by=x -> x.rank, rev=true)
    local flushFlag = false
    local hsrep

    ## Look at sequential differences based on rank -- if all ones 
    ##   then we have a straight or straight-flush.
    local rankDiff = DIFFMAT * map(x -> Int64(x.rank), vs)
    pop!(rankDiff)

    if length(unique(map(x -> x.suit, vs))) == 1
        flushFlag = true
    end

    
    ## Check for straights, flushes, and straight-flushes.
    if rankDiff == [1,1,1,1]
        hsrep = map(x -> (x.rank, [x.suit]), vs)
        if flushFlag
            return(PHrep(StraightFlush, SPHrep(hsrep)))
        else
            return(PHrep(Straight, SPHrep(hsrep)))
        end
    elseif flushFlag
        return(PHrep(Flush, SPHrep(hsrep)))
    end
    
    ## Now we have to check for: 
    ##  Singles, Pair, TwoPair, ThreeOfAKind, FullHouse, and FourOfAKind.
    ## To do this classification, we first create a 
    ##  [ (card-rank, [suits]) ] sub-representation for the hand.
    hsrep = pokerHandSubRep(vs)

    ## Classify the hand using this sub-representation
    ##  and return with the pokerRep -- a pairing of the poker hand-type 
    ##  with the sub rep.
    hsLen = length(hsrep)
    if hsLen == 5                               ## Singles -- 5 groups of 1                     Total:       = 5 groups.
        return(PHrep(Singles, hsrep))
    elseif hsLen == 4                           ## Pair    -- 1 group of 2 and 3 singles.       Total: 1 + 3 = 4 groups.
        return(PHrep(Pair, hsrep))
    elseif hsLen == 3                           ## Triple OR TwoPair 
        if pokerHandGroupLength(hsrep, 1) == 3  ##    Triple  -- 1 group  of 3 and 2 singles.   Total: 1 + 2 = 3 groups.
            return(PHrep(ThreeOfAKind, hsrep))  
        else
            return(PHrep(TwoPair, hsrep))       ##    TwoPair -- 2 groups of 2 and 1 single.    Total: 2 + 1 = 3 groups.
        end
    elseif hsLen == 2                           ## Four OR FullHouse
        if pokerHandGroupLength(hsrep, 1) == 4  ##  Four      -- 1 group of 4 and 1 single.     Total: 1 + 1 = 2 groups.
            return(PHrep(FourOfAKind, hsrep))   
        else
            return(PHrep(FullHouse, hsrep))     ##  FullHouse -- 1 group of 3 and 1 group of 2. Total: 1 + 1 = 2 groups.
        end
    end
end


function get_hand_piece(hrep, n)
    return(hrep[n])
end


"""
    Compare two poker hands. 
    First construct their poker-hand-representations: 
        (poker-hand-type, [(card-h1, [ordered-suits]), (card-h2, [ordered-suits])...]).
    Use this to determine if hr1 < hr2. Do this first by using the ranking 
        of the poker-hand-type (i.e., Pair, TwoPair, FullHouse, etc.)
    If two hands have the same type, then use ranking and their suits or 
        other means according to the rules of poker to make the determination.

    param hr1 PHrep.
    param hr2 PHrep.

    return Bool `true` if `hr1 < hr2`, otherwise `false`.
"""
function pokerHandCmp(hr1::PHrep, hr2::PHrep)

    handType1  = hr1.pt
    handType2 = hr2.pt

    ## Decide based on handType
    if handType1 < handType2
        return(true)
    end
    
    if handType1 > handType2
        return(false)
    end
    
    
    ## The hands are of the same type.
    r1, s1 = get_card_rank_and_top_suit(hr1, 1)
    r2, s2 = get_card_rank_and_top_suit(hr2, 1)
    
    if handType1 == FourOfAKind
        if r1 < r2
            return(true)
        elseif r1 > r2
            return(false)
        end
    end
    
    if handType1 in [ThreeOfAKind, FullHouse]
        if r1 < r2
          return(true)
        elseif r1 > r2
          return(false)
        elseif s1 < s2
          return(true)
        elseif s1 > s2
          return(false)
        end
    end    
    
    
    ## Now the hands are of the same type and the first group of each hand 
    ##      has the same ranking.
    ## This can only happen for: 
    ##      Singles, Pair, TwoPair, Flush, Straight, StraightFlush
    if handType1 in [Singles, Flush, Straight, StraightFlush]  # Compare the highest cards by suit.
        if r1 < r2
          return(true)
        elseif r1 > r2
          return(false)
        elseif s1 <= s2 
          return(true)
        else
          return(false)
         end
    elseif handType1 == Pair  # Compare the next highest card by card rank, if the same, compare by suit.
        if r1 < r2
            return(true)
        elseif r1 > r2
            return(false)
        else
            rr1, ss1 = get_card_rank_and_top_suit(hr1, 2)
            rr2, ss2 = get_card_rank_and_top_suit(hr2, 2)
            if rr1 < rr2
                return(true)
            elseif rr1 > rr2
                return(false)
            else
                if ss1 < ss2
                    return(true)
                elseif ss1 > ss2
                    return(false)
                end
            end
        end
    elseif handType1 == TwoPair # Compare the next highest pair by card rank, 
                                # if the same, compare by the next single card: first by rank and then by suit.
        rr1, ss1 = get_card_rank_and_top_suit(hr1, 2)
        rr2, ss2 = get_card_rank_and_top_suit(hr2, 2)
        if rr1 < rr2
            return(true)
        elseif rr1 > rr2
            return(false)
        else ## Look at the third card and compare by rank and then by suit.
            rrr1, sss1 = get_card_rank_and_top_suit(hr1, 3)
            rrr2, sss2 = get_card_rank_and_top_suit(hr2, 3)
            if rrr1 < rrr2
                return(true)
            elseif rrr1 > rrr2
                return(false)
            else
                return(sss1 < sss2)
            end
        end
    else
        throw("pokerHandCmp: The following handType should not occur at this stage: $handType1")
    end
    throw("pokerHandCmp: Should not have reached here, there is a gap in our processing. The handType is $handType1")
end


"""
    Get the rank/suit ordered list of cards from a poker hand representation.

    param phr::PHrep The poker hand representation of a poker hand.

    return A new poker hand of Cards in order.
"""
function getOrderedCardsFromRep(phr::PHrep)
    ocp = phr.subr.orderedCP
    return(vcat([Card(c, r) for (r, cs) in ocp for c in cs]...))
end


"""
    Given a poker hand, h, determine now many cards to replace, 0 to 2,
    then return the replacement. Return the a copy of the original hand if 
    no cards needed. If cards are replaced, this will mutate the deck of cards, d.

    param d A deck of cards
    param h A poker hand.

    return A pair of a new vector of Cards, and a pokerHand.
"""
function pokerTradeInCards!(d::Deck, h::Vector{Card})
    hr = pokerHand(h)
    hType = hr.pt
    
    local nh 
    local nhr

    ## Decide if we replace cards and how many.
    if hType in [Flush, Straight, StraightFlush, FullHouse] # Do nothing.
        nh  = deepcopy(h)
        nhr = deepcopy(hr)
    elseif hType in [Singles, Pair, ThreeOfAKind]           # Take two cards.
        cards = getOrderedCardsFromRep(hr)
        deleteat!(cards, [4,5])
        ncards = deal!(d, 2)
        append!(cards, ncards)
        nh  = deepcopy(cards)
        nhr = pokerHand(nh)
    elseif hType == TwoPair                                 # Take one card.
        cards = getOrderedCardsFromRep(hr)
        deleteat!(cards, [5])
        ncards = deal!(d, 1)
        append!(cards, ncards)
        nh  = deepcopy(cards)
        nhr = pokerHand(nh)
    elseif hType == FourOfAKind                             # Take one card if...
        cards = getOrderedCardsFromRep(hr)
        if Int64(cards[5].rank) < 8
            deleteat!(cards, [5])
            ncards = deal!(d, 1)
            append!(cards, ncards)
            nh  = deepcopy(cards)
            nhr = pokerHand(nh)
        else
            nh  = deepcopy(h)
            nhr = deepcopy(hr)
        end
    end
    
    return(nh, nhr)
end



"""
    Play a game of poker with two players.
    Process:
        1). Deal each player two 5 card hands.
        2). Compare two hands to see who would win.
        3). Each player can then ask for up to 2 cards.
        4). The two players are compared again to see who wins.

    Note: This function mutates the deck by dealing cards 
          to the players.

    param d A deck of cards.

    return Nothing
"""
function playPoker2!(d::Deck)
    ## Reset the deck -- put back all the cards from previous games.
    reset!(d)

    ## Shuffle the deck.
    Random.shuffle!(d)

    ## Deal poker hands for two players.
    h1 = deal!(d, 5)
    h2 = deal!(d, 5)
    println("hand 1 = $h1")
    println("hand 2 = $h2")

    ## Get the poker hand representations.
    hr1 = pokerHand(h1)
    hr2 = pokerHand(h2)
    println("")
    println("hand 1 rep = $hr1")
    println("hand 2 rep = $hr2")

    println("")

    ## Compare the hands.
    islessFlag = pokerHandCmp(hr1, hr2)
    println("BEFORE Players are given a chance at card replacement:")
    if islessFlag
        println("Player 2 wins!")
    else
        println("Player 1 wins!")
    end

    ## Allow each player a chance at replacing up to two cards.
    local nh1, nhr1 = pokerTradeInCards!(d, h1)
    local nh2, nhr2 = pokerTradeInCards!(d, h2)

    println("\nAFTER Players are given a chance at card replacement:")
    println("Updated hand 1 = $nh1")
    println("Updated hand 2 = $nh2")

    println("\nNew hand 1 rep = $nhr1")
    println("New hand 2 rep = $nhr2")
    
    println("")
    ## Now compare the new hands.
    islessFlag = pokerHandCmp(nhr1, nhr2)
    if islessFlag
        println("Player 2 wins!")
    else
        println("Player 1 wins!")
    end
    
    return(nothing)
end

end # Module

