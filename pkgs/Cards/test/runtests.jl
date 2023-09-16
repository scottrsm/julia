using Test
using Random
using Cards

@testset "Test Module \"Cards\" Fidelity" begin

@testset "Cards Test" begin
    Random.seed!(1)
    d = Deck()
    shuffle!(d)


    ## Deal two random hands.
    h1 = pokerHand([Card(♥, Four), Card(♦, Jack), Card(♣, Six), Card(♦, King), Card(♠, Six)])
    h2 = pokerHand([Card(♠, Two), Card(♥, Jack), Card(♠, Ten), Card(♦, Eight), Card(♦, Ten)])

    ## Manually construct 4 more hands...
    h3 = pokerHand([Card(♣, Ace), Card(♠, King), Card(♦, Queen), Card(♥, Jack), Card(♣, Ten)])
    h4 = pokerHand([Card(♣, Ace), Card(♣, King), Card(♣, Queen), Card(♣, Jack), Card(♣, Ten)])
    h5 = pokerHand([Card(♦, Ace), Card(♦, King), Card(♦, Queen), Card(♦, Jack), Card(♦, Ten)])
    h6 = pokerHand([Card(♦, King), Card(♦, Queen), Card(♦, Jack), Card(♦, Ten), Card(♦, Nine)])
    
    @testset "Testing poker hand comparisons of hands 0-6..." begin
        @test pokerHandCmp(h1, h2) == true
        @test pokerHandCmp(h1, h3) == true
        @test pokerHandCmp(h2, h4) == true
        @test pokerHandCmp(h4, h4) == true
        @test pokerHandCmp(h5, h5) == true
    end

    @test h1 == h1
    @test h1 != h2
    @test h1 == PHrep(Cards.Pair, SPHrep([(Six, [♣, ♠]), (King, [♦]), (Jack, [♦]), (Four, [♥])]))
    @test h2 == PHrep(Cards.Pair, SPHrep([(Ten, [♦, ♠]), (Jack, [♥]), (Eight, [♦]), (Two, [♠])]))
end
