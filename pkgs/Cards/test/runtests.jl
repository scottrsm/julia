using Test
using Random
using Cards

@testset "Test Module \"Cards\" Fidelity" begin
    @test length(detect_ambiguities(Cards)) == 0
end

@testset "Cards Test" begin
    Random.seed!(1)
    d = Deck()
    shuffle_deck!(d)

    ## Manually construct 6 hands...
    h1 = PokerHand([Card(Four, ♥ ), Card(Jack, ♦ ), Card(Six, ♣ ), Card(King, ♦ ), Card(Six, ♠ )])
    h2 = PokerHand([Card(Two, ♠ ), Card(Jack, ♥ ), Card(Ten, ♠ ), Card(Eight, ♦ ), Card(Ten, ♦ )])
    h3 = PokerHand([Card(Ace, ♣ ), Card(King, ♠ ), Card(Queen, ♦ ), Card(Jack, ♥ ), Card(Ten, ♣ )])
    h4 = PokerHand([Card(Ace, ♣ ), Card(King, ♣ ), Card(Queen, ♣ ), Card(Jack, ♣ ), Card(Ten, ♣ )])
    h5 = PokerHand([Card(Ace, ♦ ), Card(King, ♦ ), Card(Queen, ♦ ), Card(Jack, ♦ ), Card(Ten, ♦ )])
    h6 = PokerHand([Card(King, ♦ ), Card(Queen, ♦ ), Card(Jack, ♦ ), Card(Ten, ♦ ), Card(Nine, ♦ )])
    

    dh1 = deal_hand!(d)
    restore_deck!(d)
    dh2 = deal_hand!(d)

    @test h1 == h1
    @test h1 != h2
    @test h1 < h2
    @test h2 < h3
    @test h3 < h4
    @test h5 < h4
    @test h6 < h5

    @test dh1 == dh2

end
