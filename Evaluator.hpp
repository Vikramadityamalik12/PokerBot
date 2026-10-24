#pragma once
#include <bits/stdc++.h>

using Card = uint32_t;

struct HandEvaluation {
    uint16_t score; // 1 (Royal Flush) to 7462 (7-high) 
};

class HandEvaluator {
public:
    static void init();

    // Card factory
    static Card make_card(int rank, int suit); // rank: 0..12, suit: 0..3
    static Card from_string(const std::string& str); // e.g., "Ah", "Td", "2c"

    // 5-card and 7-card evaluators
    static uint16_t evaluate_5card(Card c1, Card c2, Card c3, Card c4, Card c5);
    static uint16_t evaluate_7card(const std::array<Card, 7>& cards);

    // Fast board pre-evaluation for Person 2 & Person 4
    static void precompute_board_ranks(
        const std::vector<Card>& board,
        const std::array<std::pair<Card, Card>, 1326>& all_hands,
        std::array<uint16_t, 1326>& out_ranks
    );

    // Showdown equity against range
    static float evaluate_showdown(
        int hero_hand_idx,
        const std::array<uint16_t, 1326>& board_ranks,
        const std::array<float, 1326>& villain_range
    );

private:
    static uint16_t flushes[8192];
    static uint16_t unique5[8192];
    static std::vector<std::pair<uint32_t, uint16_t>> prime_products; // sorted for binary search
    static bool initialized;
};