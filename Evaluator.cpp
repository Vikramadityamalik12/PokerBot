#pragma once
#include "Evaluator.hpp"
#include <bits/stdc++.h>

uint16_t HandEvaluator::flushes[8192];
uint16_t HandEvaluator::unique5[8192];
std::vector<std::pair<uint32_t, uint16_t>> HandEvaluator::prime_products;
bool HandEvaluator::initialized = false;

constexpr uint8_t PRIMES[13] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};
constexpr uint16_t SUIT_BITS[4] = {0x100, 0x200, 0x400, 0x800}; // S, H, D, C

constexpr uint8_t C7_5[21][5] = {
    {0,1,2,3,4}, {0,1,2,3,5}, {0,1,2,3,6}, {0,1,2,4,5}, {0,1,2,4,6},
    {0,1,2,5,6}, {0,1,3,4,5}, {0,1,3,4,6}, {0,1,3,5,6}, {0,1,4,5,6},
    {0,2,3,4,5}, {0,2,3,4,6}, {0,2,3,5,6}, {0,2,4,5,6}, {0,3,4,5,6},
    {1,2,3,4,5}, {1,2,3,4,6}, {1,2,3,5,6}, {1,2,4,5,6}, {1,3,4,5,6},
    {2,3,4,5,6}
};

Card HandEvaluator::make_card(int rank, int suit) {
    uint32_t prime = PRIMES[rank];
    uint32_t suit_mask = SUIT_BITS[suit];
    uint32_t rank_index = rank;
    uint32_t rank_mask = 1 << rank;
    return (rank_mask << 16) | (rank_index << 12) | suit_mask | prime; // Card Format
}

Card HandEvaluator::from_string(const std::string& str) {
    if (str.size() < 2) throw std::invalid_argument("Invalid format : " + str);
    char r = str[0];
    char s = str[1];
    int rank = 0;
    if (r >= '2' && r <= '9') rank = r - '2';
    else if (r == 'T' || r == 't') rank = 8;
    else if (r == 'J' || r == 'j') rank = 9;
    else if (r == 'Q' || r == 'q') rank = 10;
    else if (r == 'K' || r == 'k') rank = 11;
    else if (r == 'A' || r == 'a') rank = 12;
    else
    {
        throw std::invalid_argument("Invalid format : " + str);
    }
    int suit = 0;
    if (s == 's' || s == 'S') suit = 0;
    else if (s == 'h' || s == 'H') suit = 1;
    else if (s == 'd' || s == 'D') suit = 2;
    else if (s == 'c' || s == 'C') suit = 3;
    else
    {
        throw std::invalid_argument("Invalid format : " + str);
    }

    return make_card(rank, suit);
}

void HandEvaluator::init() {
    if (initialized) return;

    // 1. Generate Straight & High Card permutations (unique5)
    // There are 1287 unique combinations of 5 ranks
    std::vector<int> ranks_idx(13);
    std::iota(ranks_idx.begin(), ranks_idx.end(), 0);

    // Rank evaluation order generator:
    // Straight Flushes: 1..10
    // Four of a Kind: 11..166
    // Full House: 167..322
    // Flushes: 323..1599
    // Straights: 1600..1609
    // Three of a Kind: 1610..2467
    // Two Pair: 2468..3325
    // One Pair: 3326..6185
    // High Card: 6186..7462

    // Simple table generator for all subsets of size 5
    std::vector<uint32_t> unique_subsets;
    for (int a = 0; a < 13; ++a) {
        for (int b = a + 1; b < 13; ++b) {
            for (int c = b + 1; c < 13; ++c) {
                for (int d = c + 1; d < 13; ++d) {
                    for (int e = d + 1; e < 13; ++e) {
                        uint32_t mask = (1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e);
                        unique_subsets.push_back(mask);
                    }
                }
            }
        }
    }

    // Sort descending by hand strength
    auto is_straight = [](uint32_t m) -> int {
        // A, 5, 4, 3, 2 (Wheel)
        if (m == 0x100F) return 1; // 5-high straight
        for (int i = 0; i <= 8; ++i) {
            if (m == (0x1Fu << i)) return i + 2; // (i+2)-high straight
        }
        return 0;
    };

    std::sort(unique_subsets.begin(), unique_subsets.end(), [&](uint32_t x, uint32_t y) {
        int str_x = is_straight(x);
        int str_y = is_straight(y);
        if (str_x != str_y) return str_x > str_y;
        return x > y; // higher bits win
    });

    uint16_t cur_straight_flush = 1;
    uint16_t cur_flush = 323;
    uint16_t cur_straight = 1600;
    uint16_t cur_high_card = 6186;

    for (uint32_t mask : unique_subsets) {
        if (is_straight(mask)) {
            flushes[mask] = cur_straight_flush++;
            unique5[mask] = cur_straight++;
        } else {
            flushes[mask] = cur_flush++;
            unique5[mask] = cur_high_card++;
        }
    }

    // 2. Generate Paired hands & Prime Products
    // Quads: 4 of rank a, 1 of rank b
    uint16_t cur_quads = 11;
    for (int a = 12; a >= 0; --a) {
        for (int b = 12; b >= 0; --b) {
            if (a == b) continue;
            uint32_t prod = PRIMES[a] * PRIMES[a] * PRIMES[a] * PRIMES[a] * PRIMES[b];
            prime_products.push_back({prod, cur_quads++});
        }
    }

    // Full House: 3 of rank a, 2 of rank b
    uint16_t cur_full_house = 167;
    for (int a = 12; a >= 0; --a) {
        for (int b = 12; b >= 0; --b) {
            if (a == b) continue;
            uint32_t prod = PRIMES[a] * PRIMES[a] * PRIMES[a] * PRIMES[b] * PRIMES[b];
            prime_products.push_back({prod, cur_full_house++});
        }
    }

    // Trips: 3 of rank a, 1 of b, 1 of c
    uint16_t cur_trips = 1610;
    for (int a = 12; a >= 0; --a) {
        for (int b = 12; b >= 0; --b) {
            if (b == a) continue;
            for (int c = b - 1; c >= 0; --c) {
                if (c == a) continue;
                uint32_t prod = PRIMES[a] * PRIMES[a] * PRIMES[a] * PRIMES[b] * PRIMES[c];
                prime_products.push_back({prod, cur_trips++});
            }
        }
    }

    // Two Pair: 2 of rank a, 2 of b, 1 of c
    uint16_t cur_two_pair = 2468;
    for (int a = 12; a >= 0; --a) {
        for (int b = a - 1; b >= 0; --b) {
            for (int c = 12; c >= 0; --c) {
                if (c == a || c == b) continue;
                uint32_t prod = PRIMES[a] * PRIMES[a] * PRIMES[b] * PRIMES[b] * PRIMES[c];
                prime_products.push_back({prod, cur_two_pair++});
            }
        }
    }

    // One Pair: 2 of rank a, 1 of b, 1 of c, 1 of d
    uint16_t cur_one_pair = 3326;
    for (int a = 12; a >= 0; --a) {
        for (int b = 12; b >= 0; --b) {
            if (b == a) continue;
            for (int c = b - 1; c >= 0; --c) {
                if (c == a) continue;
                for (int d = c - 1; d >= 0; --d) {
                    if (d == a) continue;
                    uint32_t prod = PRIMES[a] * PRIMES[a] * PRIMES[b] * PRIMES[c] * PRIMES[d];
                    prime_products.push_back({prod, cur_one_pair++});
                }
            }
        }
    }

    std::sort(prime_products.begin(), prime_products.end());
    initialized = true;
}

uint16_t HandEvaluator::evaluate_5card(Card c1, Card c2, Card c3, Card c4, Card c5) {
    // 1. Check Flush
    if (c1 & c2 & c3 & c4 & c5 & 0xF00) {
        uint32_t q = (c1 | c2 | c3 | c4 | c5) >> 16;
        return flushes[q];
    }

    // 2. Check Straight & High Card
    uint32_t q = (c1 | c2 | c3 | c4 | c5) >> 16;
    if (__builtin_popcount(q) == 5) {
        return unique5[q];
    }

    // 3. Paired Hands via Prime Product
    uint32_t p = (c1 & 0xFF) * (c2 & 0xFF) * (c3 & 0xFF) * (c4 & 0xFF) * (c5 & 0xFF);
    
    // Binary search over 4,888 pre-sorted entries
    auto it = std::lower_bound(
        prime_products.begin(), prime_products.end(), p,
        [](const std::pair<uint32_t, uint16_t>& a, uint32_t val) {
            return a.first < val;
        }
    );
    return it->second;
}

uint16_t HandEvaluator::evaluate_7card(const std::array<Card, 7>& c) {
    uint16_t best = 9999;
    for (int i = 0; i < 21; ++i) {
        uint16_t score = evaluate_5card(
            c[C7_5[i][0]], c[C7_5[i][1]], c[C7_5[i][2]], c[C7_5[i][3]], c[C7_5[i][4]]
        );
        if (score < best) best = score;
    }
    return best;
}

void HandEvaluator::precompute_board_ranks(
    const std::vector<Card>& board,
    const std::array<std::pair<Card, Card>, 1326>& all_hands,
    std::array<uint16_t, 1326>& out_ranks
) {
    // Pack all 5 board cards into a bitwise mask
    uint32_t board_mask = 0;
    for (Card c : board) board_mask |= c;

    std::array<Card, 7> seven_cards;
    seven_cards[2] = board[0];
    seven_cards[3] = board[1];
    seven_cards[4] = board[2];
    seven_cards[5] = board[3];
    seven_cards[6] = board[4];

    for (int i = 0; i < 1326; ++i) {
        Card h1 = all_hands[i].first;
        Card h2 = all_hands[i].second;

        // Blocker check: does hero share cards with board?
        if ((h1 & board_mask) || (h2 & board_mask) || (h1 == h2)) {
            out_ranks[i] = 9999; // Invalid holding
            continue;
        }

        seven_cards[0] = h1;
        seven_cards[1] = h2;
        out_ranks[i] = evaluate_7card(seven_cards);
    }
}

float HandEvaluator::evaluate_showdown(
    int hero_idx,
    const std::array<uint16_t, 1326>& board_ranks,
    const std::array<float, 1326>& villain_range
) {
    uint16_t hero_score = board_ranks[hero_idx];
    if (hero_score == 9999) return -1.0f;

    float ev = 0.0f;
    float sum_weights = 0.0f;

    #pragma omp simd reduction(+:ev, sum_weights)
    for (int i = 0; i < 1326; ++i) {
        float w = villain_range[i];
        if (w <= 0.0f || board_ranks[i] == 9999) continue;

        sum_weights += w;
        if (hero_score < board_ranks[i]) {
            ev += w; // Hero wins (lower score is better)
        } else if (hero_score > board_ranks[i]) {
            ev -= w; // Hero loses
        }
        // Ties add 0.0
    }

    return (sum_weights > 0.0f) ? (ev / sum_weights) : 0.0f;
}