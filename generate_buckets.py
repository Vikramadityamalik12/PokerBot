"""
generate_buckets.py  —  Person 3 (Hand-Strength Abstraction)
=============================================================
Responsibilities:
  1. Pre-Flop  : Exhaustively evaluate all 1,326 two-card starting combos
                 using 2,000 Monte Carlo simulations each, then cluster into
                 50 equity buckets and export → abstraction_lut.bin

  2. Post-Flop : Randomly sample 100,000 game states per street (Flop / Turn /
                 River), evaluate each with 500 Monte Carlo simulations, then
                 cluster into 50 equity buckets and export →
                 flop_lut.bin / turn_lut.bin / river_lut.bin

All LUT files are raw binary, one uint8 (0-49) per sampled state, where
Bucket 0 = weakest equity and Bucket 49 = strongest equity.

Dependencies: eval7, scikit-learn, numpy  (all available in the venv)
"""

import sys
import struct
import random
import itertools
import time

# Force UTF-8 output so Unicode characters print correctly on Windows terminals
# that default to cp1252. This is a no-op on Linux/macOS.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eval7
import numpy as np
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

NUM_BUCKETS          = 50      # K-Means clusters (0 = worst, 49 = best)

# Pre-flop settings
PREFLOP_SIMS         = 2_000   # Monte Carlo rollouts per 2-card combo

# Post-flop settings
POSTFLOP_SAMPLES     = 100_000 # Random game states sampled per street
POSTFLOP_SIMS        = 500     # Monte Carlo rollouts per sampled state

# Output filenames
PREFLOP_LUT_FILE     = "abstraction_lut.bin"
FLOP_LUT_FILE        = "flop_lut.bin"
TURN_LUT_FILE        = "turn_lut.bin"
RIVER_LUT_FILE       = "river_lut.bin"


# ===========================================================================
# Shared utility helpers
# ===========================================================================

def _evaluate_hand(hand7: list) -> int:
    return eval7.evaluate(hand7)


def _equity_vs_random_opponent(hole_cards: list,
                                board_cards: list,
                                n_sims: int) -> float:
    """
    Estimate the equity (win + 0.5*tie rate) of `hole_cards` against a
    single random opponent, completing an incomplete board to exactly 5
    community cards via Monte Carlo.

    Parameters
    ----------
    hole_cards  : list of eval7.Card  — exactly 2 cards
    board_cards : list of eval7.Card  — 0, 3, 4, or 5 cards already dealt
    n_sims      : int                 — number of Monte Carlo rollouts

    Returns
    -------
    float in [0.0, 1.0]
    """
    assert len(hole_cards) == 2, "hole_cards must contain exactly 2 cards"
    assert len(board_cards) in (0, 3, 4, 5), \
        f"board_cards must be 0/3/4/5 cards, got {len(board_cards)}"

    known_cards = set(hole_cards) | set(board_cards)
    board_needed = 5 - len(board_cards)   # how many more community cards

    # Build the filtered deck ONCE — known_cards never change across sims.
    # Inside the loop we only copy + shuffle, which is ~3.6x faster than
    # reconstructing the deck from eval7.Deck().cards every iteration.
    base_remaining = [c for c in eval7.Deck().cards if c not in known_cards]

    wins = 0.0

    for _ in range(n_sims):
        remaining = base_remaining.copy()
        random.shuffle(remaining)

        # Deal 2 cards to the opponent, then fill the rest of the board
        opp_cards   = remaining[:2]
        runout       = remaining[2: 2 + board_needed]
        full_board   = list(board_cards) + runout

        my_score  = _evaluate_hand(list(hole_cards) + full_board)
        opp_score = _evaluate_hand(opp_cards        + full_board)

        # eval7: higher score = stronger hand 
        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5

    return wins / n_sims


def _cluster_and_sort(equities: np.ndarray) -> np.ndarray:
    """
    Run K-Means on a 1-D equity array (reshaped to column vector), sort
    cluster centres from lowest → highest, and return remapped labels as
    numpy.uint8.

    Parameters
    ----------
    equities : np.ndarray  shape (N,)  — raw equity floats in [0, 1]

    Returns
    -------
    np.ndarray  shape (N,)  dtype=uint8  — bucket IDs in [0, NUM_BUCKETS-1]
    """
    X = equities.reshape(-1, 1)

    print(f"    Running KMeans(n_clusters={NUM_BUCKETS}) on {len(X):,} samples ...")
    t0 = time.time()
    km = KMeans(n_clusters=NUM_BUCKETS, random_state=42, n_init="auto")
    raw_labels = km.fit_predict(X)
    print(f"    KMeans done in {time.time() - t0:.1f}s")

    # Sort centroids: argsort gives old_cluster_id → rank (0=lowest equity)
    centres = km.cluster_centers_.flatten()
    sorted_idx = np.argsort(centres)                        # ascending order
    mapping = np.empty(NUM_BUCKETS, dtype=np.uint8)
    for new_id, old_id in enumerate(sorted_idx):
        mapping[old_id] = np.uint8(new_id)

    sorted_labels = mapping[raw_labels]                     # vectorised remap
    return sorted_labels


def _write_lut(labels: np.ndarray, filepath: str) -> None:
    """Pack each label as an unsigned byte and write to `filepath`."""
    with open(filepath, "wb") as fh:
        for label in labels:
            fh.write(struct.pack("B", int(label)))
    print(f"    >> Exported {len(labels):,} buckets to '{filepath}'\n")


# ===========================================================================
# 1. Pre-Flop Abstraction  (exhaustive 1,326 combos)
# ===========================================================================

def get_all_starting_hands() -> list:
    """Return all C(52,2) = 1,326 unique 2-card combinations."""
    all_cards = eval7.Deck().cards
    return list(itertools.combinations(all_cards, 2))


def generate_preflop_equities() -> np.ndarray:
    """
    For every possible 2-card starting hand, compute its equity via
    PREFLOP_SIMS Monte Carlo simulations (no known board cards yet).

    Returns
    -------
    np.ndarray  shape (1326,)  dtype=float64
    """
    hands = get_all_starting_hands()
    n = len(hands)
    equities = np.zeros(n, dtype=np.float64)

    print(f"[Pre-Flop] Simulating {n:,} hands x {PREFLOP_SIMS:,} rollouts each ...")
    t0 = time.time()

    for i, hole_cards in enumerate(hands):
        equities[i] = _equity_vs_random_opponent(
            list(hole_cards), board_cards=[], n_sims=PREFLOP_SIMS
        )
        if (i + 1) % 200 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (n - i - 1)
            print(f"    {i+1:>5}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"[Pre-Flop] Equity simulation complete in {time.time()-t0:.1f}s")
    return equities


def build_preflop_lut() -> None:
    """Full pre-flop pipeline: simulate → cluster → export."""
    print("\n" + "="*60)
    print("STAGE 1 - Pre-Flop Abstraction")
    print("="*60)
    equities = generate_preflop_equities()
    print(f"    Equity range: [{equities.min():.4f}, {equities.max():.4f}]")
    labels = _cluster_and_sort(equities)
    _write_lut(labels, PREFLOP_LUT_FILE)


# ===========================================================================
# 2. Post-Flop Abstraction  (Flop / Turn / River — random sampling)
# ===========================================================================

def _sample_postflop_equities(board_size: int, n_samples: int, n_sims: int) -> np.ndarray:
    """
    Randomly sample `n_samples` game states for a given street and compute
    the equity of a random hero hand vs. a random opponent on that street.

    Parameters
    ----------
    board_size : int — number of community cards on this street (3/4/5)
    n_samples  : int — number of random game states to evaluate
    n_sims     : int — Monte Carlo rollouts per game state

    Returns
    -------
    np.ndarray  shape (n_samples,)  dtype=float64
    """
    equities = np.zeros(n_samples, dtype=np.float64)
    all_cards = eval7.Deck().cards

    t0 = time.time()
    for i in range(n_samples):
        # Sample a random game state: shuffle the deck, deal hero + board
        random.shuffle(all_cards)
        hole_cards = all_cards[:2]
        board_cards = all_cards[2: 2 + board_size]

        equities[i] = _equity_vs_random_opponent(
            list(hole_cards), list(board_cards), n_sims
        )

        if (i + 1) % 10_000 == 0 or (i + 1) == n_samples:
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (n_samples - i - 1)
            print(f"    {i+1:>7,}/{n_samples:,}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"    Sampling done in {time.time()-t0:.1f}s")
    return equities


def build_postflop_lut(street_name: str, board_size: int, filepath: str) -> None:
    """Full post-flop pipeline for one street: sample → cluster → export."""
    print("\n" + "="*60)
    print(f"STAGE - {street_name} Abstraction  (board_size={board_size})")
    print("="*60)
    print(f"    Sampling {POSTFLOP_SAMPLES:,} states x {POSTFLOP_SIMS:,} rollouts ...")
    equities = _sample_postflop_equities(
        board_size=board_size,
        n_samples=POSTFLOP_SAMPLES,
        n_sims=POSTFLOP_SIMS,
    )
    print(f"    Equity range: [{equities.min():.4f}, {equities.max():.4f}]")
    labels = _cluster_and_sort(equities)
    _write_lut(labels, filepath)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    overall_start = time.time()

    # ── Stage 1 : Pre-Flop ──────────────────────────────────────────────────
    build_preflop_lut()

    # ── Stage 2 : Flop  (3 board cards) ─────────────────────────────────────
    build_postflop_lut(
        street_name="Flop",
        board_size=3,
        filepath=FLOP_LUT_FILE,
    )

    # ── Stage 3 : Turn  (4 board cards) ─────────────────────────────────────
    build_postflop_lut(
        street_name="Turn",
        board_size=4,
        filepath=TURN_LUT_FILE,
    )

    # ── Stage 4 : River (5 board cards) ─────────────────────────────────────
    build_postflop_lut(
        street_name="River",
        board_size=5,
        filepath=RIVER_LUT_FILE,
    )

    total = time.time() - overall_start
    print("\n" + "="*60)
    print(f"ALL DONE  -  total runtime {total/60:.1f} minutes")
    print("Output files:")
    print(f"  {PREFLOP_LUT_FILE}  (1,326 entries)")
    print(f"  {FLOP_LUT_FILE}     ({POSTFLOP_SAMPLES:,} entries)")
    print(f"  {TURN_LUT_FILE}     ({POSTFLOP_SAMPLES:,} entries)")
    print(f"  {RIVER_LUT_FILE}    ({POSTFLOP_SAMPLES:,} entries)")
    print("="*60)
