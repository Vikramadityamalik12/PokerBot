"""
lut_loader.py  — Runtime Interface
=============================================
Loads the four pre-computed hand-strength abstraction LUTs into memory
and exposes a clean, fast API for CFR+ Solver and Bayesian Tracker to call at runtime.

LUT files produced by generate_buckets.py:
    abstraction_lut.bin   1,326 bytes  — pre-flop  bucket per hand combo
    flop_lut.bin        100,000 bytes  — flop       bucket distribution
    turn_lut.bin        100,000 bytes  — turn       bucket distribution
    river_lut.bin       100,000 bytes  — river      bucket distribution

Bucket convention (all streets):
    0  = absolute weakest equity
    49 = absolute strongest equity

"""

import struct
import random
import itertools
import pathlib
import time
from typing import Union

import eval7

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BUCKETS       = 50          # must match generate_buckets.py
RUNTIME_SIMS      = 200         # Monte Carlo rollouts for post-flop runtime
                                # (200 sims ≈ 1-3ms per call, ~1.5% std error)

# Default file paths — all relative to this file's directory
_HERE = pathlib.Path(__file__).parent
PREFLOP_LUT_PATH  = _HERE / "abstraction_lut.bin"
FLOP_LUT_PATH     = _HERE / "flop_lut.bin"
TURN_LUT_PATH     = _HERE / "turn_lut.bin"
RIVER_LUT_PATH    = _HERE / "river_lut.bin"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_bin(path: pathlib.Path) -> list:
    """Read a raw .bin LUT file and return a list of uint8 bucket IDs."""
    data = pathlib.Path(path).read_bytes()
    return list(struct.unpack(f"{len(data)}B", data))


def _cards_to_eval7(cards: list) -> list:
    """
    Accept cards as eval7.Card objects or strings like 'Ah', 'Td', '2c'.
    Always returns a list of eval7.Card objects.
    """
    result = []
    for c in cards:
        if isinstance(c, str):
            result.append(eval7.Card(c))
        else:
            result.append(c)    # already an eval7.Card
    return result


def _equity_monte_carlo(hole_cards: list,
                         board_cards: list,
                         n_sims: int) -> float:
    """
    Fast Monte Carlo equity estimate.
    hole_cards / board_cards must be lists of eval7.Card objects.
    Returns win rate in [0.0, 1.0] (higher = stronger hand).
    """
    known      = set(hole_cards) | set(board_cards)
    base_deck  = [c for c in eval7.Deck().cards if c not in known]
    board_need = 5 - len(board_cards)

    wins = 0.0
    for _ in range(n_sims):
        deck = base_deck.copy()
        random.shuffle(deck)
        opp       = deck[:2]
        runout    = deck[2: 2 + board_need]
        full_board = list(board_cards) + runout

        my_score  = eval7.evaluate(list(hole_cards) + full_board)
        opp_score = eval7.evaluate(opp + full_board)

        # eval7 convention: higher score = stronger hand
        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5

    return wins / n_sims


def _equity_to_bucket(equity: float) -> int:
    """
    Map a raw equity float in [0, 1] to a sorted bucket ID in [0, 49].
    Uses linear quantisation — valid because K-Means on a 1-D equity
    distribution with 50 clusters produces approximately uniform centroids
    across [0, 1].
    """
    bucket = int(equity * NUM_BUCKETS)
    return min(bucket, NUM_BUCKETS - 1)     # clamp top edge (equity == 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LUTLoader:
    """
    Loads all four hand-strength LUTs once and exposes fast bucket lookups.

    Instantiate once at bot startup and reuse throughout the session:
        lut = LUTLoader()
    """

    def __init__(self,
                 preflop_path: str = None,
                 flop_path:    str = None,
                 turn_path:    str = None,
                 river_path:   str = None):
        """
        Load all four LUT files into memory.
        Optionally override file paths (useful if running from a different CWD).
        """
        t0 = time.time()

        # ── Load raw byte tables ──────────────────────────────────────────
        self._preflop_lut = _load_bin(preflop_path or PREFLOP_LUT_PATH)
        self._flop_lut    = _load_bin(flop_path    or FLOP_LUT_PATH)
        self._turn_lut    = _load_bin(turn_path    or TURN_LUT_PATH)
        self._river_lut   = _load_bin(river_path   or RIVER_LUT_PATH)

        # ── Build pre-flop index  (card_pair frozenset → LUT index) ──────
        # The index order is identical to itertools.combinations(deck, 2)
        # with eval7's canonical deck ordering used during training.
        deck = eval7.Deck().cards
        self._hand_to_idx: dict = {}
        for i, (c1, c2) in enumerate(itertools.combinations(deck, 2)):
            self._hand_to_idx[frozenset([c1, c2])] = i

        elapsed = time.time() - t0
        print(f"[LUTLoader] Loaded all 4 LUTs in {elapsed*1000:.1f}ms")
        print(f"  Pre-flop : {len(self._preflop_lut):,} entries  ({PREFLOP_LUT_PATH.name})")
        print(f"  Flop     : {len(self._flop_lut):,} entries  ({FLOP_LUT_PATH.name})")
        print(f"  Turn     : {len(self._turn_lut):,} entries  ({TURN_LUT_PATH.name})")
        print(f"  River    : {len(self._river_lut):,} entries  ({RIVER_LUT_PATH.name})")

    # ------------------------------------------------------------------
    # Pre-flop lookup  (exact, O(1), no Monte Carlo needed)
    # ------------------------------------------------------------------

    def preflop_bucket(self,
                       card1: Union[str, eval7.Card],
                       card2: Union[str, eval7.Card]) -> int:
        """
        Return the pre-flop bucket (0-49) for a two-card starting hand.

        Parameters
        ----------
        card1, card2 : str or eval7.Card
            e.g. "Ah", "Ks", eval7.Card("Td")

        Returns
        -------
        int in [0, 49]  — 0 = weakest, 49 = strongest
        """
        c1, c2 = _cards_to_eval7([card1, card2])
        key     = frozenset([c1, c2])
        idx     = self._hand_to_idx.get(key)
        if idx is None:
            raise ValueError(f"Unknown card pair: {card1}, {card2}")
        return int(self._preflop_lut[idx])

    def preflop_bucket_by_index(self, hand_index: int) -> int:
        """
        Direct lookup by pre-computed hand index (0-1325).
        Fastest possible call — used internally by Person 2.
        """
        return int(self._preflop_lut[hand_index])

    def hand_index(self,
                   card1: Union[str, eval7.Card],
                   card2: Union[str, eval7.Card]) -> int:
        """
        Return the canonical hand index (0-1325) for a hole card pair.
        Person 4 uses this to index into the 1,326-entry villain range array.
        """
        c1, c2 = _cards_to_eval7([card1, card2])
        key     = frozenset([c1, c2])
        idx     = self._hand_to_idx.get(key)
        if idx is None:
            raise ValueError(f"Unknown card pair: {card1}, {card2}")
        return idx

    # ------------------------------------------------------------------
    # Post-flop lookup  (equity computed, then bucketed)
    # ------------------------------------------------------------------

    def postflop_bucket(self,
                        hole_cards:  list,
                        board_cards: list,
                        street:      str = None,
                        n_sims:      int = RUNTIME_SIMS) -> int:
        """
        Return the post-flop bucket (0-49) for a hero hand on a given board.

        The equity is estimated via Monte Carlo (default: 200 sims ≈ 2ms)
        and then mapped to a bucket using the same 50-cluster scale as the
        K-Means training.

        Parameters
        ----------
        hole_cards  : list of str or eval7.Card  — exactly 2 hero cards
        board_cards : list of str or eval7.Card  — 3 (flop), 4 (turn), 5 (river)
        street      : str, optional — "flop" | "turn" | "river"
                      Not required for bucket computation but used for
                      validation and debug logging if provided.
        n_sims      : int — Monte Carlo rollouts (trade speed vs. accuracy)

        Returns
        -------
        int in [0, 49]
        """
        h = _cards_to_eval7(hole_cards)
        b = _cards_to_eval7(board_cards)

        n_board = len(b)
        if n_board not in (3, 4, 5):
            raise ValueError(f"board_cards must have 3/4/5 cards, got {n_board}")

        equity = _equity_monte_carlo(h, b, n_sims)
        return _equity_to_bucket(equity)

    def compute_equity(self,
                       hole_cards:  list,
                       board_cards: list,
                       n_sims:      int = RUNTIME_SIMS) -> float:
        """
        Return raw equity float in [0.0, 1.0] without bucketing.
        Useful for Person 2's EV calculation and the dashboard display.

        Parameters
        ----------
        hole_cards  : list of str or eval7.Card  — 2 hero cards
        board_cards : list of str or eval7.Card  — 0, 3, 4, or 5 board cards
        n_sims      : int — Monte Carlo rollouts

        Returns
        -------
        float in [0.0, 1.0]
        """
        h = _cards_to_eval7(hole_cards)
        b = _cards_to_eval7(board_cards)
        return _equity_monte_carlo(h, b, n_sims)

    # ------------------------------------------------------------------
    # Convenience: get bucket for any street automatically
    # ------------------------------------------------------------------

    def get_bucket(self,
                   hole_cards:  list,
                   board_cards: list,
                   n_sims:      int = RUNTIME_SIMS) -> int:
        """
        Auto-detect street from board length and return the bucket.
        The single entry point Person 2 and Person 4 should call.

        board_cards length:
            0  → pre-flop  (exact lookup, n_sims ignored)
            3  → flop      (Monte Carlo)
            4  → turn      (Monte Carlo)
            5  → river     (Monte Carlo)

        Returns
        -------
        int in [0, 49]
        """
        b = _cards_to_eval7(board_cards)

        if len(b) == 0:
            # Pre-flop: exact O(1) lookup
            h = _cards_to_eval7(hole_cards)
            return self.preflop_bucket(h[0], h[1])
        else:
            # Post-flop: Monte Carlo
            return self.postflop_bucket(hole_cards, board_cards, n_sims=n_sims)

    # ------------------------------------------------------------------
    # Utility: all 1,326 hands as (card1, card2) string tuples
    # ------------------------------------------------------------------

    def all_hands(self) -> list:
        """
        Return the full ordered list of all 1,326 canonical hand tuples.
        Index i in this list corresponds to index i in the pre-flop LUT.
        Used by Person 4 to initialise the villain range array.

        Returns
        -------
        list of (str, str) — e.g. [("2c","2d"), ("2c","2h"), ..., ("Ah","As")]
        """
        deck = eval7.Deck().cards
        return [(str(c1), str(c2))
                for c1, c2 in itertools.combinations(deck, 2)]


# ---------------------------------------------------------------------------
# Quick self-test  (run:  python lut_loader.py)
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     print("=" * 55)
#     print("LUTLoader Self-Test")
#     print("=" * 55)

#     lut = LUTLoader()
#     print()

#     # ── Pre-flop sanity checks ────────────────────────────────────────────
#     aa_bucket  = lut.preflop_bucket("Ah", "As")
#     bad_bucket = lut.preflop_bucket("2c", "7h")
#     print(f"Pre-flop AA   bucket : {aa_bucket:>2}  (expect 48 or 49)")
#     print(f"Pre-flop 72o  bucket : {bad_bucket:>2}  (expect 0-5)")
#     assert aa_bucket > bad_bucket, "AA must be in a higher bucket than 72o"
#     print("  PASSED: AA > 72o")

#     # ── Hand index ───────────────────────────────────────────────────────
#     idx_aa  = lut.hand_index("Ah", "As")
#     idx_72o = lut.hand_index("2c", "7h")
#     print(f"\nHand index AA   : {idx_aa}   (expect 1325)")
#     print(f"Hand index 72o  : {idx_72o}")

#     # ── Post-flop checks ─────────────────────────────────────────────────
#     print("\nPost-flop equity tests (200 sims each, ~1-2s total) ...")

#     # AA on dry board — should be very high bucket
#     aa_flop = lut.postflop_bucket(["Ah", "As"], ["Kd", "7c", "2h"])
#     print(f"AA on K72 flop   bucket : {aa_flop:>2}  (expect >= 35)")

#     # 72o on AAA board — should be near bottom
#     bad_river = lut.postflop_bucket(["2c", "7h"], ["Ah", "Ad", "As", "Kd", "Qh"])
#     print(f"72o on AAAKQ board bucket: {bad_river:>2}  (expect <= 10)")

#     # Raw equity
#     eq = lut.compute_equity(["Ah", "As"], ["Kd", "7c", "2h"])
#     print(f"\nRaw equity AA on K72 flop: {eq:.3f}  (expect ~0.90)")

#     # ── Auto-detect street ───────────────────────────────────────────────
#     auto_pre  = lut.get_bucket(["Ah", "As"], [])
#     auto_flop = lut.get_bucket(["Ah", "As"], ["Kd", "7c", "2h"])
#     print(f"\nget_bucket() pre-flop : {auto_pre}")
#     print(f"get_bucket() flop     : {auto_flop}")

#     print("\n" + "=" * 55)
#     print("All self-tests PASSED")
#     print("=" * 55)
