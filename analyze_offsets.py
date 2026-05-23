#!/usr/bin/env python3
"""
KINO Toroidal Square Offset Analysis
Finds best offset rules for predicting next-draw numbers from single-square triggers.
"""

import json
import glob
from collections import defaultdict
from itertools import combinations

# ─── 1. Load all draws ───────────────────────────────────────────────────────

files = sorted(glob.glob('/home/user/Game/data/raw/kino_raw_*.json'))
all_draws = []
for path in files:
    with open(path) as f:
        d = json.load(f)
    all_draws.extend(d['draws'])

# Sort by draw ID to ensure chronological order
all_draws.sort(key=lambda x: x['id'])
print(f"Total draws loaded: {len(all_draws)}")

# ─── 2. Build toroidal square detector ───────────────────────────────────────
# Grid: rows 0-7 (8 rows), cols 0-9 (10 cols)
# number = row*10 + col + 1  (1-80)
# Square roots at (r, c): the 4 corners are:
#   (r,c), (r,(c+2)%10), ((r+2)%8,c), ((r+2)%8,(c+2)%10)
# In 1-indexed numbers:
#   root = r*10 + c + 1
#   n2   = r*10 + (c+2)%10 + 1
#   n3   = ((r+2)%8)*10 + c + 1
#   n4   = ((r+2)%8)*10 + (c+2)%10 + 1

def build_all_squares():
    """Return list of (root, frozenset_of_4_numbers)."""
    squares = []
    for r in range(8):
        for c in range(10):
            root = r * 10 + c + 1
            n2 = r * 10 + (c + 2) % 10 + 1
            n3 = ((r + 2) % 8) * 10 + c + 1
            n4 = ((r + 2) % 8) * 10 + (c + 2) % 10 + 1
            squares.append((root, frozenset([root, n2, n3, n4])))
    return squares

ALL_SQUARES = build_all_squares()
print(f"Total possible toroidal squares: {len(ALL_SQUARES)}")

# Pre-build lookup: for each number, which squares contain it
num_to_squares = defaultdict(list)
for sq_idx, (root, sq_set) in enumerate(ALL_SQUARES):
    for num in sq_set:
        num_to_squares[num].append(sq_idx)

# ─── 3. Find single-square triggers ──────────────────────────────────────────

def find_squares_in_draw(draw_set):
    """Return list of (root, sq_set) for all squares fully contained in draw_set."""
    # Candidate squares: those touching any number in draw_set
    candidate_indices = set()
    for num in draw_set:
        candidate_indices.update(num_to_squares[num])

    found = []
    for idx in candidate_indices:
        root, sq_set = ALL_SQUARES[idx]
        if sq_set.issubset(draw_set):
            found.append((root, sq_set))
    return found

triggers = []  # list of (draw_index, root)

for i, draw in enumerate(all_draws):
    draw_set = set(draw['n'])
    squares = find_squares_in_draw(draw_set)
    if len(squares) == 1:
        root = squares[0][0]
        triggers.append((i, root))

print(f"Single-square triggers: {len(triggers)}")

# ─── 4. Collect winning offsets for P1 (next draw) ───────────────────────────

offset_counts = [0] * 80  # offset 0-79

# For triggers where there IS a next draw
valid_triggers = [(i, root) for i, root in triggers if i + 1 < len(all_draws)]
print(f"Valid triggers (with next draw): {len(valid_triggers)}")

# Store per-trigger offset sets for combination analysis
trigger_offset_sets = []  # list of sets of winning offsets

for i, root in valid_triggers:
    next_draw = all_draws[i + 1]['n']
    winning_offsets = set()
    for num in next_draw:
        offset = (num - 1 - (root - 1)) % 80
        offset_counts[offset] += 1
        winning_offsets.add(offset)
    trigger_offset_sets.append(winning_offsets)

print(f"\nOffset frequency computed over {len(valid_triggers)} triggers.")

# ─── 5. Top 20 most frequent offsets ─────────────────────────────────────────

sorted_offsets = sorted(range(80), key=lambda o: -offset_counts[o])
top20 = sorted_offsets[:20]

print("\n=== TOP 20 MOST FREQUENT INDIVIDUAL OFFSETS ===")
print(f"{'Rank':<6} {'Offset':<8} {'Count':<8} {'% of triggers':<15}")
for rank, o in enumerate(top20, 1):
    pct = offset_counts[o] / len(valid_triggers) * 100
    print(f"{rank:<6} {o:<8} {offset_counts[o]:<8} {pct:.2f}%")

# ─── 6. Evaluate a rule (list of offsets) against all triggers ────────────────

def evaluate_rule(offsets, trigger_offset_sets):
    """Return (all_hit_count, at_least_4_count) for a 6-offset rule."""
    offsets_set = frozenset(offsets)
    n = len(offsets)
    all_hits = 0
    at_least_4 = 0
    for winning in trigger_offset_sets:
        hits = len(offsets_set & winning)
        if hits == n:
            all_hits += 1
        if hits >= 4:
            at_least_4 += 1
    return all_hits, at_least_4

# ─── 7. Best 6-offset combinations from top 20 ───────────────────────────────

print("\n=== SEARCHING 6-OFFSET COMBINATIONS FROM TOP 20 ===")
print("(This may take a moment...)")

# C(20,6) = 38,760 combinations — very fast
best_all6 = []  # (all_hit_count, at_least4_count, combo)
best_4of6 = []  # (at_least4_count, all_hit_count, combo)

for combo in combinations(top20, 6):
    all_hits, at4 = evaluate_rule(combo, trigger_offset_sets)
    best_all6.append((all_hits, at4, combo))
    best_4of6.append((at4, all_hits, combo))

best_all6.sort(reverse=True)
best_4of6.sort(reverse=True)

print("\n=== TOP 5 RULES BY ALL-6 HITS IN P1 ===")
print(f"{'Rank':<6} {'Offsets':<35} {'All-6 hits':<12} {'4-of-6 hits':<12}")
for rank, (ah, a4, combo) in enumerate(best_all6[:5], 1):
    print(f"{rank:<6} {str(list(combo)):<35} {ah:<12} {a4:<12}")

print("\n=== TOP 5 RULES BY 4-OF-6 HITS IN P1 ===")
print(f"{'Rank':<6} {'Offsets':<35} {'4-of-6 hits':<12} {'All-6 hits':<12}")
for rank, (a4, ah, combo) in enumerate(best_4of6[:5], 1):
    print(f"{rank:<6} {str(list(combo)):<35} {a4:<12} {ah:<12}")

# ─── 8. 7-offset combinations for R6 ─────────────────────────────────────────

print("\n=== SEARCHING 7-OFFSET COMBINATIONS FROM TOP 20 ===")
print("(C(20,7) = 77,520 combinations...)")

def evaluate_rule7(offsets, trigger_offset_sets):
    """Return (all_hit_count, at_least_4_count) for a 7-offset rule."""
    offsets_set = frozenset(offsets)
    n = len(offsets)
    all_hits = 0
    at_least_4 = 0
    for winning in trigger_offset_sets:
        hits = len(offsets_set & winning)
        if hits == n:
            all_hits += 1
        if hits >= 4:
            at_least_4 += 1
    return all_hits, at_least_4

best_all7 = []
best_4of7 = []

for combo in combinations(top20, 7):
    all_hits, at4 = evaluate_rule7(combo, trigger_offset_sets)
    best_all7.append((all_hits, at4, combo))
    best_4of7.append((at4, all_hits, combo))

best_all7.sort(reverse=True)
best_4of7.sort(reverse=True)

print("\n=== TOP 5 SEVEN-OFFSET RULES BY ALL-7 HITS IN P1 ===")
print(f"{'Rank':<6} {'Offsets':<45} {'All-7 hits':<12} {'4-of-7 hits':<12}")
for rank, (ah, a4, combo) in enumerate(best_all7[:5], 1):
    print(f"{rank:<6} {str(list(combo)):<45} {ah:<12} {a4:<12}")

print("\n=== TOP 5 SEVEN-OFFSET RULES BY 4-OF-7 HITS IN P1 ===")
print(f"{'Rank':<6} {'Offsets':<45} {'4-of-7 hits':<12} {'All-7 hits':<12}")
for rank, (a4, ah, combo) in enumerate(best_4of7[:5], 1):
    print(f"{rank:<6} {str(list(combo)):<45} {a4:<12} {ah:<12}")

# ─── 9. Current rule performance ─────────────────────────────────────────────

current_rules = {
    'R1': [6, 8, 33, 38, 45, 61],
    'R2': [28, 31, 39, 45, 47, 78],
    'R3': [1, 17, 29, 30, 58, 64],
    'R4': [22, 23, 48, 63, 66, 67],
    'R5': [6, 7, 12, 16, 26, 56],
    'R6': [0, 3, 24, 32, 35, 51, 73],
}

print("\n=== CURRENT RULE PERFORMANCE ===")
print(f"{'Rule':<6} {'Offsets':<45} {'All-N hits':<12} {'4-of-N hits':<12} {'N':<4}")
for name, offsets in current_rules.items():
    n = len(offsets)
    all_hits, at4 = evaluate_rule(offsets, trigger_offset_sets)
    print(f"{name:<6} {str(offsets):<45} {all_hits:<12} {at4:<12} {n:<4}")

print("\nDone.")
