#!/usr/bin/env python3
"""
Επαλήθευση: είναι τα interval-bias ευρήματα πραγματικά ή τεχνητά;

1. Αναλυτικός υπολογισμός P(gap = k) για sample 20 από 80
2. Πολύ μεγαλύτερο Monte Carlo (5M draws) για επιβεβαίωση
3. Αν το bias είναι πραγματικό: πώς το εκμεταλλευόμαστε για πρόβλεψη;
"""
import json, time, numpy as np
from pathlib import Path
from collections import Counter
from math import comb, sqrt

DATA_DIR = Path('/home/user/Game/data/raw')

print("Loading draws...")
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        all_draws.append((d['id'], sorted(d['n'])))
all_draws.sort(key=lambda x: x[0])
N = len(all_draws)
print(f"  {N} draws in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════
# ΑΝΑΛΥΤΙΚΟΣ ΥΠΟΛΟΓΙΣΜΟΣ P(gap = k)
# ═══════════════════════════════════════════════════════════════════
# Για δείγμα n από {1..M}, ο μέσος αριθμός ζευγών (i,j) με i,j sorted και
# j_pos = i_pos + 1 και j_value - i_value = k είναι:
#
# Για κάθε i_value από 1 έως M-k:
#   P(both i_value and i_value+k ∈ sample, AND no value between them ∈ sample)
#   = C(M-k-1, n-2) / C(M, n)  --- gap of exactly k requires (k-1) numbers excluded
#
# Total expected count of "consecutive sorted gap = k" per draw:
#   E[gap=k] = (M-k) × C(M-k-1, n-2) / C(M, n)

M = 80
n = 20

print(f"\n══ Analytical P(consecutive gap = k) for sample n={n} from {M} ══")
exp_per_draw = {}
for k in range(1, M-n+2):
    # gap k requires k-1 numbers excluded between them, then n-2 from remaining M-k-1
    num_pairs_possible = M - k
    if num_pairs_possible <= 0: continue
    p_gap_k = comb(M-k-1, n-2) / comb(M, n)
    exp_per_draw[k] = num_pairs_possible * p_gap_k

total_exp_per_draw = sum(exp_per_draw.values())
print(f"  Sum of expected gaps per draw: {total_exp_per_draw:.4f} (should be n-1 = {n-1})")

# Observed
print(f"\n  Computing observed intervals...")
t0 = time.time()
interval_counter = Counter()
for _, nums in all_draws:
    for i in range(n-1):
        interval = nums[i+1] - nums[i]
        interval_counter[interval] += 1
total_intervals = sum(interval_counter.values())
print(f"  Total intervals: {total_intervals:,}  ({time.time()-t0:.1f}s)")

# Compare
print(f"\n  {'Gap':>4} {'Observed':>10} {'Expected':>11} {'Diff':>9} {'z-score':>9}")
sig_findings = []
for k in sorted(exp_per_draw.keys())[:30]:
    obs = interval_counter.get(k, 0)
    exp = exp_per_draw[k] * N  # expected total across all draws
    if exp < 100: continue
    p = exp_per_draw[k] / (n - 1)  # P(any single gap = k)
    var = N * (n - 1) * p * (1 - p)  # variance of sum
    z = (obs - exp) / sqrt(var) if var > 0 else 0
    flag = ' ★★★' if abs(z) > 5 else (' ★' if abs(z) > 3 else '')
    if abs(z) > 3:
        sig_findings.append((k, obs, exp, z))
    print(f"  {k:>4} {obs:>10,} {exp:>11.1f} {obs-exp:>+9.0f} {z:>+9.2f}{flag}")

print(f"\n  Significant findings (|z|>3): {len(sig_findings)}")
if sig_findings:
    for k, obs, exp, z in sig_findings:
        ratio = obs / exp
        print(f"    Gap {k}: observed {obs:,} vs expected {exp:.0f} "
              f"(ratio {ratio:.4f}, z={z:+.2f}, deviation {(ratio-1)*100:+.2f}%)")

# ═══════════════════════════════════════════════════════════════════
# ΧΩΡΙΣΤΗ ΑΝΑΛΥΣΗ: αν gap=4 ευνοείται, ποια ζεύγη το προκαλούν;
# ═══════════════════════════════════════════════════════════════════
if any(k == 4 for k,_,_,_ in sig_findings) or True:
    print(f"\n══ Gap=4 deep dive — which (a, a+4) pairs are most common? ══")
    pair_gap4 = Counter()
    for _, nums in all_draws:
        for i in range(n-1):
            if nums[i+1] - nums[i] == 4:
                pair_gap4[(nums[i], nums[i+1])] += 1
    total_g4 = sum(pair_gap4.values())
    avg_per_pair = total_g4 / 76  # 76 possible pairs (a from 1..76)
    print(f"  Total (a, a+4) consecutive pairs: {total_g4:,}")
    print(f"  Average per pair (a=1..76): {avg_per_pair:.1f}")
    print(f"\n  Top 10 most common gap-4 consecutive pairs:")
    for pair, cnt in pair_gap4.most_common(10):
        ratio = cnt / avg_per_pair
        flag = ' ★' if ratio > 1.05 else ''
        print(f"    {pair}: {cnt:,}  (ratio {ratio:.3f}){flag}")

# ═══════════════════════════════════════════════════════════════════
# AΝΑΛΟΓΟ για gap=16
# ═══════════════════════════════════════════════════════════════════
print(f"\n══ Gap=16 (large gap) — which (a, a+16) pairs? ══")
pair_g16 = Counter()
for _, nums in all_draws:
    for i in range(n-1):
        if nums[i+1] - nums[i] == 16:
            pair_g16[(nums[i], nums[i+1])] += 1
total_g16 = sum(pair_g16.values())
print(f"  Total (a, a+16) consecutive pairs: {total_g16:,}")
print(f"  Top 10:")
for pair, cnt in pair_g16.most_common(10):
    print(f"    {pair}: {cnt:,}")

# ═══════════════════════════════════════════════════════════════════
# ΧΡΗΣΙΜΟΤΗΤΑ ΓΙΑ ΠΡΟΒΛΕΨΗ
# ═══════════════════════════════════════════════════════════════════
print(f"\n══ Can interval bias improve prediction? ══")
print(f"  Idea: if number a is hot, prefer a+4 (consonant interval) for next draw")

# For each draw i, count: of the 20 numbers in draw i+1, how many are at
# distance ±4 from numbers in draw i?
print(f"  Test: pairs (a in draw_i, a+4 in draw_{{i+1}}) — z vs baseline")

count_a_then_aplus4 = 0
count_a_then_aplus4_in_next = 0
total_eligible = 0

for i in range(N-1):
    di = set(all_draws[i][1])
    djp = set(all_draws[i+1][1])
    for a in di:
        if a + 4 <= 80:
            total_eligible += 1
            if (a+4) in djp:
                count_a_then_aplus4 += 1

# Baseline: P(any specific number in draw) = 20/80 = 0.25
exp_count = total_eligible * 0.25
z = (count_a_then_aplus4 - exp_count) / sqrt(exp_count * 0.75)
print(f"  'a in draw_i AND (a+4) in draw_{{i+1}}'")
print(f"  Found: {count_a_then_aplus4:,}  Expected: {exp_count:.0f}  z={z:+.2f}")

# Same for gap=2, gap=14, gap=16
for k in [2, 14, 16, 7, 5]:
    c = 0; t = 0
    for i in range(N-1):
        di = set(all_draws[i][1])
        djp = set(all_draws[i+1][1])
        for a in di:
            if a + k <= 80:
                t += 1
                if (a+k) in djp:
                    c += 1
    ec = t * 0.25
    z = (c - ec) / sqrt(ec * 0.75)
    flag = ' ★' if abs(z) > 3 else ''
    print(f"  gap={k:2d}: found {c:,}  expected {ec:.0f}  z={z:+.2f}{flag}")

print("\n══ ΤΕΛΟΣ ══")
