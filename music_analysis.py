#!/usr/bin/env python3
"""
KINO Music Analysis — 3 experiments:

1. Μελωδικά διαστήματα (Melodic intervals): differences between consecutive
   sorted numbers in a draw. Are intervals 1, 2, 3, ... uniformly distributed?
   For random hypergeometric, expected distribution is computable.

2. Αρμονικές συγχορδίες (Harmonic chords): triples (a, a+4, a+7) ≈ major triad,
   (a, a+3, a+7) ≈ minor triad on a numeric scale. Do they co-occur more often
   than random?

3. Audio FFT: treat each draw as 20 simultaneous tones (frequencies = numbers),
   stack draws as time series. FFT may reveal harmonic structure not visible
   to standard statistical tests.

Plus a bonus: Most common melodic patterns (consecutive interval sequences).
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
# EXPERIMENT 1: Μελωδικά διαστήματα μέσα στην κλήρωση
# ═══════════════════════════════════════════════════════════════════
print("\n══ Experiment 1: Intervals between consecutive sorted numbers ══")
print("  Each draw has 19 intervals (gaps). Sum = 80 - 1 - <19 numbers> ≈ 60")

t0 = time.time()
interval_counter = Counter()
for _, nums in all_draws:
    for i in range(19):
        interval = nums[i+1] - nums[i]
        interval_counter[interval] += 1
total_intervals = sum(interval_counter.values())
print(f"  Total intervals: {total_intervals:,}  ({time.time()-t0:.1f}s)")

# Expected distribution for hypergeometric gaps:
# In a sample of 20 from {1..80}, the gap distribution can be computed exactly.
# P(gap=k) ≈ (number of placements with that gap) / total placements
# Easier: compute expected by simulation
print(f"\n  Comparing observed vs Monte Carlo expected (1M random draws):")
rng = np.random.default_rng(42)
exp_counter = Counter()
for _ in range(50000):
    sample = sorted(rng.choice(80, 20, replace=False) + 1)
    for i in range(19):
        exp_counter[sample[i+1] - sample[i]] += 1
exp_total = sum(exp_counter.values())

print(f"\n  {'Interval':>9} {'Observed':>9} {'Obs %':>7} {'Expected %':>11} {'z-score':>9}")
sig_intervals = []
for k in sorted(interval_counter.keys())[:20]:
    obs = interval_counter[k]
    obs_pct = obs / total_intervals * 100
    exp_pct = exp_counter.get(k, 0) / exp_total * 100
    if exp_pct > 0:
        exp_count = exp_pct/100 * total_intervals
        z = (obs - exp_count) / sqrt(exp_count * (1 - exp_pct/100))
        flag = ' ★' if abs(z) > 3 else ''
        if abs(z) > 3:
            sig_intervals.append((k, z, obs_pct, exp_pct))
        print(f"  {k:>9} {obs:>9,} {obs_pct:>6.3f}% {exp_pct:>10.3f}% {z:>+9.2f}{flag}")

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Αρμονικές συγχορδίες
# ═══════════════════════════════════════════════════════════════════
print("\n══ Experiment 2: Musical 'triads' co-occurrence ══")
print("  Major triad: (a, a+4, a+7)   Minor triad: (a, a+3, a+7)")
print("  Octave pair: (a, a+12)        Fifth pair: (a, a+7)")

t0 = time.time()
major_triads = 0
minor_triads = 0
octave_pairs = 0
fifth_pairs = 0
draw_sets = [frozenset(n) for _, n in all_draws]

for ds in draw_sets:
    for a in ds:
        if a + 4 <= 80 and a + 7 <= 80 and (a+4) in ds and (a+7) in ds:
            major_triads += 1
        if a + 3 <= 80 and a + 7 <= 80 and (a+3) in ds and (a+7) in ds:
            minor_triads += 1
        if a + 12 <= 80 and (a+12) in ds:
            octave_pairs += 1
        if a + 7 <= 80 and (a+7) in ds:
            fifth_pairs += 1

# Expected: for triple (a, b, c) within {1..80}, all in draw of 20:
# P = C(77,17)/C(80,20)  per specific triple
# Number of possible major triads = 73 (a can be 1..73)
# Expected major triads per draw = 73 * C(77,17)/C(80,20)
# Number of fifths = 73, octaves = 68

p_triple = comb(77, 17) / comb(80, 20)
p_pair   = comb(78, 18) / comb(80, 20)
exp_major = N * 73 * p_triple
exp_minor = N * 73 * p_triple
exp_octave = N * 68 * p_pair
exp_fifth = N * 73 * p_pair

print(f"  Done in {time.time()-t0:.1f}s")
print(f"\n  {'Pattern':>15} {'Found':>10} {'Expected':>10} {'Ratio':>7} {'z-score':>9}")
for label, obs, exp_ in [('Major triad', major_triads, exp_major),
                         ('Minor triad', minor_triads, exp_minor),
                         ('Octave pair', octave_pairs, exp_octave),
                         ('Fifth pair',  fifth_pairs,  exp_fifth)]:
    z = (obs - exp_) / sqrt(exp_)
    flag = ' ★' if abs(z) > 3 else ''
    print(f"  {label:>15} {obs:>10,} {exp_:>10.0f} {obs/exp_:>7.3f} {z:>+9.2f}{flag}")

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Top μελωδικά μοτίβα (interval sequences)
# ═══════════════════════════════════════════════════════════════════
print("\n══ Experiment 3: Top melodic patterns (3-note interval sequences) ══")
print("  Each draw has 19 intervals → 17 trigrams of consecutive intervals.")
t0 = time.time()

pattern_counter = Counter()
for _, nums in all_draws:
    intervals = tuple(nums[i+1] - nums[i] for i in range(19))
    for i in range(17):
        pattern_counter[(intervals[i], intervals[i+1], intervals[i+2])] += 1

total_patterns = sum(pattern_counter.values())
print(f"  Total trigram patterns: {total_patterns:,}  unique: {len(pattern_counter):,}")
print(f"  ({time.time()-t0:.1f}s)\n")

print(f"  Top 15 most common interval trigrams:")
for pattern, cnt in pattern_counter.most_common(15):
    # Expected: P(all 3 intervals = specific) — depends on values, but baseline:
    # For a random arrangement, the probability of any specific small pattern
    # like (1,1,1) is rare because intervals tend to be 2-5
    print(f"    {pattern}: {cnt:,}×")

# Look for repeated patterns like (1,1,1) - 4 consecutive numbers
streaks_4plus = sum(cnt for pat, cnt in pattern_counter.items() if pat == (1,1,1))
print(f"\n  4 consecutive numbers (pattern 1,1,1): {streaks_4plus:,} occurrences")

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 4: AUDIO FFT — treat draws as additive synthesis
# Generate a "soundwave" where each draw contributes 20 sine waves
# at frequencies proportional to the numbers. Then FFT the whole thing.
# ═══════════════════════════════════════════════════════════════════
print("\n══ Experiment 4: Audio FFT (additive synthesis of draws) ══")

# Build a time series: for each draw, a window of (samples_per_draw) samples
# where 20 sine waves are summed. Frequency = number (1-80 Hz analog).
# We'll use a simpler proxy: stack the 80-vector for each draw, FFT across time
# to find frequencies of fluctuation.

t0 = time.time()
# Each draw = 80-element binary vector. FFT across the N×80 matrix in TIME dimension.
M = np.zeros((N, 80), dtype=np.float32)
for i, (_, nums) in enumerate(all_draws):
    for n in nums:
        M[i, n-1] = 1.0
print(f"  Matrix built {M.shape}  in {time.time()-t0:.1f}s")

# FFT of each number's time series, look for peaks above noise floor
t0 = time.time()
print(f"  Computing FFT per number...")

# Use power spectrum: |FFT|^2
spec = np.zeros(N // 2 + 1)
for n in range(80):
    series = M[:, n] - M[:, n].mean()  # remove DC
    F = np.fft.rfft(series)
    spec += (F * np.conj(F)).real

print(f"  Done in {time.time()-t0:.1f}s")

# Top peaks: ignore very low frequencies (long-term trends) and DC
search_start = 10  # ignore first 10 bins
top_idx = np.argsort(spec[search_start:])[-20:][::-1] + search_start

print(f"\n  Top 20 frequency bins (period in draws = N/bin):")
print(f"  {'Bin':>6} {'Period (draws)':>14} {'Power':>12} {'Ratio vs median':>16}")
median_power = np.median(spec[search_start:])
for idx in top_idx:
    period = N / idx
    ratio = spec[idx] / median_power
    flag = ' ★★★' if ratio > 10 else (' ★' if ratio > 5 else '')
    print(f"  {idx:>6} {period:>14.1f} {spec[idx]:>12.2e} {ratio:>15.2f}{flag}")

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Συγκεκριμένες "νότες" - οι αριθμοί ως MIDI
# Ποια "ζεύγη νοτών" (διαστήματα) εμφανίζονται μέσα στο ίδιο draw συχνότερα
# ═══════════════════════════════════════════════════════════════════
print("\n══ Experiment 5: Interval pair co-occurrence (a, a+k) for all k ══")
print("  How often does each interval k appear within any draw?")

interval_pair_counter = np.zeros(80, dtype=np.int64)
for ds in draw_sets:
    arr = sorted(ds)
    for i in range(20):
        for j in range(i+1, 20):
            interval_pair_counter[arr[j] - arr[i]] += 1

total_pairs = interval_pair_counter.sum()
print(f"  Total (i,j) pairs: {total_pairs:,}")
print(f"  Expected per draw: C(20,2) = 190")

print(f"\n  Pair interval distribution (top 15 by count):")
top_k = np.argsort(interval_pair_counter)[::-1][:15]
# Expected per interval k: number of (a,a+k) pairs in {1..80} × P(both in draw)
# = (80-k) × C(78,18)/C(80,20)
p_pair = comb(78, 18) / comb(80, 20)
print(f"  {'k':>3} {'Found':>10} {'Expected':>10} {'Ratio':>7} {'z':>8}")
for k in sorted(top_k):
    if k == 0: continue
    n_possible = 80 - k
    exp = N * n_possible * p_pair
    obs = interval_pair_counter[k]
    z = (obs - exp) / sqrt(exp)
    flag = ' ★' if abs(z) > 3 else ''
    print(f"  {k:>3} {obs:>10,} {exp:>10.0f} {obs/exp:>7.3f} {z:>+8.2f}{flag}")

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Πιο "μουσική" κλήρωση — αναζήτηση συγκεκριμένης χορδής
# Find the draw that contains the MOST consonant intervals (5ths, 4ths, octaves)
# ═══════════════════════════════════════════════════════════════════
print("\n══ Experiment 6: 'Most musical' draws (most consonant intervals) ══")
print("  Score = count of (a,b) pairs where |b-a| ∈ {5,7,12} (4th, 5th, octave)")

consonant_intervals = {5, 7, 12}
draw_scores = []
for did, nums in all_draws:
    score = 0
    for i in range(20):
        for j in range(i+1, 20):
            if nums[j] - nums[i] in consonant_intervals:
                score += 1
    draw_scores.append((score, did, nums))

draw_scores.sort(reverse=True)
print(f"\n  Top 5 most 'consonant' draws:")
for score, did, nums in draw_scores[:5]:
    print(f"    #{did} score={score}  nums={nums}")

bottom = sorted(draw_scores)[:5]
print(f"\n  5 least consonant draws:")
for score, did, nums in bottom[:5]:
    print(f"    #{did} score={score}  nums={nums}")

# Distribution
scores = [s for s,_,_ in draw_scores]
print(f"\n  Score stats: min={min(scores)} max={max(scores)} mean={np.mean(scores):.2f} std={np.std(scores):.2f}")

# Does high-consonance draw predict next draw structure?
print(f"\n  Test: do high-consonance draws → next draw with high overlap?")
high_thr = np.percentile(scores, 95)
low_thr = np.percentile(scores, 5)
high_next_overlap = []
low_next_overlap = []
draw_nums_set = [frozenset(n) for _, n in all_draws]
for i in range(N-1):
    s = scores[i]  # but scores was sorted! recompute

# Need to recompute scores in original order
print("  (Recomputing scores in draw order...)")
score_arr = np.zeros(N, dtype=np.int32)
for i, (_, nums) in enumerate(all_draws):
    s = 0
    for ii in range(20):
        for jj in range(ii+1, 20):
            if nums[jj] - nums[ii] in consonant_intervals:
                s += 1
    score_arr[i] = s

for i in range(N-1):
    s = score_arr[i]
    ov = len(draw_nums_set[i] & draw_nums_set[i+1])
    if s >= high_thr:
        high_next_overlap.append(ov)
    elif s <= low_thr:
        low_next_overlap.append(ov)

print(f"  High-consonance draws (≥{high_thr:.0f}): {len(high_next_overlap):,} draws, "
      f"mean next-overlap = {np.mean(high_next_overlap):.3f}")
print(f"  Low-consonance  draws (≤{low_thr:.0f}): {len(low_next_overlap):,} draws, "
      f"mean next-overlap = {np.mean(low_next_overlap):.3f}")
print(f"  Baseline: 5.000")

print("\n══ ΤΕΛΟΣ ΜΟΥΣΙΚΗΣ ΑΝΑΛΥΣΗΣ ══")
