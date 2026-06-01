#!/usr/bin/env python3
"""
ΕΠΙΠΛΕΟΝ ΣΗΜΑΤΑ — Prior: ΔΕΝ είναι τυχαίο.

Tests που δεν είχαμε κάνει:
 1. 2-back Markov: P(n at i+1 | n at i AND n at i-1)
 2. Long-range autocorr per number (lag 2..10)
 3. Conditional pair → number: top pairs from before
 4. High/low balance: count(n≤40) per draw + transitions
 5. Consecutive integer pairs per draw (e.g., 23,24)
 6. Minute-of-hour bias (within an hour)
 7. Spatial centroid (mean position of drawn numbers)
 8. Triplet co-occurrence in single draw
 9. Sum 5-draw moving avg autocorrelation
10. Recent-hot vs all-time: top 8 most-frequent in last 100 draws
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from math import sqrt, log, comb

DATA_DIR = Path('/home/user/Game/data/raw')

print("="*70); print("LOADING"); print("="*70)
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        all_draws.append((d['id'], sorted(d['n'])))
all_draws.sort(key=lambda x: x[0])
N = len(all_draws)
print(f"  {N:,} draws ({time.time()-t0:.1f}s)")

M = np.zeros((N, 81), dtype=np.int8)
for i, (_, nums) in enumerate(all_draws):
    for n in nums:
        M[i, n] = 1

ANCHOR_ID = 1303293
ANCHOR_DT = datetime(2026,5,31,23,55,tzinfo=timezone(timedelta(hours=3)))
def draw_time(idx):
    return ANCHOR_DT + timedelta(minutes=(all_draws[idx][0] - ANCHOR_ID) * 5.28)

p = 20/80
all_signals = []
def add_signal(cat, name, z, det):
    all_signals.append((cat, name, z, det))

# ═══════════════════════════════════════════════════════════════════
# TEST 1: 2-BACK MARKOV
# P(n at i+1 | n at i AND n at i-1) vs marginal
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] 2-BACK MARKOV: P(n at i+1 | n at i AND n at i-1)")
print("="*70)
t0 = time.time()
results_2back = []
for n in range(1, 81):
    in_i_minus_1 = M[:-2, n] == 1
    in_i = M[1:-1, n] == 1
    in_i_plus_1 = M[2:, n] == 1
    cond = in_i_minus_1 & in_i
    cnt_cond = int(cond.sum())
    if cnt_cond < 200: continue
    cnt_continue = int((cond & in_i_plus_1).sum())
    p_cont = cnt_continue / cnt_cond
    z = (p_cont - p) / sqrt(p*(1-p)/cnt_cond)
    results_2back.append((n, cnt_cond, p_cont, z))
results_2back.sort(key=lambda x: -abs(x[3]))
print(f"  Top 10 numbers with 2-back continuation bias  ({time.time()-t0:.1f}s)")
for n, c, pc, z in results_2back[:10]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    #{n:>2}: P(n_{{i+1}} | n_i, n_{{i-1}}) = {pc:.4f}  n={c:,}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("2back", f"n{n}", z, f"#{n} 2-back continuation")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: LONG-RANGE AUTOCORR per number (lag 2..10)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] LONG-RANGE AUTOCORR per number (lag 2..10)")
print("="*70)
t0 = time.time()
strongest_lag = []
for n in range(1, 81):
    arr = M[:, n].astype(np.float64)
    arr = arr - arr.mean()
    var = np.dot(arr, arr) / N
    if var == 0: continue
    for lag in range(2, 11):
        n_pts = N - lag
        cov = np.dot(arr[:-lag], arr[lag:]) / n_pts
        corr = cov / var
        z = corr * sqrt(n_pts)
        if abs(z) > 2.5:
            strongest_lag.append((n, lag, corr, z))
strongest_lag.sort(key=lambda x: -abs(x[3]))
print(f"  Found {len(strongest_lag)} (number, lag) pairs with |z|>2.5  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for n, lag, c, z in strongest_lag[:15]:
    flag = " ★" if abs(z) > 3 else ""
    print(f"    #{n:>2}  lag={lag:>2}  acf={c:+.6f}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("lag", f"n{n}@lag{lag}", z, f"#{n} lag {lag}")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: CONDITIONAL PAIR → NUMBER
# If pair (a,b) in draw_i, what's P(c in draw_{i+1})?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] CONDITIONAL PAIR → NEXT NUMBER")
print("="*70)
t0 = time.time()
# Take top 20 strongest pairs from before (by frequency deviation), test each.
# Need pair occurrences first
pair_idx = defaultdict(list)
for i in range(N-1):
    nums = all_draws[i][1]
    for a_ in range(20):
        for b_ in range(a_+1, 20):
            pair_idx[(nums[a_], nums[b_])].append(i)
print(f"  Indexed {len(pair_idx)} pairs  ({time.time()-t0:.1f}s)")

# Test only most frequent pairs (those with enough samples)
top_pairs_to_test = [(p, len(idx)) for p, idx in pair_idx.items() if len(idx) > 10000]
top_pairs_to_test.sort(key=lambda x: -x[1])
print(f"  Testing {len(top_pairs_to_test)} pairs with >10K occurrences")
# For each such pair, test all 80 candidate next-numbers
cond_pair_signals = []
for (a, b), idx_list in pair_idx.items():
    if len(idx_list) < 12000: continue
    # Next-draw matrix for these indices
    next_indices = np.array(idx_list, dtype=np.int32)
    next_indices = next_indices[next_indices < N-1]
    if len(next_indices) < 12000: continue
    next_M = M[next_indices + 1]  # (count, 81)
    counts_next = next_M.sum(axis=0)
    for c_ in range(1, 81):
        if c_ in (a, b): continue
        cnt = int(counts_next[c_])
        exp = len(next_indices) * p
        var = len(next_indices) * p * (1-p)
        z = (cnt - exp) / sqrt(var)
        if abs(z) > 3.5:
            cond_pair_signals.append(((a,b), c_, cnt, len(next_indices), z))
cond_pair_signals.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(cond_pair_signals)} pair→number signals with |z|>3.5")
for (a,b), c_, cnt, total, z in cond_pair_signals[:15]:
    pc = cnt/total
    print(f"    ({a:>2},{b:>2}) → {c_:>2}  P={pc:.4f}  n={total}  z={z:+.2f}")
    if abs(z) > 4.0:
        add_signal("p2n", f"({a},{b})→{c_}", z, f"pair ({a},{b}) → #{c_}")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: HIGH/LOW BALANCE per draw
# count(n ≤ 40) per draw; expected = 10
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4] HIGH/LOW BALANCE: count(n≤40) per draw")
print("="*70)
t0 = time.time()
low_counts = np.array([sum(1 for n in d if n <= 40) for _, d in all_draws])
# Expected: 10, variance = hypergeometric
exp_low = 10.0
exp_var = 20 * (40/80) * (40/80) * (60/79)
z = (low_counts.mean() - exp_low) / (sqrt(exp_var/N))
print(f"  Mean low count: {low_counts.mean():.4f} (exp 10.0, z={z:+.2f})")
# Markov on low count: transition matrix
binned = np.clip(low_counts, 3, 17)  # bin 3..17
trans_lc = np.zeros((15, 15), dtype=np.int32)
for i in range(N-1):
    trans_lc[binned[i]-3, binned[i+1]-3] += 1
total_lc = trans_lc.sum()
row_lc = trans_lc.sum(axis=1)
col_lc = trans_lc.sum(axis=0)
strongest_lc = []
for i in range(15):
    for j in range(15):
        exp = row_lc[i] * col_lc[j] / total_lc
        if exp > 100:
            z = (trans_lc[i,j] - exp) / sqrt(exp)
            if abs(z) > 3:
                strongest_lc.append((i+3, j+3, trans_lc[i,j], z))
print(f"  Strongest low-count transitions:  ({time.time()-t0:.1f}s)")
for i, j, c, z in sorted(strongest_lc, key=lambda x: -abs(x[3]))[:5]:
    print(f"    low {i} → low {j}: {c} (z={z:+.2f})")
    if abs(z) > 3.5:
        add_signal("low_trans", f"{i}→{j}", z, f"low count {i}→{j}")

# ═══════════════════════════════════════════════════════════════════
# TEST 5: CONSECUTIVE INTEGER PAIRS per draw
# Count (i, i+1) both in draw. Expected ≈ 4.75
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[5] CONSECUTIVE INTEGER PAIRS per draw (e.g., 23+24)")
print("="*70)
t0 = time.time()
consec_counts = np.zeros(N, dtype=np.int8)
for i, (_, nums) in enumerate(all_draws):
    s = set(nums)
    consec_counts[i] = sum(1 for n in nums if (n+1) in s)
# Expected: 79 possible consecutive pairs, P(both in draw) = 20*19/(80*79)
exp_consec = 79 * (20*19)/(80*79)  # ≈ 4.75
# Variance: complicated, use empirical baseline
# Simpler: just compare mean with std/sqrt(N)
print(f"  Mean: {consec_counts.mean():.4f} (exp ≈ {exp_consec:.2f})")
print(f"  Std:  {consec_counts.std():.4f}")
# Test distribution vs expected (via Monte Carlo for variance)
import random as _r
_r.seed(0)
mc_samples = []
for _ in range(20000):
    s = sorted(_r.sample(range(1,81), 20))
    s_set = set(s)
    mc_samples.append(sum(1 for x in s if (x+1) in s_set))
mc_mean = np.mean(mc_samples)
mc_std = np.std(mc_samples)
z = (consec_counts.mean() - mc_mean) / (mc_std / sqrt(N))
print(f"  Monte Carlo baseline: mean={mc_mean:.4f}  std={mc_std:.4f}")
print(f"  z-score vs MC: {z:+.2f}  ({time.time()-t0:.1f}s)")
if abs(z) > 2:
    add_signal("consec", "mean", z, "consecutive int pairs")

# Markov on consec count
trans_cc = np.zeros((20, 20), dtype=np.int32)
for i in range(N-1):
    a = min(consec_counts[i], 19); b = min(consec_counts[i+1], 19)
    trans_cc[a, b] += 1
total_cc = trans_cc.sum()
row_cc = trans_cc.sum(axis=1)
col_cc = trans_cc.sum(axis=0)
strongest_cc = []
for i in range(20):
    for j in range(20):
        exp = row_cc[i] * col_cc[j] / total_cc
        if exp > 50:
            z = (trans_cc[i,j] - exp) / sqrt(exp)
            if abs(z) > 3:
                strongest_cc.append((i, j, trans_cc[i,j], z))
strongest_cc.sort(key=lambda x: -abs(x[3]))
print(f"  Strongest consec count transitions:")
for i, j, c, z in strongest_cc[:5]:
    print(f"    {i} → {j}: {c} (z={z:+.2f})")
    if abs(z) > 3.5:
        add_signal("consec_trans", f"{i}→{j}", z, f"consec {i}→{j}")

# ═══════════════════════════════════════════════════════════════════
# TEST 6: MINUTE-OF-HOUR BIAS
# Each draw lands on a specific minute mod 60.
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] MINUTE-OF-HOUR BIAS")
print("="*70)
t0 = time.time()
# Each draw is at minute = (id * 5.28) mod 60, anchored
minute_counts = np.zeros((60, 81), dtype=np.int32)
minute_totals = np.zeros(60, dtype=np.int32)
for i in range(N):
    dt = draw_time(i)
    minute = dt.minute  # 0-59
    minute_totals[minute] += 1
    for n in all_draws[i][1]:
        minute_counts[minute, n] += 1

top_minute = []
for m_ in range(60):
    T = minute_totals[m_]
    if T < 1000: continue
    for n in range(1, 81):
        exp = T * p
        var = T * p * (1-p)
        z = (minute_counts[m_, n] - exp) / sqrt(var)
        if abs(z) > 3.5:
            top_minute.append((n, m_, z))
top_minute.sort(key=lambda x: -abs(x[2]))
print(f"  Found {len(top_minute)} (number, minute) cells with |z|>3.5  ({time.time()-t0:.1f}s)")
print(f"  Top 10:")
for n, m_, z in top_minute[:10]:
    print(f"    #{n:>2}  minute={m_:>2}  z={z:+.2f}")
    add_signal("minute", f"n{n}@m{m_}", z, f"#{n} at minute {m_}")

# ═══════════════════════════════════════════════════════════════════
# TEST 7: SPATIAL CENTROID per draw
# Mean row & mean col of drawn numbers
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[7] SPATIAL CENTROID per draw")
print("="*70)
t0 = time.time()
rows_arr = np.zeros(N, dtype=np.float32)
cols_arr = np.zeros(N, dtype=np.float32)
for i, (_, nums) in enumerate(all_draws):
    rows = [(n-1)//10 for n in nums]
    cols = [(n-1)%10 for n in nums]
    rows_arr[i] = np.mean(rows)
    cols_arr[i] = np.mean(cols)
exp_row_mean = 3.5  # mean of 0..7
exp_col_mean = 4.5  # mean of 0..9
z_row = (rows_arr.mean() - exp_row_mean) / (rows_arr.std()/sqrt(N))
z_col = (cols_arr.mean() - exp_col_mean) / (cols_arr.std()/sqrt(N))
print(f"  Row centroid: mean={rows_arr.mean():.4f} (exp 3.5, z={z_row:+.2f})")
print(f"  Col centroid: mean={cols_arr.mean():.4f} (exp 4.5, z={z_col:+.2f})")
# Autocorr of centroid (does centroid drift)
for name, arr in [('row', rows_arr), ('col', cols_arr)]:
    a = arr - arr.mean()
    for lag in [1, 2, 5, 10]:
        cov = np.dot(a[:-lag], a[lag:]) / (N-lag)
        var = np.dot(a, a) / N
        c = cov / var
        z = c * sqrt(N-lag)
        flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
        print(f"  {name} centroid autocorr lag={lag}: {c:+.6f} z={z:+.2f}{flag}")
        if abs(z) > 2.5:
            add_signal("centroid", f"{name}_lag{lag}", z, f"{name} centroid lag {lag}")
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 8: TRIPLET CO-OCCURRENCE in single draw
# Top triplets vs expected
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[8] TRIPLET CO-OCCURRENCE (top 20 most-frequent)")
print("="*70)
t0 = time.time()
triplet_counts = defaultdict(int)
for _, nums in all_draws:
    for i in range(20):
        for j in range(i+1, 20):
            for k in range(j+1, 20):
                triplet_counts[(nums[i], nums[j], nums[k])] += 1
# Expected per triplet: N * C(20,3)/C(80,3) ≈ N * 0.01443
n_triplets = comb(80, 3)  # 82,160
exp_triplet = N * (20 * 19 * 18) / (80 * 79 * 78)
var_triplet_approx = exp_triplet * (1 - exp_triplet/N)
print(f"  Expected per triplet: {exp_triplet:.2f}  std≈{sqrt(var_triplet_approx):.2f}  ({time.time()-t0:.1f}s)")
top_triplet = [(t, c, (c-exp_triplet)/sqrt(var_triplet_approx)) for t, c in triplet_counts.items()]
top_triplet.sort(key=lambda x: -abs(x[2]))
print(f"  Top 15 by |z|:")
for t, c, z in top_triplet[:15]:
    flag = " ★" if abs(z) > 4 else ""
    print(f"    {t}: count={c:>5}  z={z:+.2f}{flag}")
    if abs(z) > 4:
        add_signal("triplet", str(t), z, f"triplet {t}")

# ═══════════════════════════════════════════════════════════════════
# TEST 9: SUM 5-DRAW MOVING AVERAGE
# Does the smoothed sum series have memory?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[9] SUM 5-DRAW MOVING AVG AUTOCORR")
print("="*70)
t0 = time.time()
sums_arr = np.array([sum(d) for _, d in all_draws], dtype=np.float64)
W = 5
ma = np.convolve(sums_arr, np.ones(W)/W, mode='valid')
ma_centered = ma - ma.mean()
var = np.dot(ma_centered, ma_centered) / len(ma_centered)
print(f"  Moving avg autocorr (after smoothing W={W}):")
for lag in [1, 2, 5, 10, 50, 100, 272]:
    if lag >= len(ma): continue
    cov = np.dot(ma_centered[:-lag], ma_centered[lag:]) / (len(ma_centered)-lag)
    c = cov / var
    z = c * sqrt(len(ma_centered)-lag)
    # Correct for the smoothing (correlation introduced by overlap)
    flag = " ★" if abs(z) > 5 else ""  # higher threshold due to overlap
    print(f"    lag={lag:>4}: acf={c:+.6f}  z={z:+.2f}{flag}")

# Direct sum series autocorrelation (no smoothing)
sums_c = sums_arr - sums_arr.mean()
var = np.dot(sums_c, sums_c) / N
print(f"  Raw sum autocorr:")
for lag in [1, 2, 5, 10, 100, 272, 1904]:
    if lag >= N: continue
    cov = np.dot(sums_c[:-lag], sums_c[lag:]) / (N-lag)
    c = cov / var
    z = c * sqrt(N-lag)
    flag = " ★" if abs(z) > 3 else ""
    print(f"    lag={lag:>5}: acf={c:+.6f}  z={z:+.2f}{flag}")
    if abs(z) > 3:
        add_signal("sum_acf", f"lag{lag}", z, f"sum autocorr lag {lag}")
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 10: RECENT-HOT TOP-8 vs ALL-TIME
# Do the top 8 most-frequent numbers in last 100 draws predict next?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[10] RECENT-HOT TOP-8 vs ALL-TIME")
print("="*70)
t0 = time.time()
W = 100
# For each i >= W, count appearances in last W draws per number
# Check: is the top-8 'hot' set more likely to appear at i?
total_hot_in_next = 0
total_hot_baseline = 0
n_tests = 0
for i in range(W, N):
    last_W = M[i-W:i]
    freq = last_W.sum(axis=0)  # 81
    # Top 8 (excluding 0)
    top8 = np.argsort(-freq[1:])[:8] + 1
    in_next = sum(1 for n in top8 if M[i, n])
    total_hot_in_next += in_next
    n_tests += 1
exp_hot_hits = n_tests * 8 * p
var_hot = n_tests * 8 * p * (1-p)
z = (total_hot_in_next - exp_hot_hits) / sqrt(var_hot)
print(f"  Tests: {n_tests:,}")
print(f"  Hot-top-8 hits: {total_hot_in_next:,} (exp {exp_hot_hits:,.0f})")
print(f"  Avg hot/draw: {total_hot_in_next/n_tests:.4f} (exp 2.0)")
print(f"  z = {z:+.2f}  ({time.time()-t0:.1f}s)")
if abs(z) > 2:
    add_signal("hot_top8", "all", z, f"hot top-8 in last {W} predictive")

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ — {len(all_signals)} νέα σήματα")
print("="*70)
all_signals.sort(key=lambda x: -abs(x[2]))
print(f"\n  Top 30 (sorted by |z|):")
for i, (cat, name, z, det) in enumerate(all_signals[:30], 1):
    print(f"  {i:>3}.  [{cat:>12}]  {name:>18}  |z|={abs(z):>6.2f}  {'+' if z>0 else '-'}  {det}")

print(f"\n  Σύνοψη ανά κατηγορία:")
cats = defaultdict(list)
for cat, name, z, det in all_signals:
    cats[cat].append(abs(z))
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"    {cat:>14}: {len(arr):>3} σήματα, max |z|={max(arr):.2f}, mean={np.mean(arr):.2f}")

out = {'signals': [(c, n, float(z), d) for c, n, z, d in all_signals], 'N': N}
with open('/home/user/Game/more_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: more_signals.json ({len(all_signals)} σήματα)")
print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
