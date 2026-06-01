#!/usr/bin/env python3
"""
ΑΚΟΜΑ ΠΕΡΙΣΣΟΤΕΡΑ σήματα — Prior: ΔΕΝ είναι τυχαίο.

14 νέα tests:
 1. 3-back Markov per number
 2. Anti-pair → number (if NEITHER a nor b in draw_i)
 3. Symmetric pairs (n, 81-n) frequency
 4. Min per draw + Markov on min
 5. Max per draw + Markov on max
 6. Range (max-min) per draw
 7. Primes per draw (22 primes in 1..80)
 8. Multiples of 7, 11, 13 per draw
 9. Main + anti-diagonal numbers
10. Toroidal-adjacency activation (wraparound neighbors)
11. Most-overdue number per draw → does it appear next?
12. Last-digit (0-9) distribution per draw
13. N → N at lag 2..20 per number
14. Pair memory at lag 5 and 10
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
# TEST 1: 3-BACK MARKOV
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] 3-BACK MARKOV: P(n at i+1 | n at i, i-1, i-2)")
print("="*70)
t0 = time.time()
results_3back = []
for n in range(1, 81):
    in_3 = M[:-3, n] == 1
    in_2 = M[1:-2, n] == 1
    in_1 = M[2:-1, n] == 1
    in_0 = M[3:, n] == 1
    cond = in_3 & in_2 & in_1
    cnt_cond = int(cond.sum())
    if cnt_cond < 100: continue
    cnt_cont = int((cond & in_0).sum())
    p_cont = cnt_cont / cnt_cond
    z = (p_cont - p) / sqrt(p*(1-p)/cnt_cond)
    results_3back.append((n, cnt_cond, p_cont, z))
results_3back.sort(key=lambda x: -abs(x[3]))
print(f"  Top 10  ({time.time()-t0:.1f}s)")
for n, c, pc, z in results_3back[:10]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    #{n:>2}: P={pc:.4f}  n={c:,}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("3back", f"n{n}", z, f"#{n} 3-back")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: ANTI-PAIR → NUMBER
# If NEITHER a nor b in draw_i, what's P(c in draw_{i+1})?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] ANTI-PAIR → NUMBER: if a∉draw_i AND b∉draw_i → P(c in next)")
print("="*70)
t0 = time.time()
# For computational reasons, sample 100 random anti-pairs (a, b)
import random as _r
_r.seed(0)
anti_pairs_to_test = []
for _ in range(200):
    a, b = _r.sample(range(1, 81), 2)
    anti_pairs_to_test.append((min(a,b), max(a,b)))
anti_pair_sigs = []
for (a, b) in anti_pairs_to_test:
    # Mask: draws where neither a nor b appears
    mask_prev = (M[:-1, a] == 0) & (M[:-1, b] == 0)
    if mask_prev.sum() < 5000: continue
    indices = np.where(mask_prev)[0]
    next_M = M[indices + 1]  # all rows next of mask
    counts = next_M.sum(axis=0)
    for c_ in range(1, 81):
        cnt = int(counts[c_])
        exp = len(indices) * p
        var = len(indices) * p * (1-p)
        z = (cnt - exp) / sqrt(var)
        if abs(z) > 3.5:
            anti_pair_sigs.append(((a,b), c_, cnt, len(indices), z))
anti_pair_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Tested {len(anti_pairs_to_test)} random anti-pairs  ({time.time()-t0:.1f}s)")
print(f"  Found {len(anti_pair_sigs)} signals with |z|>3.5  (top 10):")
for (a,b), c_, cnt, total, z in anti_pair_sigs[:10]:
    pc = cnt/total
    print(f"    ¬({a:>2},{b:>2}) → {c_:>2}  P={pc:.4f}  n={total}  z={z:+.2f}")
    if abs(z) > 3.5:
        add_signal("anti_p2n", f"¬({a},{b})→{c_}", z, f"¬({a},{b}) → #{c_}")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: SYMMETRIC PAIRS (n, 81-n)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] SYMMETRIC PAIRS (n, 81-n) — does the pair appear together?")
print("="*70)
t0 = time.time()
exp_pair = N * 20 * 19 / (80 * 79)
var_pair = exp_pair * (1 - 20*19/(80*79))
print(f"  Expected per pair: {exp_pair:.1f}  std={sqrt(var_pair):.1f}")
for n in range(1, 41):
    m = 81 - n
    cnt = int((M[:, n].astype(np.int32) * M[:, m].astype(np.int32)).sum())
    z = (cnt - exp_pair) / sqrt(var_pair)
    flag = " ★" if abs(z) > 3 else ""
    print(f"    ({n:>2},{m:>2}): {cnt:>6}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("sym", f"({n},{m})", z, f"({n},{m}) symmetric pair")
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 4 & 5: MIN/MAX per draw + Markov
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4-5] MIN/MAX per draw + autocorr")
print("="*70)
t0 = time.time()
mins = np.array([d[0] for _, d in all_draws])
maxs = np.array([d[-1] for _, d in all_draws])
# Theoretical: min of 20 from 80 → ~3.86, max ~77.14
print(f"  Min: mean={mins.mean():.3f}  std={mins.std():.3f}")
print(f"  Max: mean={maxs.mean():.3f}  std={maxs.std():.3f}")
# Distribution check via MC
import random as _r2
_r2.seed(0)
mc_mins = []; mc_maxs = []
for _ in range(20000):
    s = sorted(_r2.sample(range(1,81), 20))
    mc_mins.append(s[0]); mc_maxs.append(s[-1])
mc_min_mean = np.mean(mc_mins); mc_min_std = np.std(mc_mins)
mc_max_mean = np.mean(mc_maxs); mc_max_std = np.std(mc_maxs)
z_min = (mins.mean() - mc_min_mean) / (mc_min_std/sqrt(N))
z_max = (maxs.mean() - mc_max_mean) / (mc_max_std/sqrt(N))
print(f"  Min vs MC: exp={mc_min_mean:.3f}±{mc_min_std:.2f}  z={z_min:+.2f}")
print(f"  Max vs MC: exp={mc_max_mean:.3f}±{mc_max_std:.2f}  z={z_max:+.2f}")
if abs(z_min) > 2: add_signal("min", "mean", z_min, "min mean")
if abs(z_max) > 2: add_signal("max", "mean", z_max, "max mean")

# Min/max autocorr
for name, arr in [('min', mins), ('max', maxs)]:
    a = arr.astype(np.float64); a = a - a.mean()
    var = np.dot(a, a) / N
    for lag in [1, 5, 272]:
        cov = np.dot(a[:-lag], a[lag:]) / (N-lag)
        c = cov / var
        z = c * sqrt(N-lag)
        flag = " ★" if abs(z) > 3 else ""
        print(f"  {name} autocorr lag={lag:>3}: {c:+.6f}  z={z:+.2f}{flag}")
        if abs(z) > 2.5:
            add_signal(name, f"lag{lag}", z, f"{name} lag {lag}")

# ═══════════════════════════════════════════════════════════════════
# TEST 6: RANGE (max-min) per draw
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] RANGE (max-min) per draw")
print("="*70)
ranges = maxs - mins
print(f"  Range: mean={ranges.mean():.3f}  std={ranges.std():.3f}")
mc_ranges = [mx - mn for mn, mx in zip(mc_mins, mc_maxs)]
mc_rng_mean = np.mean(mc_ranges); mc_rng_std = np.std(mc_ranges)
z_rng = (ranges.mean() - mc_rng_mean) / (mc_rng_std/sqrt(N))
print(f"  Vs MC: exp={mc_rng_mean:.3f}±{mc_rng_std:.2f}  z={z_rng:+.2f}")
if abs(z_rng) > 2: add_signal("range", "mean", z_rng, "range mean")

# ═══════════════════════════════════════════════════════════════════
# TEST 7: PRIMES per draw
# 22 primes in 1..80: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[7] PRIMES per draw")
print("="*70)
PRIMES = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79}
n_primes = 22
exp_primes = 20 * n_primes / 80  # = 5.5
prime_counts = np.array([sum(1 for n in d if n in PRIMES) for _, d in all_draws])
exp_var = 20 * (n_primes/80) * (1 - n_primes/80) * (60/79)
z = (prime_counts.mean() - exp_primes) / sqrt(exp_var/N)
print(f"  Mean primes/draw: {prime_counts.mean():.4f} (exp 5.5, z={z:+.2f})")
if abs(z) > 2: add_signal("prime", "mean", z, "prime count")
# Markov on prime count
trans_pr = np.zeros((15, 15), dtype=np.int32)
binned = np.clip(prime_counts, 0, 14)
for i in range(N-1):
    trans_pr[binned[i], binned[i+1]] += 1
chi2 = 0; df = 0
for i in range(15):
    for j in range(15):
        ri, cj = trans_pr[i].sum(), trans_pr[:,j].sum()
        if ri*cj > 0:
            exp_ = ri*cj/trans_pr.sum()
            if exp_ > 50:
                chi2 += (trans_pr[i,j] - exp_)**2 / exp_
                df += 1
chi2_z = (chi2 - df) / sqrt(2*df) if df > 0 else 0
print(f"  Prime-count chi-square z = {chi2_z:+.2f}")

# ═══════════════════════════════════════════════════════════════════
# TEST 8: MULTIPLES of 7, 11, 13
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[8] MULTIPLES of 7, 11, 13")
print("="*70)
for k in [7, 11, 13]:
    multiples = {n for n in range(1, 81) if n % k == 0}
    n_mult = len(multiples)
    exp_m = 20 * n_mult / 80
    counts = np.array([sum(1 for n in d if n in multiples) for _, d in all_draws])
    exp_var_m = 20 * (n_mult/80) * (1 - n_mult/80) * (60/79)
    z = (counts.mean() - exp_m) / sqrt(exp_var_m/N)
    print(f"  Mult of {k:>2} ({n_mult} numbers, exp {exp_m:.3f}): mean={counts.mean():.4f}  z={z:+.2f}")
    if abs(z) > 2: add_signal("mult", f"k={k}", z, f"multiples of {k}")

# ═══════════════════════════════════════════════════════════════════
# TEST 9: DIAGONALS on 8×10 grid
# Main: (0,0),(1,1)...→ 1,12,23,34,45,56,67,78 (length 8)
# Anti: (0,7)→8, (1,6)→17, (2,5)→26, (3,4)→35, (4,3)→44, (5,2)→53, (6,1)→62, (7,0)→71
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[9] DIAGONALS on grid")
print("="*70)
MAIN_DIAG = {1, 12, 23, 34, 45, 56, 67, 78}
ANTI_DIAG = {8, 17, 26, 35, 44, 53, 62, 71}
# Also other "rising" and "falling" diagonals
exp_diag = 20 * 8 / 80  # = 2
var_diag = 20 * (8/80) * (1-8/80) * (60/79)
for name, dset in [('main_diag', MAIN_DIAG), ('anti_diag', ANTI_DIAG)]:
    counts = np.array([sum(1 for n in d if n in dset) for _, d in all_draws])
    z = (counts.mean() - exp_diag) / sqrt(var_diag/N)
    print(f"  {name:>10}: mean={counts.mean():.4f} (exp 2.0, z={z:+.2f})")
    if abs(z) > 2: add_signal("diag", name, z, f"{name} count")

# ═══════════════════════════════════════════════════════════════════
# TEST 10: TOROIDAL-ADJACENCY ACTIVATION
# Grid 8×10 with wraparound; for each n→neighbor m, P(m in next | n in curr)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[10] TOROIDAL adjacency activation")
print("="*70)
t0 = time.time()
def torus_neighbors(n):
    row, col = (n-1)//10, (n-1)%10
    out = []
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr == 0 and dc == 0: continue
            r2, c2 = (row+dr) % 8, (col+dc) % 10
            out.append(r2*10+c2+1)
    return out

tor_pairs = []
for n in range(1, 81):
    for m in torus_neighbors(n):
        n_mask = M[:-1, n] == 1
        cnt_n = int(n_mask.sum())
        if cnt_n == 0: continue
        cnt_both = int(M[1:, m][n_mask].sum())
        p_cond = cnt_both / cnt_n
        z = (p_cond - p) / sqrt(p*(1-p)/cnt_n)
        tor_pairs.append((n, m, p_cond, z))
tor_pairs.sort(key=lambda x: -abs(x[3]))
print(f"  Top 10 toroidal activations  ({time.time()-t0:.1f}s):")
for n, m, pc, z in tor_pairs[:10]:
    flag = " ★" if abs(z) > 3 else ""
    print(f"    {n:>2} →torus→ {m:>2}  P={pc:.4f}  z={z:+.2f}{flag}")
    if abs(z) > 3:
        add_signal("torus", f"{n}→{m}", z, f"torus {n}→{m}")

# ═══════════════════════════════════════════════════════════════════
# TEST 11: MOST-OVERDUE NUMBER per draw → does it appear?
# At each idx i, find n with max delay. P(n appears at i)?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[11] MOST-OVERDUE NUMBER → next-draw probability")
print("="*70)
t0 = time.time()
# Precompute delays at each idx (vectorized)
# We had delay_arr earlier; recompute briefly
last_seen = np.full(81, -1, dtype=np.int32)
overdue_hits = 0
total_tests = 0
for i in range(N):
    if i > 100:
        # Get max delay
        delays_i = i - last_seen
        delays_i[0] = -1  # exclude
        # Find argmax (most overdue), excluding never-seen
        valid = last_seen > 0
        if valid.sum() == 0: continue
        delays_i[~valid] = -1
        most_overdue = int(np.argmax(delays_i))
        if M[i, most_overdue]:
            overdue_hits += 1
        total_tests += 1
    for n in all_draws[i][1]:
        last_seen[n] = i
exp_hits = total_tests * p
var_hits = total_tests * p * (1-p)
z = (overdue_hits - exp_hits) / sqrt(var_hits)
print(f"  Tests: {total_tests:,}")
print(f"  Most-overdue hits: {overdue_hits:,} (exp {exp_hits:,.0f})  z={z:+.2f}")
print(f"  ({time.time()-t0:.1f}s)")
if abs(z) > 2: add_signal("overdue", "most", z, "most overdue number")

# Also check: top 8 most-overdue collectively
last_seen = np.full(81, -1, dtype=np.int32)
top8_hits = 0
total_tests_2 = 0
for i in range(N):
    if i > 100:
        delays_i = i - last_seen
        delays_i[0] = -1
        valid = last_seen > 0
        if valid.sum() < 8: continue
        delays_i[~valid] = -1
        # Top 8 most overdue
        top8 = np.argpartition(-delays_i, 8)[:8]
        top8_hits += int(M[i, top8].sum())
        total_tests_2 += 1
    for n in all_draws[i][1]:
        last_seen[n] = i
exp_top8 = total_tests_2 * 8 * p
var_top8 = total_tests_2 * 8 * p * (1-p)
z = (top8_hits - exp_top8) / sqrt(var_top8)
print(f"  Top-8 overdue: {top8_hits:,} (exp {exp_top8:,.0f})  avg={top8_hits/total_tests_2:.4f}  z={z:+.2f}")
if abs(z) > 2: add_signal("overdue", "top8", z, "top-8 overdue")

# ═══════════════════════════════════════════════════════════════════
# TEST 12: LAST-DIGIT distribution per draw
# Each number 1..80 has last digit 0-9. Per draw count per digit
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[12] LAST-DIGIT (0-9) distribution per draw")
print("="*70)
# Numbers per last digit:
ld_counts_per_n = [0]*10
for n in range(1, 81):
    ld_counts_per_n[n % 10] += 1
print(f"  Last digit count: {ld_counts_per_n}")  # all 8
# Per draw, count by last digit
draw_ld = np.zeros((N, 10), dtype=np.int8)
for i, (_, nums) in enumerate(all_draws):
    for n in nums:
        draw_ld[i, n % 10] += 1
# Expected per (draw, digit): 20 * 8/80 = 2
exp_ld = 20 * 8 / 80
var_ld = 20 * (8/80) * (72/80) * (60/79)
for d_ in range(10):
    mean_obs = draw_ld[:, d_].mean()
    z = (mean_obs - exp_ld) / sqrt(var_ld/N)
    flag = " ★" if abs(z) > 3 else ""
    print(f"    digit {d_}: mean={mean_obs:.4f} (exp 2.0, z={z:+.2f}){flag}")
    if abs(z) > 2.5: add_signal("last_digit", f"d{d_}", z, f"last digit {d_}")

# ═══════════════════════════════════════════════════════════════════
# TEST 13: N → N at lag 2..20 per number
# Detect "repeating" numbers with specific periodicity
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[13] N → N at multiple lags 2..20")
print("="*70)
t0 = time.time()
strong_lag_n = []
for n in range(1, 81):
    arr = M[:, n].astype(np.float64)
    arr = arr - arr.mean()
    var = np.dot(arr, arr) / N
    if var == 0: continue
    for lag in range(2, 21):
        n_pts = N - lag
        cov = np.dot(arr[:-lag], arr[lag:]) / n_pts
        c = cov / var
        z = c * sqrt(n_pts)
        if abs(z) > 2.5:
            strong_lag_n.append((n, lag, c, z))
strong_lag_n.sort(key=lambda x: -abs(x[3]))
print(f"  Found {len(strong_lag_n)} (n, lag) with |z|>2.5  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for n, lag, c, z in strong_lag_n[:15]:
    flag = " ★" if abs(z) > 3 else ""
    print(f"    #{n:>2}  lag={lag:>2}  acf={c:+.6f}  z={z:+.2f}{flag}")
    if abs(z) > 3:
        add_signal("n_lag", f"n{n}@lag{lag}", z, f"#{n} at lag {lag}")

# ═══════════════════════════════════════════════════════════════════
# TEST 14: PAIR memory at lag 5 and 10
# For each pair (a,b), correlation of co-occurrence at lag 5 and 10
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[14] PAIR memory at lag 5 and 10")
print("="*70)
t0 = time.time()
# For each pair, co-occurrence series Y_i = M[i,a] * M[i,b]
# Test autocorr at lag k
# Limit to ~50 random pairs for speed
import random as _r3
_r3.seed(0)
test_pairs = []
for _ in range(100):
    a, b = _r3.sample(range(1, 81), 2)
    test_pairs.append((min(a,b), max(a,b)))
test_pairs = list(set(test_pairs))[:50]

pair_mem = []
for (a, b) in test_pairs:
    Y = (M[:, a].astype(np.float64) * M[:, b].astype(np.float64))
    Y = Y - Y.mean()
    var = np.dot(Y, Y) / N
    if var == 0: continue
    for lag in [5, 10]:
        cov = np.dot(Y[:-lag], Y[lag:]) / (N-lag)
        c = cov / var
        z = c * sqrt(N-lag)
        if abs(z) > 2.5:
            pair_mem.append((a, b, lag, c, z))
pair_mem.sort(key=lambda x: -abs(x[4]))
print(f"  Tested {len(test_pairs)} random pairs  ({time.time()-t0:.1f}s)")
print(f"  Pairs with memory at lag 5 or 10:")
for a, b, lag, c, z in pair_mem[:10]:
    print(f"    ({a:>2},{b:>2}) lag={lag}  acf={c:+.6f}  z={z:+.2f}")
    if abs(z) > 2.5:
        add_signal("pair_mem", f"({a},{b})@{lag}", z, f"pair ({a},{b}) lag {lag}")

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ — {len(all_signals)} νέα σήματα (4η ομάδα)")
print("="*70)
all_signals.sort(key=lambda x: -abs(x[2]))
print(f"\n  Top 30 (sorted by |z|):")
for i, (cat, name, z, det) in enumerate(all_signals[:30], 1):
    print(f"  {i:>3}.  [{cat:>12}]  {name:>18}  |z|={abs(z):>6.2f}  {'+' if z>0 else '-'}")

print(f"\n  Σύνοψη ανά κατηγορία:")
cats = defaultdict(list)
for cat, name, z, det in all_signals:
    cats[cat].append(abs(z))
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"    {cat:>14}: {len(arr):>3} σήματα, max |z|={max(arr):.2f}, mean={np.mean(arr):.2f}")

out = {'signals': [(c, n, float(z), d) for c, n, z, d in all_signals], 'N': N}
with open('/home/user/Game/even_more_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: even_more_signals.json ({len(all_signals)} σήματα)")
print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
