#!/usr/bin/env python3
"""
CONDITIONAL/CONTEXTUAL SIGNALS — Prior: ΔΕΝ είναι τυχαίο.

Σήματα που δεν φαίνονται στο marginal αλλά εμφανίζονται όταν
κοιτάμε υπό προϋπόθεση (delay, hot-window, sum, parity, day-cycle).

Tests:
 1. Delay → P(appears next)               — true "cold/hot" effect
 2. Number → Number transition (80×80)    — pair memory
 3. Hot-window count → P(appears next)    — momentum effect
 4. Daily cycle autocorr (lag 272)        — same hour next day
 5. Weekly cycle autocorr (lag 1904)      — same time next week
 6. Sum momentum (sum bucket transitions)
 7. Parity (odd-count) transitions
 8. Modular biases (mod 4, 5, 7) — per draw and conditional
 9. Grid-neighbor activation
10. Conditional pair → next number
11. Hour-of-day × delay interaction
12. Repeat-from-last-draw (numbers carrying over)
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from math import sqrt, log

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

# Binary matrix
M = np.zeros((N, 81), dtype=np.int8)
for i, (_, nums) in enumerate(all_draws):
    for n in nums:
        M[i, n] = 1

ANCHOR_ID = 1303293
ANCHOR_DT = datetime(2026,5,31,23,55,tzinfo=timezone(timedelta(hours=3)))
def draw_time(idx):
    return ANCHOR_DT + timedelta(minutes=(all_draws[idx][0] - ANCHOR_ID) * 5.28)

p = 20/80
all_signals = []  # (cat, name, z, det)
per_number_boost = defaultdict(lambda: defaultdict(float))  # boost[number][condition_key] = score
def add_signal(cat, name, z, det):
    all_signals.append((cat, name, z, det))

# ═══════════════════════════════════════════════════════════════════
# Precompute delays at every index (vectorized, in chunks)
# delays[i, n] = how many draws back was n last seen
# ═══════════════════════════════════════════════════════════════════
print("\nPrecomputing delays...")
t0 = time.time()
# For each number, find indices where it appears, then compute delay sequentially
delay_arr = np.full((N, 81), 999, dtype=np.int32)  # 999 = never seen yet
last_seen = np.full(81, -999, dtype=np.int32)
for i in range(N):
    for n in range(1, 81):
        if last_seen[n] >= 0:
            delay_arr[i, n] = i - last_seen[n]
    for n in all_draws[i][1]:
        last_seen[n] = i
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 1: DELAY → P(appears next)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] DELAY → P(appears next draw)")
print("="*70)
t0 = time.time()
# For each delay value d (1..50), what's P(n appears at idx | delay = d at idx)?
# This is the most basic "cold" intuition test.
delay_results = []
for d in range(1, 51):
    # All (idx, n) pairs where delay_arr[idx, n] == d
    mask = delay_arr == d
    if mask.sum() < 1000: continue
    # Did number n appear at idx?  → M[idx, n]
    appears = M[mask].astype(np.int32).sum() if False else 0  # not right
    # Simpler: for every i, for every n where delay_arr[i,n]==d, check M[i,n]
    # But delay_arr[i,n]==d means n was last seen at i-d. M[i,n]=1 means n appears NOW.
    # So we're testing P(n appears at i | last seen at i-d)
    total = int(mask.sum())
    appeared = int(M[mask].sum())
    p_cond = appeared / total
    z = (p_cond - p) / sqrt(p*(1-p)/total)
    delay_results.append((d, total, p_cond, z))

print(f"  Delay  Count       P(appears)   z-score   ({time.time()-t0:.1f}s)")
for d, total, pc, z in delay_results[:25]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    {d:>3}  {total:>10,}  {pc:.4f}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("delay", f"d={d}", z, f"delay={d}")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: NUMBER → NUMBER transition matrix (80×80)
# trans[a,b] = P(b in draw_{i+1} | a in draw_i)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] NUMBER → NUMBER transition matrix")
print("="*70)
t0 = time.time()
M_curr = M[:-1].astype(np.int32)   # N-1 x 81
M_next = M[1:].astype(np.int32)    # N-1 x 81
# Count: trans[a,b] = sum over i of M_curr[i,a] * M_next[i,b]
trans = M_curr.T @ M_next  # 81x81
# For each a, count_a = sum M_curr[:,a]
count_a = M_curr.sum(axis=0)  # 81
# P(b in next | a in curr) = trans[a,b] / count_a
# Expected under independence: P(b) = 0.25
# z = (count - count_a * 0.25) / sqrt(count_a * 0.25 * 0.75)
all_trans = []
for a in range(1, 81):
    if count_a[a] == 0: continue
    for b in range(1, 81):
        c = int(trans[a, b])
        exp = count_a[a] * p
        var = count_a[a] * p * (1-p)
        z = (c - exp) / sqrt(var)
        if abs(z) > 3.0:
            all_trans.append((a, b, c, z))
all_trans.sort(key=lambda x: -abs(x[3]))
print(f"  Found {len(all_trans)} transitions with |z|>3  ({time.time()-t0:.1f}s)")
print(f"  Top 20: 'if a in draw_i, then b appears in draw_{{i+1}} with deviation:'")
for a, b, c, z in all_trans[:20]:
    p_cond = c / count_a[a]
    print(f"    {a:>2} → {b:>2}  P={p_cond:.4f}  count={c:>6}  z={z:+.2f}")
    add_signal("trans", f"{a}→{b}", z, f"{a}→{b} P={p_cond:.4f}")
    per_number_boost[b]['trans'] += z * 0.01

# ═══════════════════════════════════════════════════════════════════
# TEST 3: HOT-WINDOW COUNT → P(appears next)
# For window W, count appearances of n in last W draws, see P(appears)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] HOT-WINDOW: count in last W → P(appears next)")
print("="*70)
t0 = time.time()
for W in [10, 30, 100]:
    # For each i, for each n, count = sum(M[i-W..i-1, n])
    # We want: given count = c, P(n at i)?
    # Use convolution: rolling sum
    rolling = np.zeros((N, 81), dtype=np.int32)
    if W < N:
        cumsum = np.zeros((N+1, 81), dtype=np.int32)
        cumsum[1:] = np.cumsum(M, axis=0)
        rolling[W:] = cumsum[W:N] - cumsum[:N-W]
    # For each count value c (0..min(W,20)), aggregate
    exp_count = W * p
    # Distribution of counts (EXCLUDE n=0 column — placeholder, never appears)
    print(f"\n  Window W={W} (expected count = {exp_count:.1f})")
    results = []
    for c in range(0, min(W+1, 22)):
        mask = (rolling[W:, 1:] == c)  # 2D mask, columns 1..80 only
        total = int(mask.sum())
        if total < 1000: continue
        appears_arr = M[W:, 1:][mask]
        appeared = int(appears_arr.sum())
        p_cond = appeared / total
        z = (p_cond - p) / sqrt(p*(1-p)/total)
        results.append((c, total, p_cond, z))
    print(f"    count  total      P(next)  z")
    for c, t, pc, z in results:
        flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
        print(f"    {c:>5}  {t:>9,}  {pc:.4f}  z={z:+.2f}{flag}")
        if abs(z) > 2.5:
            add_signal("hot", f"W{W}c{c}", z, f"W={W} count={c}")
print(f"\n  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: DAILY CYCLE (lag 272 ≈ 1 day) per number
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4] DAILY CYCLE: autocorrelation at lag 272 per number")
print("="*70)
t0 = time.time()
DAY_LAG = 272  # ~272 draws/day at 5.28 min
daily_corr = []
for n in range(1, 81):
    arr = M[:, n].astype(np.float64)
    arr = arr - arr.mean()
    if arr.std() == 0: continue
    # Correlation at lag DAY_LAG
    n_pts = N - DAY_LAG
    cov = np.dot(arr[:-DAY_LAG], arr[DAY_LAG:]) / n_pts
    var = np.dot(arr, arr) / N
    corr = cov / var
    z = corr * sqrt(n_pts)
    daily_corr.append((n, corr, z))
daily_corr.sort(key=lambda x: -abs(x[2]))
print(f"  Top 10 numbers with daily cycle  ({time.time()-t0:.1f}s)")
for n, c, z in daily_corr[:10]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    #{n:>2}: autocorr@272 = {c:+.6f}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("daily", f"n{n}", z, f"#{n} daily cycle")
        per_number_boost[n]['daily'] += z * 0.03

# ═══════════════════════════════════════════════════════════════════
# TEST 5: WEEKLY CYCLE (lag 1904 ≈ 7 days)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[5] WEEKLY CYCLE: autocorrelation at lag 1904 per number")
print("="*70)
t0 = time.time()
WEEK_LAG = 1904
weekly_corr = []
for n in range(1, 81):
    arr = M[:, n].astype(np.float64)
    arr = arr - arr.mean()
    if arr.std() == 0: continue
    n_pts = N - WEEK_LAG
    cov = np.dot(arr[:-WEEK_LAG], arr[WEEK_LAG:]) / n_pts
    var = np.dot(arr, arr) / N
    corr = cov / var
    z = corr * sqrt(n_pts)
    weekly_corr.append((n, corr, z))
weekly_corr.sort(key=lambda x: -abs(x[2]))
print(f"  Top 10 numbers with weekly cycle  ({time.time()-t0:.1f}s)")
for n, c, z in weekly_corr[:10]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    #{n:>2}: autocorr@1904 = {c:+.6f}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("weekly", f"n{n}", z, f"#{n} weekly cycle")
        per_number_boost[n]['weekly'] += z * 0.02

# ═══════════════════════════════════════════════════════════════════
# TEST 6: SUM MOMENTUM
# Bin sums into deciles, check if current bin predicts next bin
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] SUM MOMENTUM: P(next sum bin | current sum bin)")
print("="*70)
t0 = time.time()
sums = np.array([sum(d) for _, d in all_draws])
# Decile bins
bin_edges = np.percentile(sums, [10,20,30,40,50,60,70,80,90])
bins = np.digitize(sums, bin_edges)  # 0..9
# Transition matrix
trans_sum = np.zeros((10, 10), dtype=np.int32)
for i in range(N-1):
    trans_sum[bins[i], bins[i+1]] += 1
# Chi-square test: expected = row_sum * col_sum / total
total = trans_sum.sum()
row_sums = trans_sum.sum(axis=1)
col_sums = trans_sum.sum(axis=0)
chi2 = 0.0
strongest = []
for i in range(10):
    for j in range(10):
        exp = row_sums[i] * col_sums[j] / total
        if exp > 0:
            z = (trans_sum[i,j] - exp) / sqrt(exp)
            chi2 += (trans_sum[i,j] - exp)**2 / exp
            if abs(z) > 2.5:
                strongest.append((i, j, trans_sum[i,j], z))
strongest.sort(key=lambda x: -abs(x[3]))
df = 81
# Chi-square p-value (approx via z)
chi2_z = (chi2 - df) / sqrt(2*df)
print(f"  Chi-square: {chi2:.2f}  df={df}  z={chi2_z:+.2f}  ({time.time()-t0:.1f}s)")
print(f"  Strongest sum-bucket transitions (|z|>2.5):")
for i, j, c, z in strongest[:8]:
    print(f"    bin {i} → bin {j}: {c} (z={z:+.2f})")
    add_signal("sum_trans", f"{i}→{j}", z, f"sum bin {i}→{j}")

# ═══════════════════════════════════════════════════════════════════
# TEST 7: PARITY (odd count) transitions
# Each draw has 0..20 odd numbers (expected 10)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[7] PARITY: odd-count transitions")
print("="*70)
t0 = time.time()
odd_counts = np.zeros(N, dtype=np.int8)
for i, (_, nums) in enumerate(all_draws):
    odd_counts[i] = sum(1 for n in nums if n % 2 == 1)
# Bin: 0-7, 8, 9, 10, 11, 12, 13+
def bin_odd(c):
    if c <= 7: return 0
    if c >= 13: return 6
    return c - 7
binned = np.array([bin_odd(c) for c in odd_counts])
n_bins = 7
trans_parity = np.zeros((n_bins, n_bins), dtype=np.int32)
for i in range(N-1):
    trans_parity[binned[i], binned[i+1]] += 1
total_p = trans_parity.sum()
row_p = trans_parity.sum(axis=1)
col_p = trans_parity.sum(axis=0)
chi2_p = 0
strongest_p = []
for i in range(n_bins):
    for j in range(n_bins):
        exp = row_p[i] * col_p[j] / total_p
        if exp > 0:
            z = (trans_parity[i,j] - exp) / sqrt(exp)
            chi2_p += (trans_parity[i,j] - exp)**2 / exp
            if abs(z) > 2.5:
                strongest_p.append((i, j, trans_parity[i,j], z))
df_p = (n_bins-1)**2
chi2_z_p = (chi2_p - df_p) / sqrt(2*df_p)
print(f"  Chi-square: {chi2_p:.2f}  df={df_p}  z={chi2_z_p:+.2f}  ({time.time()-t0:.1f}s)")
for i, j, c, z in sorted(strongest_p, key=lambda x: -abs(x[3]))[:5]:
    print(f"    bin {i} → bin {j}: {c} (z={z:+.2f})")
    add_signal("parity", f"{i}→{j}", z, f"parity {i}→{j}")

# ═══════════════════════════════════════════════════════════════════
# TEST 8: MODULAR BIASES per draw — does the count of n%K=r have bias?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[8] MODULAR BIASES per draw")
print("="*70)
t0 = time.time()
for K in [4, 5, 7]:
    # For each draw, count how many numbers ≡ r (mod K), for r=0..K-1
    counts = np.zeros((N, K), dtype=np.int32)
    for i, (_, nums) in enumerate(all_draws):
        for n in nums:
            counts[i, n % K] += 1
    # Expected per (draw, r) = 20 * (#numbers ≡ r) / 80
    # Numbers 1..80, count per residue:
    res_count = [sum(1 for n in range(1,81) if n%K==r) for r in range(K)]
    exp_per_r = [20 * rc/80 for rc in res_count]
    var_per_r = [20 * (rc/80) * (1-rc/80) for rc in res_count]
    print(f"  mod {K}:")
    for r in range(K):
        mean_obs = counts[:, r].mean()
        z = (mean_obs - exp_per_r[r]) / (sqrt(var_per_r[r])/sqrt(N))
        flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
        print(f"    r={r}: mean={mean_obs:.4f}  exp={exp_per_r[r]:.4f}  z={z:+.2f}{flag}")
        if abs(z) > 2.5:
            add_signal("mod", f"mod{K}_r{r}", z, f"mod {K} residue {r}")
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 9: GRID-NEIGHBOR ACTIVATION
# If n appeared in draw_i, does its grid-neighbor m appear in draw_{i+1}?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[9] GRID-NEIGHBOR (8-direction) activation in next draw")
print("="*70)
t0 = time.time()
def neighbors(n):
    """8-directional neighbors in 8×10 grid (flat, no toroidal)."""
    row, col = (n-1)//10, (n-1)%10
    out = []
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr == 0 and dc == 0: continue
            r2, c2 = row+dr, col+dc
            if 0 <= r2 < 8 and 0 <= c2 < 10:
                out.append(r2*10+c2+1)
    return out

# For each (n, m) where m is neighbor of n, compute:
#   P(m in draw_{i+1} | n in draw_i)
# Compare to P(m in draw) = 0.25
neighbor_pairs = []
for n in range(1, 81):
    for m in neighbors(n):
        # count appearances of n in draws[:N-1], and m in draws[1:N] simultaneously
        n_mask = M[:-1, n] == 1
        cnt_n = int(n_mask.sum())
        if cnt_n == 0: continue
        cnt_both = int((M[1:, m][n_mask]).sum())
        p_cond = cnt_both / cnt_n
        z = (p_cond - p) / sqrt(p*(1-p)/cnt_n)
        neighbor_pairs.append((n, m, p_cond, z))
neighbor_pairs.sort(key=lambda x: -abs(x[3]))
print(f"  Top 15 neighbor activation pairs  ({time.time()-t0:.1f}s)")
for n, m, pc, z in neighbor_pairs[:15]:
    flag = " ★" if abs(z) > 3 else ""
    print(f"    #{n:>2} → neighbor #{m:>2}  P={pc:.4f}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("neighbor", f"{n}→{m}", z, f"neighbor {n}→{m}")
        per_number_boost[m]['neighbor'] += z * 0.005

# ═══════════════════════════════════════════════════════════════════
# TEST 10: REPEAT-FROM-LAST-DRAW (how many numbers carry over)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[10] REPEAT COUNT: how many numbers from draw_i appear in draw_{i+1}?")
print("="*70)
t0 = time.time()
overlap = np.zeros(N-1, dtype=np.int8)
for i in range(N-1):
    overlap[i] = int(np.dot(M[i].astype(np.int32), M[i+1].astype(np.int32)))
# Build conditional: given overlap[i-1], P(overlap[i] = k)?
# Just test marginal first
exp_overlap = 5.0
exp_std = sqrt(20 * (60/80) * (20/79) * (60/79))
z = (overlap.mean() - exp_overlap) / (exp_std/sqrt(N-1))
print(f"  Mean overlap: {overlap.mean():.4f} (exp 5.0, z={z:+.2f})")
# Distribution
counter = Counter(overlap.tolist())
print(f"  Distribution:")
from math import comb
for k in range(0, 13):
    obs = counter.get(k, 0)
    # Hypergeometric expected
    exp_k = (N-1) * comb(20,k)*comb(60,20-k)/comb(80,20)
    if exp_k > 0:
        z_k = (obs - exp_k) / sqrt(exp_k)
        flag = " ★" if abs(z_k) > 3 else ""
        print(f"    {k:>2} repeats: obs={obs:>6}  exp={exp_k:>9.1f}  z={z_k:+.2f}{flag}")
        if abs(z_k) > 2.5:
            add_signal("repeat", f"k{k}", z_k, f"repeat count {k}")
print(f"  ({time.time()-t0:.1f}s)")

# Markov chain on overlap: does overlap[i-1] predict overlap[i]?
print(f"\n  Markov on overlap (does overlap_{{i-1}} predict overlap_i?):")
trans_ov = np.zeros((11, 11), dtype=np.int32)
for i in range(1, N-1):
    a = min(overlap[i-1], 10)
    b = min(overlap[i], 10)
    trans_ov[a, b] += 1
total_ov = trans_ov.sum()
row_ov = trans_ov.sum(axis=1)
col_ov = trans_ov.sum(axis=0)
strongest_ov = []
chi2_ov = 0
for i in range(11):
    for j in range(11):
        exp = row_ov[i] * col_ov[j] / total_ov
        if exp > 100:
            z = (trans_ov[i,j] - exp) / sqrt(exp)
            chi2_ov += (trans_ov[i,j] - exp)**2 / exp
            if abs(z) > 2.5:
                strongest_ov.append((i,j,trans_ov[i,j],z))
df_ov = 100
chi2_z_ov = (chi2_ov - df_ov) / sqrt(2*df_ov)
print(f"  Chi-square: {chi2_ov:.2f}  df={df_ov}  z={chi2_z_ov:+.2f}")
for i, j, c, z in sorted(strongest_ov, key=lambda x: -abs(x[3]))[:5]:
    print(f"    overlap {i}→{j}: {c} (z={z:+.2f})")

# ═══════════════════════════════════════════════════════════════════
# TEST 11: HOUR × DELAY interaction
# For each (hour, delay), what's P(number appears)?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[11] HOUR × DELAY interaction")
print("="*70)
t0 = time.time()
# Pre-compute hour at each draw
hours = np.array([draw_time(i).hour for i in range(N)], dtype=np.int8)
# For each (hour h, delay d), find all (i, n) where hours[i]==h AND delay_arr[i,n]==d
# This is too many cells (24×50×80). Aggregate over numbers per (hour, delay).
interaction = []
for h in range(24):
    h_mask = hours == h
    for d in range(1, 20):
        # Find indices where hours[i]==h
        idx_h = np.where(h_mask)[0]
        if len(idx_h) < 100: continue
        # For each n, count where delay_arr[idx, n] == d
        delays_at_h = delay_arr[idx_h]  # (count_h, 81)
        for n in range(1, 81):
            mask_dn = delays_at_h[:, n] == d
            total = int(mask_dn.sum())
            if total < 200: continue
            # Among these, how many appeared (M[idx_h[mask_dn], n] == 1)?
            appeared = int(M[idx_h[mask_dn], n].sum())
            p_cond = appeared / total
            z = (p_cond - p) / sqrt(p*(1-p)/total)
            if abs(z) > 3.5:
                interaction.append((h, d, n, total, p_cond, z))
interaction.sort(key=lambda x: -abs(x[5]))
print(f"  Found {len(interaction)} hour×delay×number cells with |z|>3.5  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for h, d, n, t_, pc, z in interaction[:15]:
    print(f"    h={h:>2} delay={d:>2} #{n:>2}: P={pc:.4f} n={t_} z={z:+.2f}")
    add_signal("hxd", f"h{h}d{d}n{n}", z, f"h={h} d={d} #{n}")
    per_number_boost[n][f'hxd_h{h}'] = z * 0.05

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ — {len(all_signals)} conditional σήματα")
print("="*70)
all_signals.sort(key=lambda x: -abs(x[2]))
print(f"\n  Top 30 (sorted by |z|):")
for i, (cat, name, z, det) in enumerate(all_signals[:30], 1):
    print(f"  {i:>3}.  [{cat:>10}]  {name:>15}  |z|={abs(z):>6.2f}  {'+' if z>0 else '-'}  {det}")

# Per-category summary
print(f"\n  Σύνοψη ανά κατηγορία:")
cats = defaultdict(list)
for cat, name, z, det in all_signals:
    cats[cat].append(abs(z))
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"    {cat:>12}: {len(arr):>3} σήματα, max |z|={max(arr):.2f}, mean={np.mean(arr):.2f}")

# Save
out = {
    'signals': [(c, n, float(z), d) for c, n, z, d in all_signals],
    'per_number_boost': {n: dict(per_number_boost[n]) for n in per_number_boost},
    'N': N,
}
with open('/home/user/Game/conditional_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: conditional_signals.json ({len(all_signals)} σήματα)")

print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
