#!/usr/bin/env python3
"""
ΤΕΛΕΥΤΑΙΑ μεγάλη ομάδα — Prior: ΔΕΝ είναι τυχαίο.

10 πιο σύνθετα tests:
 1. Largest-component centroid + autocorr (chain spatial bias)
 2. Overlap-conditional Markov (P(carry-over | overlap_{i-1}=k))
 3. Pair → next pair (sparse, top pairs)
 4. Burstiness coefficient per number (σ/μ of inter-arrival)
 5. FFT per number → top frequency component
 6. Modular sum (mod 7, 11, 13)
 7. Sum-of-squares per draw + Markov
 8. Cross-correlation pair-wise (a,m) at lag 1
 9. AND-conditional: P(b in i+1 | a in i AND b in i)
10. Hour × DoW × number 3-way interaction
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from math import sqrt, log, comb, pi

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
# TEST 1: LARGEST-COMPONENT CENTROID per draw + autocorr
# Use flat adjacency (8-directional, no toroidal)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] LARGEST-COMPONENT centroid per draw + autocorr")
print("="*70)
t0 = time.time()
def adj_8(n):
    row, col = (n-1)//10, (n-1)%10
    out = []
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr == 0 and dc == 0: continue
            r2, c2 = row+dr, col+dc
            if 0 <= r2 < 8 and 0 <= c2 < 10:
                out.append(r2*10+c2+1)
    return out

# Precompute neighbors
NEIGH = {n: adj_8(n) for n in range(1, 81)}

def largest_component_size_and_centroid(nums):
    s = set(nums)
    visited = set()
    best_size = 0
    best_centroid = (3.5, 4.5)
    for start in nums:
        if start in visited: continue
        # BFS
        stack = [start]
        comp = []
        while stack:
            x = stack.pop()
            if x in visited: continue
            visited.add(x)
            comp.append(x)
            for nb in NEIGH[x]:
                if nb in s and nb not in visited:
                    stack.append(nb)
        if len(comp) > best_size:
            best_size = len(comp)
            rs = [(c-1)//10 for c in comp]
            cs = [(c-1)%10 for c in comp]
            best_centroid = (np.mean(rs), np.mean(cs))
    return best_size, best_centroid

print(f"  Computing per-draw largest component (slow)...")
comp_sizes = np.zeros(N, dtype=np.int8)
comp_rows = np.zeros(N, dtype=np.float32)
comp_cols = np.zeros(N, dtype=np.float32)
for i, (_, nums) in enumerate(all_draws):
    sz, (r, c) = largest_component_size_and_centroid(nums)
    comp_sizes[i] = min(sz, 127)
    comp_rows[i] = r
    comp_cols[i] = c
print(f"  Done ({time.time()-t0:.1f}s)")

# Distribution
sz_counts = Counter(comp_sizes.tolist())
print(f"  Component size distribution (matches CLAUDE.md if accurate):")
for sz in sorted(sz_counts.keys())[:15]:
    print(f"    n={sz}: {sz_counts[sz]:,} ({sz_counts[sz]/N*100:.1f}%)")

# Autocorr of centroid
for name, arr in [('row', comp_rows), ('col', comp_cols), ('size', comp_sizes.astype(np.float32))]:
    a = arr.astype(np.float64) - arr.mean()
    var = np.dot(a, a) / N
    if var == 0: continue
    for lag in [1, 2, 5, 10, 272]:
        cov = np.dot(a[:-lag], a[lag:]) / (N-lag)
        c = cov / var
        z = c * sqrt(N-lag)
        flag = " ★" if abs(z) > 3 else ""
        print(f"  comp_{name:>4} autocorr lag={lag:>3}: {c:+.6f} z={z:+.2f}{flag}")
        if abs(z) > 2.5:
            add_signal("comp_acf", f"{name}@lag{lag}", z, f"comp {name} lag {lag}")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: OVERLAP-CONDITIONAL MARKOV
# Given overlap_{i-1} = k, P(specific numbers carry over)?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] OVERLAP-CONDITIONAL: P(carry-over | prev overlap)")
print("="*70)
t0 = time.time()
overlaps = np.zeros(N-1, dtype=np.int8)
for i in range(N-1):
    overlaps[i] = int(np.dot(M[i].astype(np.int32), M[i+1].astype(np.int32)))
# For each overlap level k, compute next overlap distribution
trans = np.zeros((13, 13), dtype=np.int32)
for i in range(1, N-1):
    a = min(overlaps[i-1], 12)
    b = min(overlaps[i], 12)
    trans[a, b] += 1
total = trans.sum()
row = trans.sum(axis=1)
col = trans.sum(axis=0)
strong = []
chi2 = 0; df = 0
for i in range(13):
    for j in range(13):
        if row[i] * col[j] > 0:
            exp = row[i] * col[j] / total
            if exp > 100:
                z = (trans[i,j] - exp) / sqrt(exp)
                chi2 += (trans[i,j] - exp)**2 / exp
                df += 1
                if abs(z) > 2.5:
                    strong.append((i, j, trans[i,j], z))
chi2_z = (chi2 - df) / sqrt(2*df) if df > 0 else 0
print(f"  Overall chi-square z = {chi2_z:+.2f}  ({time.time()-t0:.1f}s)")
strong.sort(key=lambda x: -abs(x[3]))
for i, j, c, z in strong[:8]:
    print(f"    overlap {i} → {j}: {c}  z={z:+.2f}")
    if abs(z) > 3:
        add_signal("ov_markov", f"{i}→{j}", z, f"overlap {i}→{j}")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: PAIR → next PAIR (sparse)
# For each high-frequency pair (a,b), find which next-pair (c,d) is biased
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] PAIR → next PAIR")
print("="*70)
t0 = time.time()
# Use top 30 pairs to test
M32 = M.astype(np.int32)
pair_total = M32.T @ M32  # 81×81
# Get high-freq pairs (sorted)
pair_freq = []
for a in range(1, 81):
    for b in range(a+1, 81):
        pair_freq.append((pair_total[a,b], a, b))
pair_freq.sort(reverse=True)
top_pairs = [(a, b) for _, a, b in pair_freq[:50]]

# For each test pair (a,b), find indices and check next-draw pairs
pair_to_pair_sigs = []
exp_pp = N * 20*19/(80*79)  # rough
for (a, b) in top_pairs[:30]:
    indices_with_pair = []
    for i in range(N-1):
        if M[i, a] and M[i, b]:
            indices_with_pair.append(i)
    n_with = len(indices_with_pair)
    if n_with < 10000: continue
    # Next draws
    next_M_sub = M[np.array(indices_with_pair)+1].astype(np.int32)
    pair_next_count = next_M_sub.T @ next_M_sub  # 81x81 for these next draws
    # For each candidate (c,d), is pair_next_count[c,d] biased vs expected?
    # Expected: n_with * 20*19/(80*79)
    exp_cd = n_with * 20*19/(80*79)
    var_cd = exp_cd * (1 - 20*19/(80*79))
    for c_ in range(1, 81):
        for d_ in range(c_+1, 81):
            cnt = int(pair_next_count[c_, d_])
            z = (cnt - exp_cd) / sqrt(var_cd)
            if abs(z) > 4.5:
                pair_to_pair_sigs.append(((a,b), (c_,d_), cnt, n_with, z))
pair_to_pair_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(pair_to_pair_sigs)} pair→pair signals with |z|>4.5  ({time.time()-t0:.1f}s)")
print(f"  Top 10:")
for (a,b), (c_,d_), cnt, n_with, z in pair_to_pair_sigs[:10]:
    print(f"    ({a:>2},{b:>2}) → ({c_:>2},{d_:>2})  count={cnt}/{n_with}  z={z:+.2f}")
    if abs(z) > 4.5:
        add_signal("pp", f"({a},{b})→({c_},{d_})", z, f"pair→pair")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: BURSTINESS COEFFICIENT per number
# B = (σ - μ) / (σ + μ); B>0 means bursty
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4] BURSTINESS coefficient per number")
print("="*70)
t0 = time.time()
burst = []
for n in range(1, 81):
    indices = np.where(M[:, n] == 1)[0]
    if len(indices) < 100: continue
    gaps = np.diff(indices)
    mu = gaps.mean()
    sigma = gaps.std()
    B = (sigma - mu) / (sigma + mu)
    # For geometric distribution (random), σ = sqrt(μ(μ-1)) ≈ μ-0.5
    # So B_expected ≈ (μ-0.5-μ)/(μ-0.5+μ) ≈ -0.5/(2μ-0.5) ≈ small negative
    # Variance of B estimate ~ ?
    n_g = len(gaps)
    # For geometric μ≈4: σ≈3.46, B_expected ≈ -0.07
    # Bootstrap-like estimate using SE
    se = sqrt(2/n_g) * (1 - B**2)  # rough
    z = (B - (-0.069)) / max(se, 0.001)
    burst.append((n, B, z))
burst.sort(key=lambda x: -abs(x[2]))
print(f"  Top 10 burstiest numbers  ({time.time()-t0:.1f}s):")
for n, B, z in burst[:10]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    #{n:>2}: B={B:+.4f}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("burst", f"n{n}", z, f"#{n} burstiness")

# ═══════════════════════════════════════════════════════════════════
# TEST 5: FFT per number → top frequency component
# Find dominant periodicity for each number
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[5] FFT per number — properly normalized (power follows χ²(2) for i.i.d.)")
print("="*70)
t0 = time.time()
# For each number, compute periodogram and find anomalous peaks.
# For i.i.d. white noise, periodogram bins follow exp(λ) where λ = N*p*(1-p).
# z-score in exponential: z = log(power/expected) (extreme value)
# Use proper: ratio = power / mean_power. P(ratio > x) = exp(-x).
# For K bins, by chance one bin will hit ratio ~ ln(K). Threshold ratio > 5*ln(K) is anomalous.
fft_results = []
for n in range(1, 81):
    arr = M[:, n].astype(np.float64) - p
    F = np.fft.rfft(arr)
    P = np.abs(F)**2
    K = len(P) - 1  # exclude DC
    mean_P = P[1:].mean()
    # For i.i.d., P(ratio > x) = exp(-x). Expect max ratio ≈ ln(K).
    # Threshold: ratio > 3 * ln(K) ≈ 33 for K=120K  → P < 1/exp(33) per bin
    threshold_ratio = 3 * log(K)
    max_idx = int(np.argmax(P[1:]) + 1)
    ratio = P[max_idx] / mean_P
    if ratio > threshold_ratio:
        period = N / max_idx if max_idx > 0 else 0
        # Convert to pseudo-z via inverse exp
        z = ratio - log(K)  # how many "standard deviations" above expected max
        fft_results.append((n, period, ratio, z))
fft_results.sort(key=lambda x: -x[3])
print(f"  Numbers with FFT anomaly above threshold ({time.time()-t0:.1f}s):")
print(f"  (mean ratio for noise ≈ 1.0; expected max ratio ≈ ln(K)={log(N//2):.1f})")
for n, period, ratio, z in fft_results[:10]:
    print(f"    #{n:>2}  period={period:>7.1f}  ratio={ratio:>7.2f}  excess={z:+.2f}")
    if z > 3:
        add_signal("fft", f"n{n}@p{period:.0f}", z, f"#{n} period {period:.0f}")

# ═══════════════════════════════════════════════════════════════════
# TEST 6: MODULAR SUM (sum mod K)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] MODULAR SUM per draw")
print("="*70)
sums = np.array([sum(d) for _, d in all_draws])
for K in [7, 11, 13, 17, 19, 23]:
    mods = sums % K
    cnts = Counter(mods.tolist())
    # Expected: N/K per residue
    exp = N / K
    chi2 = sum((cnts.get(r, 0) - exp)**2 / exp for r in range(K))
    df = K - 1
    chi2_z = (chi2 - df) / sqrt(2*df)
    flag = " ★" if abs(chi2_z) > 3 else ""
    print(f"  sum mod {K:>2}: chi² = {chi2:.2f}  df={df}  z={chi2_z:+.2f}{flag}")
    if abs(chi2_z) > 2.5:
        add_signal("sum_mod", f"mod{K}", chi2_z, f"sum mod {K}")

# ═══════════════════════════════════════════════════════════════════
# TEST 7: SUM OF SQUARES per draw + Markov
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[7] SUM OF SQUARES per draw")
print("="*70)
sq_sums = np.array([sum(n*n for n in d) for _, d in all_draws])
# Theoretical: E[sum X²] = 20 * E[X²] = 20 * (1²+...+80²)/80 = 20 * 2173.5 = 43,470
exp_sq_mean = 20 * (80*81*161/6) / 80
# Std: use large MC (100K samples) for accurate reference
import random as _r
_r.seed(0)
mc = [sum(n*n for n in _r.sample(range(1,81), 20)) for _ in range(100000)]
mc_std = np.std(mc)
z_sq = (sq_sums.mean() - exp_sq_mean) / (mc_std/sqrt(N))
print(f"  Mean: {sq_sums.mean():.2f} (theory {exp_sq_mean:.2f}, σ={mc_std:.1f})  z={z_sq:+.2f}")
if abs(z_sq) > 2: add_signal("sq_sum", "mean", z_sq, "sum of squares")

# Autocorr
sq_c = sq_sums.astype(np.float64) - sq_sums.mean()
var = np.dot(sq_c, sq_c) / N
for lag in [1, 5, 272]:
    cov = np.dot(sq_c[:-lag], sq_c[lag:]) / (N-lag)
    c = cov / var
    z = c * sqrt(N-lag)
    flag = " ★" if abs(z) > 3 else ""
    print(f"  autocorr lag={lag:>3}: {c:+.6f}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("sq_acf", f"lag{lag}", z, f"sq autocorr {lag}")

# ═══════════════════════════════════════════════════════════════════
# TEST 8: CROSS-CORRELATION pair-wise (a,m) at lag 1
# r_{a,m}(1) = corr(M[:,a], M[lag:,m])
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[8] CROSS-CORRELATION (a,m) at lag 1 — different from a→m transition")
print("="*70)
t0 = time.time()
# This is similar to trans matrix from earlier (a in i → m in i+1)
# but normalized differently
# For each a, m: corr(M[:-1, a], M[1:, m])
cross = []
arr_curr = M[:-1].astype(np.float64) - p
arr_next = M[1:].astype(np.float64) - p
denom = (N-1) * p * (1-p)
for a in range(1, 81):
    for m in range(1, 81):
        if a == m: continue
        cov = np.dot(arr_curr[:, a], arr_next[:, m])
        r = cov / denom
        z = r * sqrt(N-1)
        if abs(z) > 3.5:
            cross.append((a, m, r, z))
cross.sort(key=lambda x: -abs(x[3]))
print(f"  Found {len(cross)} cross-corrs with |z|>3.5  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for a, m, r, z in cross[:15]:
    flag = " ★" if abs(z) > 4 else ""
    print(f"    {a:>2} → {m:>2}  r={r:+.6f}  z={z:+.2f}{flag}")
    if abs(z) > 3.5:
        add_signal("xcorr", f"{a}→{m}", z, f"xcorr {a}→{m}")

# ═══════════════════════════════════════════════════════════════════
# TEST 9: AND-CONDITIONAL: P(b in i+1 | a in i AND b in i)
# If both a and b in draw_i, does b carry over?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[9] AND-CONDITIONAL: P(b carries over | a AND b in prev)")
print("="*70)
t0 = time.time()
and_cond_sigs = []
# Test top 50 pairs
M32 = M.astype(np.int32)
pair_count = M32.T @ M32
top_pairs_by_freq = []
for a in range(1, 81):
    for b in range(a+1, 81):
        top_pairs_by_freq.append((int(pair_count[a,b]), a, b))
top_pairs_by_freq.sort(reverse=True)
# For each pair (a,b), among draws containing both, does b appear in NEXT?
for cnt_ab, a, b in top_pairs_by_freq[:200]:
    mask = (M[:-1, a] == 1) & (M[:-1, b] == 1)
    n_both = int(mask.sum())
    if n_both < 5000: continue
    # Of those, how many have b in next?
    b_next = int((M[1:, b][mask]).sum())
    a_next = int((M[1:, a][mask]).sum())
    p_b = b_next / n_both
    p_a = a_next / n_both
    z_b = (p_b - p) / sqrt(p*(1-p)/n_both)
    z_a = (p_a - p) / sqrt(p*(1-p)/n_both)
    if abs(z_b) > 3.5:
        and_cond_sigs.append((a, b, 'b', p_b, n_both, z_b))
    if abs(z_a) > 3.5:
        and_cond_sigs.append((a, b, 'a', p_a, n_both, z_a))
and_cond_sigs.sort(key=lambda x: -abs(x[5]))
print(f"  Found {len(and_cond_sigs)} signals  ({time.time()-t0:.1f}s)")
print(f"  Top 10:")
for a, b, which, p_, n_, z in and_cond_sigs[:10]:
    target = b if which == 'b' else a
    print(f"    if {a:>2}&{b:>2} → {target:>2}: P={p_:.4f}  n={n_}  z={z:+.2f}")
    if abs(z) > 4:
        add_signal("and_cond", f"{a}&{b}→{target}", z, f"both→{target}")

# ═══════════════════════════════════════════════════════════════════
# TEST 10: HOUR × DOW × NUMBER (3-way interaction)
# Some numbers may have bias only on specific (hour, day) combos
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[10] HOUR × DOW × NUMBER")
print("="*70)
t0 = time.time()
hours = np.array([draw_time(i).hour for i in range(N)], dtype=np.int8)
dows = np.array([draw_time(i).weekday() for i in range(N)], dtype=np.int8)
# (24 × 7 × 80) cells = 13,440. Many will be sparse.
hxdxn = []
for h in range(24):
    for d in range(7):
        mask = (hours == h) & (dows == d)
        T = int(mask.sum())
        if T < 200: continue
        for n in range(1, 81):
            cnt = int(M[mask, n].sum())
            exp = T * p
            var = T * p * (1-p)
            z = (cnt - exp) / sqrt(var)
            if abs(z) > 4.0:
                hxdxn.append((h, d, n, T, cnt, z))
hxdxn.sort(key=lambda x: -abs(x[5]))
print(f"  Found {len(hxdxn)} (h, d, n) cells with |z|>4.0  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
DOW = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
for h, d, n, T, cnt, z in hxdxn[:15]:
    print(f"    h={h:>2} {DOW[d]} #{n:>2}: {cnt}/{T} z={z:+.2f}")
    add_signal("hxdxn", f"h{h}{DOW[d]}n{n}", z, f"h={h} {DOW[d]} #{n}")

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ — {len(all_signals)} νέα σήματα (5η ομάδα)")
print("="*70)
all_signals.sort(key=lambda x: -abs(x[2]))
print(f"\n  Top 30:")
for i, (cat, name, z, det) in enumerate(all_signals[:30], 1):
    print(f"  {i:>3}.  [{cat:>12}]  {name:>20}  |z|={abs(z):>6.2f}  {'+' if z>0 else '-'}")

print(f"\n  Σύνοψη ανά κατηγορία:")
cats = defaultdict(list)
for cat, name, z, det in all_signals:
    cats[cat].append(abs(z))
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"    {cat:>14}: {len(arr):>3} σήματα, max |z|={max(arr):.2f}")

out = {'signals': [(c, n, float(z), d) for c, n, z, d in all_signals], 'N': N}
with open('/home/user/Game/final_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: final_signals.json ({len(all_signals)} σήματα)")
print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
