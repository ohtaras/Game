#!/usr/bin/env python3
"""
9η ομάδα — τα τελευταία novel tests.

Tests:
 1. Pair time-since-cooccurrence (gap distribution for each pair)
 2. 3-way temporal: a in i, b in i+1, c in i+2 (sequential signature)
 3. Hour-specific n→n at LAG 2 (h, n, k=2)
 4. Conditional triple: top triple → next pair
 5. Long-distance pair lag (50, 100)
 6. Sum mod 4 Markov (transitions between residue classes)
 7. Position-in-historical-rank autocorr
 8. Anti-Multi-pair: NEITHER pair (a,b) NOR (c,d) in i → P(e)
 9. Last-K draws hot pair → recurrence
10. Number "loyalty": cross-half stability check
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

hours = np.array([draw_time(i).hour for i in range(N)], dtype=np.int8)

# ═══════════════════════════════════════════════════════════════════
# TEST 1: PAIR time-since-cooccurrence — gap distribution
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] PAIR time-between-cooccurrences")
print("="*70)
t0 = time.time()
# Top 50 most-frequent pairs
M32 = M.astype(np.int32)
pair_total = M32.T @ M32
top_pairs = []
for a in range(1, 81):
    for b in range(a+1, 81):
        top_pairs.append((int(pair_total[a,b]), a, b))
top_pairs.sort(reverse=True)

pair_gap_sigs = []
exp_pair_gap = N / np.mean([cnt for cnt, _, _ in top_pairs[:100]])  # approx
for cnt, a, b in top_pairs[:200]:
    # All indices where pair (a,b) appears
    idx = np.where((M[:, a] == 1) & (M[:, b] == 1))[0]
    if len(idx) < 100: continue
    gaps = np.diff(idx)
    mu = gaps.mean()
    sigma = gaps.std()
    n_g = len(gaps)
    # For geometric distribution with same mean, std ≈ mu - 0.5
    exp_sigma = sqrt(mu * (mu - 1)) if mu > 1 else 0.1
    if exp_sigma == 0: continue
    se_sigma = exp_sigma / sqrt(2 * n_g)
    z = (sigma - exp_sigma) / se_sigma
    if abs(z) > 3.5:
        pair_gap_sigs.append((a, b, mu, sigma, exp_sigma, z))
pair_gap_sigs.sort(key=lambda x: -abs(x[5]))
print(f"  Pairs with anomalous gap-std (top 200 tested)  ({time.time()-t0:.1f}s)")
print(f"  Top 10:")
for a, b, mu, s, es, z in pair_gap_sigs[:10]:
    print(f"    ({a:>2},{b:>2})  μ={mu:.2f}  σ={s:.2f} (exp {es:.2f})  z={z:+.2f}")
    if abs(z) > 4:
        add_signal("pair_gap", f"({a},{b})", z, f"pair ({a},{b}) gap-std")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: 3-WAY TEMPORAL: a in i, b in i+1, c in i+2
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] 3-WAY TEMPORAL CASCADE: a→b→c across 3 draws")
print("="*70)
t0 = time.time()
# For each (a, b), what's the bias on c in draw_{i+2}?
# This is a 3-D table: 80×80×80 = 512K combinations
# Limit: only test pairs (a→b) with strong lag-1 transition
# From earlier work: top pairs with strong n→n trans
# Re-compute trans matrix
trans = M32[:-1].T @ M32[1:]
# Top 20 a→b transitions by deviation
exp_t = M32[:-1].sum(axis=0) * p
trans_strong = []
for a in range(1, 81):
    c_a = int(M32[:-1, a].sum())
    if c_a == 0: continue
    for b in range(1, 81):
        cnt = int(trans[a, b])
        exp = c_a * p
        var = c_a * p * (1-p)
        z = (cnt - exp) / sqrt(var)
        if abs(z) > 2.5:
            trans_strong.append((a, b, z))
trans_strong.sort(key=lambda x: -abs(x[2]))
print(f"  Testing top {min(50, len(trans_strong))} a→b for 3-step cascade")

cascade_sigs = []
for a, b, z_ab in trans_strong[:50]:
    cond = (M[:-2, a] == 1) & (M[1:-1, b] == 1)
    T = int(cond.sum())
    if T < 4000: continue
    next_M = M[2:][cond]
    counts = next_M.sum(axis=0)
    for c in range(1, 81):
        if c in (a, b): continue
        cnt = int(counts[c])
        exp = T * p
        var = T * p * (1-p)
        z = (cnt - exp) / sqrt(var)
        if abs(z) > 4.0:
            cascade_sigs.append((a, b, c, T, cnt, z))
cascade_sigs.sort(key=lambda x: -abs(x[5]))
print(f"  Found {len(cascade_sigs)} cascade signals  ({time.time()-t0:.1f}s)")
for a, b, c, T, cnt, z in cascade_sigs[:10]:
    print(f"    {a:>2}→{b:>2}→{c:>2}: T={T}  P={cnt/T:.4f}  z={z:+.2f}")
    if abs(z) > 4:
        add_signal("cascade", f"{a}→{b}→{c}", z, f"cascade {a}→{b}→{c}")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: HOUR-SPECIFIC n→n at LAG 2
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] HOUR-SPECIFIC n→n at LAG 2 (skip-one Markov)")
print("="*70)
t0 = time.time()
hxn_lag2 = []
for h in range(24):
    h_mask = hours[:-2] == h
    for n in range(1, 81):
        in_i = (M[:-2, n] == 1) & h_mask
        T = int(in_i.sum())
        if T < 500: continue
        # Check n at i+2 (lag 2)
        appears = M[2:, n][in_i] == 1
        cnt = int(appears.sum())
        p_c = cnt / T
        z = (p_c - p) / sqrt(p*(1-p)/T)
        if abs(z) > 3.0:
            hxn_lag2.append((h, n, T, p_c, z))
hxn_lag2.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(hxn_lag2)} (h, n) lag-2 with |z|>3.0  ({time.time()-t0:.1f}s)")
for h, n, T, pc, z in hxn_lag2[:10]:
    print(f"    h={h:>2} #{n:>2}: P(at i+2)={pc:.4f} T={T} z={z:+.2f}")
    if abs(z) > 3.5:
        add_signal("hxn_lag2", f"h{h}n{n}", z, f"h={h} #{n} lag 2")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: ANTI-MULTI-PAIR (neither pair in i → P(e))
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4] ANTI-MULTI-PAIR: ¬(a,b) ∧ ¬(c,d) → P(e)")
print("="*70)
t0 = time.time()
# Sample top pairs and find anomalous absence patterns
import random as _r
_r.seed(0)
strong_pairs = [(75,80), (50,67), (68,80), (26,79), (15,22), (71,74), (12,17), (63,64)]
amp_sigs = []
for i_p in range(len(strong_pairs)):
    for j_p in range(i_p+1, len(strong_pairs)):
        a, b = strong_pairs[i_p]
        c, d = strong_pairs[j_p]
        if len({a,b,c,d}) < 4: continue
        cond = ((M[:-1, a] == 0) | (M[:-1, b] == 0)) & ((M[:-1, c] == 0) | (M[:-1, d] == 0))
        T = int(cond.sum())
        if T < 50000: continue
        next_M = M[1:][cond]
        counts = next_M.sum(axis=0)
        for e in range(1, 81):
            cnt = int(counts[e])
            exp = T * p
            var = T * p * (1-p)
            z = (cnt - exp) / sqrt(var)
            if abs(z) > 4.0:
                amp_sigs.append(((a,b), (c,d), e, T, cnt, z))
amp_sigs.sort(key=lambda x: -abs(x[5]))
print(f"  Found {len(amp_sigs)} signals  ({time.time()-t0:.1f}s)")
for (ab, cd, e, T, cnt, z) in amp_sigs[:8]:
    print(f"    ¬{ab} ∧ ¬{cd} → {e:>2}: T={T} P={cnt/T:.4f} z={z:+.2f}")
    if abs(z) > 4:
        add_signal("anti_mp", f"¬{ab}¬{cd}→{e}", z, f"¬{ab}∧¬{cd} → {e}")

# ═══════════════════════════════════════════════════════════════════
# TEST 5: SUM-MOD-4 MARKOV (transitions between sum residue classes)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[5] SUM MOD 4 Markov")
print("="*70)
t0 = time.time()
sums = np.array([sum(d) for _, d in all_draws])
mod_classes = sums % 4
trans_m = np.zeros((4, 4), dtype=np.int32)
for i in range(N-1):
    trans_m[mod_classes[i], mod_classes[i+1]] += 1
total = trans_m.sum()
row = trans_m.sum(axis=1)
col = trans_m.sum(axis=0)
chi2 = 0; df = 0
strong = []
for i in range(4):
    for j in range(4):
        if row[i] * col[j] > 0:
            exp = row[i] * col[j] / total
            z = (trans_m[i,j] - exp) / sqrt(exp)
            chi2 += (trans_m[i,j] - exp)**2 / exp
            df += 1
            if abs(z) > 2:
                strong.append((i, j, trans_m[i,j], z))
chi2_z = (chi2 - df) / sqrt(2*df)
print(f"  Chi-square: {chi2:.2f}  df={df}  z={chi2_z:+.2f}  ({time.time()-t0:.1f}s)")
for i, j, c, z in sorted(strong, key=lambda x: -abs(x[3]))[:5]:
    print(f"    {i} → {j}: count={c}  z={z:+.2f}")
    if abs(z) > 3:
        add_signal("sum_mod_mark", f"{i}→{j}", z, f"sum mod 4: {i}→{j}")

# ═══════════════════════════════════════════════════════════════════
# TEST 6: LONG-DISTANCE PAIR LAG (50, 100, 272)
# Top 30 pairs at very long lags
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] LONG-DISTANCE PAIR LAG (50, 100, 272)")
print("="*70)
t0 = time.time()
ld_sigs = []
for cnt, a, b in top_pairs[:50]:
    Y = (M[:, a] & M[:, b]).astype(np.float64)
    Y = Y - Y.mean()
    var = np.dot(Y, Y) / N
    if var == 0: continue
    for lag in [50, 100, 272]:
        cov = np.dot(Y[:-lag], Y[lag:]) / (N-lag)
        c = cov / var
        z = c * sqrt(N-lag)
        if abs(z) > 3.0:
            ld_sigs.append((a, b, lag, c, z))
ld_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(ld_sigs)} long-distance pair memories  ({time.time()-t0:.1f}s)")
for a, b, lag, c, z in ld_sigs[:10]:
    print(f"    ({a:>2},{b:>2}) lag={lag:>3}  acf={c:+.6f}  z={z:+.2f}")
    if abs(z) > 3:
        add_signal("pair_ld", f"({a},{b})@{lag}", z, f"pair ({a},{b}) lag {lag}")

# ═══════════════════════════════════════════════════════════════════
# TEST 7: LAST-K HOT PAIR → RECURRENCE
# Identify pairs hot in last 30 draws — do they recur?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[7] HOT-PAIR RECURRENCE (last 30 draws)")
print("="*70)
t0 = time.time()
W = 30
# For each draw i >= W, identify the most-frequent pair in last W draws
# Check if that pair appears in draw i
hot_hits = 0
hot_tests = 0
for i in range(W, N):
    # Find pair with max count in last W draws
    window = M[i-W:i].astype(np.int32)
    pair_w = window.T @ window
    np.fill_diagonal(pair_w, 0)  # exclude diagonal
    # Find argmax (skip n=0)
    idx_flat = int(np.argmax(pair_w[1:, 1:].flatten()))
    a = (idx_flat // 80) + 1
    b = (idx_flat % 80) + 1
    if a > b: a, b = b, a
    if a == b: continue
    if M[i, a] and M[i, b]:
        hot_hits += 1
    hot_tests += 1
# Expected: P(pair in random draw) = 20*19/(80*79) = 0.0601
exp_p = 20*19/(80*79)
exp_h = hot_tests * exp_p
var_h = hot_tests * exp_p * (1 - exp_p)
z = (hot_hits - exp_h) / sqrt(var_h)
print(f"  Tests: {hot_tests:,}  Hot pair recurs: {hot_hits:,} ({hot_hits/hot_tests*100:.2f}%)")
print(f"  Expected: {exp_h:,.0f}  z={z:+.2f}  ({time.time()-t0:.1f}s)")
if abs(z) > 2.5:
    add_signal("hot_pair", "recur", z, "hot pair recurrence")

# ═══════════════════════════════════════════════════════════════════
# TEST 8: NUMBER "LOYALTY" — first half vs second half frequency
# (Stability check)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[8] NUMBER LOYALTY — half-split stability")
print("="*70)
t0 = time.time()
mid = N // 2
freq_h1 = M[:mid].sum(axis=0)
freq_h2 = M[mid:].sum(axis=0)
exp_per_half = mid * p
# Z-score of difference
diff_z = []
for n in range(1, 81):
    diff = int(freq_h1[n]) - int(freq_h2[n])
    # SE of difference of two Binomial means
    se = sqrt(2 * exp_per_half * (1-p))
    z = diff / se
    diff_z.append((n, freq_h1[n], freq_h2[n], z))
diff_z.sort(key=lambda x: -abs(x[3]))
print(f"  Top 10 numbers with most DRIFT between halves  ({time.time()-t0:.1f}s)")
for n, h1, h2, z in diff_z[:10]:
    flag = " ★" if abs(z) > 3 else ""
    print(f"    #{n:>2}: H1={h1:,}  H2={h2:,}  z={z:+.2f}{flag}")
    if abs(z) > 3:
        add_signal("drift", f"n{n}", z, f"#{n} half-drift")

# ═══════════════════════════════════════════════════════════════════
# TEST 9: PAIR → TRIPLE (top pair → which triple in next?)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[9] PAIR → TRIPLE (sample)")
print("="*70)
t0 = time.time()
# For top 10 pairs, find indices, check top 100 triples in next draws
strongest_pairs = [(63,64), (53,80), (75,80), (50,67), (68,80), (26,79), (15,22), (71,74), (28,38), (41,55)]
# Need to count triples in next draws
p2t_sigs = []
for (a, b) in strongest_pairs[:5]:  # limit for speed
    mask = (M[:-1, a] == 1) & (M[:-1, b] == 1)
    idx = np.where(mask)[0]
    if len(idx) < 5000: continue
    # Count triples in next draws — too many; use itertools on each
    triple_count = Counter()
    for i_ in idx:
        nums = all_draws[i_+1][1]
        for i1 in range(20):
            for i2 in range(i1+1, 20):
                for i3 in range(i2+1, 20):
                    triple_count[(nums[i1], nums[i2], nums[i3])] += 1
    # Compare to expected
    exp_per_triple = len(idx) * 20*19*18/(80*79*78)
    var_per_triple = exp_per_triple * (1 - 20*19*18/(80*79*78))
    sigs_for_this = []
    for trip, cnt in triple_count.items():
        if cnt < exp_per_triple * 0.85 or cnt > exp_per_triple * 1.15:
            z = (cnt - exp_per_triple) / sqrt(var_per_triple)
            if abs(z) > 5:
                sigs_for_this.append((trip, cnt, z))
    sigs_for_this.sort(key=lambda x: -abs(x[2]))
    for trip, cnt, z in sigs_for_this[:2]:
        print(f"    pair {(a,b)} → triple {trip}: count={cnt} (exp {exp_per_triple:.1f}) z={z:+.2f}")
        if abs(z) > 5:
            p2t_sigs.append(((a,b), trip, z))
            add_signal("p2t", f"({a},{b})→{trip}", z, f"({a},{b}) → {trip}")
print(f"  Found {len(p2t_sigs)} pair→triple  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 10: HOUR + DELAY combined deeper (any delay, ANY hour)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[10] FINAL TEST: TRIPLE COMBINED (hour, delay-range, number)")
print("="*70)
t0 = time.time()
# Group delays into ranges: 1-3 (recent), 4-7 (medium), 8+ (overdue)
delay_arr = np.full((N, 81), 999, dtype=np.int32)
last_seen = np.full(81, -999, dtype=np.int32)
for i in range(N):
    for n in range(1, 81):
        if last_seen[n] >= 0:
            delay_arr[i, n] = i - last_seen[n]
    for n in all_draws[i][1]:
        last_seen[n] = i

triple_combo_sigs = []
for h in range(24):
    h_mask = hours == h
    idx_h = np.where(h_mask)[0]
    if len(idx_h) < 5000: continue
    for d_range_name, d_lo, d_hi in [('recent', 1, 3), ('medium', 4, 7), ('overdue', 8, 20)]:
        for n in range(1, 81):
            mask_dn = (delay_arr[idx_h, n] >= d_lo) & (delay_arr[idx_h, n] <= d_hi)
            total = int(mask_dn.sum())
            if total < 500: continue
            appeared = int(M[idx_h[mask_dn], n].sum())
            p_c = appeared / total
            z = (p_c - p) / sqrt(p*(1-p)/total)
            if abs(z) > 3.5:
                triple_combo_sigs.append((h, d_range_name, n, total, p_c, z))
triple_combo_sigs.sort(key=lambda x: -abs(x[5]))
print(f"  Found {len(triple_combo_sigs)} (h, d_range, n) signals  ({time.time()-t0:.1f}s)")
for h, dr, n, T, pc, z in triple_combo_sigs[:15]:
    print(f"    h={h:>2} d={dr:>7} #{n:>2}: P={pc:.4f} T={T} z={z:+.2f}")
    if abs(z) > 4:
        add_signal("hxdrng_n", f"h{h}_{dr}_n{n}", z, f"h={h} {dr} #{n}")

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ — {len(all_signals)} νέα σήματα (9η ομάδα)")
print("="*70)
all_signals.sort(key=lambda x: -abs(x[2]))
print(f"\n  Top 30:")
for i, (cat, name, z, det) in enumerate(all_signals[:30], 1):
    print(f"  {i:>3}.  [{cat:>14}]  {name:>22}  |z|={abs(z):>6.2f}  {'+' if z>0 else '-'}")

print(f"\n  Σύνοψη ανά κατηγορία:")
cats = defaultdict(list)
for cat, name, z, det in all_signals:
    cats[cat].append(abs(z))
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"    {cat:>14}: {len(arr):>3} σήματα, max |z|={max(arr):.2f}")

out = {'signals': [(c, n, float(z), d) for c, n, z, d in all_signals], 'N': N}
with open('/home/user/Game/ninth_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: ninth_signals.json ({len(all_signals)} σήματα)")
print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
