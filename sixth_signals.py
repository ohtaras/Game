#!/usr/bin/env python3
"""
6η ομάδα tests — pair × time, multi-lag correlations, deeper sequential patterns.

Tests:
 1. Pair frequency × hour (does pair bias depend on time?)
 2. Pair frequency × dow
 3. Pair lag co-occurrence (pair at lag 1, 2, 5)
 4. 4-back Markov per number
 5. Run alternation per number
 6. Gap variance per number (conditional on history)
 7. Pair → number at LAG 2 (not immediate)
 8. Number runs (length of consecutive appearances/absences)
 9. Most-frequent partner per number (who does each number go with?)
10. Reverse conditional: P(a in i-1 | a in i)
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
dows = np.array([draw_time(i).weekday() for i in range(N)], dtype=np.int8)

# ═══════════════════════════════════════════════════════════════════
# TEST 1: PAIR × HOUR
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] PAIR × HOUR (does pair frequency depend on hour?)")
print("="*70)
t0 = time.time()
# For each hour, compute pair count matrix and find anomalous pairs
M32 = M.astype(np.int32)
pair_hour_sigs = []
for h in range(24):
    mask = hours == h
    T = int(mask.sum())
    if T < 5000: continue
    M_h = M32[mask]  # (T, 81)
    pair_h = M_h.T @ M_h  # 81x81
    exp_pair_h = T * 20*19/(80*79)
    var_h = exp_pair_h * (1 - 20*19/(80*79))
    for a in range(1, 81):
        for b in range(a+1, 81):
            cnt = int(pair_h[a, b])
            z = (cnt - exp_pair_h) / sqrt(var_h)
            if abs(z) > 4.0:
                pair_hour_sigs.append((h, a, b, cnt, z))
pair_hour_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(pair_hour_sigs)} (h, pair) cells with |z|>4.0  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for h, a, b, cnt, z in pair_hour_sigs[:15]:
    print(f"    h={h:>2}  ({a:>2},{b:>2})  count={cnt}  z={z:+.2f}")
    if abs(z) > 4:
        add_signal("pair_h", f"h{h}_({a},{b})", z, f"pair ({a},{b}) at h={h}")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: PAIR × DOW
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] PAIR × DOW")
print("="*70)
t0 = time.time()
DOW = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
pair_dow_sigs = []
for d_ in range(7):
    mask = dows == d_
    T = int(mask.sum())
    if T < 10000: continue
    M_d = M32[mask]
    pair_d = M_d.T @ M_d
    exp_pair_d = T * 20*19/(80*79)
    var_d = exp_pair_d * (1 - 20*19/(80*79))
    for a in range(1, 81):
        for b in range(a+1, 81):
            cnt = int(pair_d[a, b])
            z = (cnt - exp_pair_d) / sqrt(var_d)
            if abs(z) > 4.0:
                pair_dow_sigs.append((d_, a, b, cnt, z))
pair_dow_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(pair_dow_sigs)} (dow, pair) with |z|>4.0  ({time.time()-t0:.1f}s)")
print(f"  Top 10:")
for d_, a, b, cnt, z in pair_dow_sigs[:10]:
    print(f"    {DOW[d_]}  ({a:>2},{b:>2})  count={cnt}  z={z:+.2f}")
    if abs(z) > 4:
        add_signal("pair_dow", f"{DOW[d_]}_({a},{b})", z, f"pair ({a},{b}) on {DOW[d_]}")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: PAIR co-occurrence at LAG (does pair (a,b) at draw i correlate with pair at i+k?)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] PAIR co-occurrence at LAG k (does pair (a,b) cluster in time?)")
print("="*70)
t0 = time.time()
# For top 30 most frequent pairs, autocorrelation at lag 1, 2, 5
top_pairs = []
for a in range(1, 81):
    for b in range(a+1, 81):
        cnt = int((M[:, a] & M[:, b]).sum())
        top_pairs.append((cnt, a, b))
top_pairs.sort(reverse=True)
test_pairs = [(a, b) for _, a, b in top_pairs[:50]]

pair_lag_sigs = []
for (a, b) in test_pairs:
    Y = (M[:, a].astype(np.float64) * M[:, b].astype(np.float64))
    Y = Y - Y.mean()
    var = np.dot(Y, Y) / N
    if var == 0: continue
    for lag in [1, 2, 5]:
        cov = np.dot(Y[:-lag], Y[lag:]) / (N-lag)
        c = cov / var
        z = c * sqrt(N-lag)
        if abs(z) > 3:
            pair_lag_sigs.append((a, b, lag, c, z))
pair_lag_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Tested {len(test_pairs)} pairs  ({time.time()-t0:.1f}s)")
print(f"  Top pair memory:")
for a, b, lag, c, z in pair_lag_sigs[:10]:
    print(f"    ({a:>2},{b:>2}) lag={lag}  acf={c:+.6f}  z={z:+.2f}")
    if abs(z) > 3:
        add_signal("pair_lag", f"({a},{b})@{lag}", z, f"pair ({a},{b}) lag {lag}")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: 4-BACK MARKOV
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4] 4-BACK MARKOV per number")
print("="*70)
t0 = time.time()
results = []
for n in range(1, 81):
    in_4 = M[:-4, n] == 1
    in_3 = M[1:-3, n] == 1
    in_2 = M[2:-2, n] == 1
    in_1 = M[3:-1, n] == 1
    in_0 = M[4:, n] == 1
    cond = in_4 & in_3 & in_2 & in_1
    cnt_c = int(cond.sum())
    if cnt_c < 50: continue
    cnt_cont = int((cond & in_0).sum())
    p_c = cnt_cont / cnt_c
    z = (p_c - p) / sqrt(p*(1-p)/cnt_c)
    results.append((n, cnt_c, p_c, z))
results.sort(key=lambda x: -abs(x[3]))
print(f"  Top 10  ({time.time()-t0:.1f}s)")
for n, c, pc, z in results[:10]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    #{n:>2}: P={pc:.4f}  n={c:,}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("4back", f"n{n}", z, f"#{n} 4-back")

# ═══════════════════════════════════════════════════════════════════
# TEST 5: RUN ALTERNATION per number
# Length of consecutive presence/absence runs
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[5] RUN ALTERNATION per number")
print("="*70)
t0 = time.time()
alt_results = []
for n in range(1, 81):
    arr = M[:, n]
    # Count transitions
    trans = int((arr[1:] != arr[:-1]).sum())
    # Expected: 2*N*p*(1-p) = 2*238855*0.1875 = 89,571
    exp_trans = 2 * N * p * (1-p)
    var_trans = 4 * N * p * (1-p) * (1 - 3*p*(1-p))
    z = (trans - exp_trans) / sqrt(var_trans)
    alt_results.append((n, trans, z))
alt_results.sort(key=lambda x: -abs(x[2]))
print(f"  Top 10 numbers with run-alternation bias  ({time.time()-t0:.1f}s)")
for n, t_, z in alt_results[:10]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    #{n:>2}: transitions={t_:,}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("alt", f"n{n}", z, f"#{n} alternation")

# ═══════════════════════════════════════════════════════════════════
# TEST 6: GAP VARIANCE conditional on number's recent activity
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] GAP STD per number (variance ratio)")
print("="*70)
t0 = time.time()
gap_std_results = []
for n in range(1, 81):
    idx = np.where(M[:, n] == 1)[0]
    if len(idx) < 100: continue
    gaps = np.diff(idx)
    sigma = gaps.std()
    mu = gaps.mean()
    # For geometric distribution with p=0.25: σ² = (1-p)/p² = 12; σ ≈ 3.46
    # Theory: σ for waiting time ≈ sqrt(1-p)/p ≈ 3.46
    exp_sigma = sqrt((1-p)/(p**2))  # = 3.46
    # z = (σ_obs - σ_exp) / SE; SE for std of K samples ≈ σ/sqrt(2K)
    se = exp_sigma / sqrt(2 * len(gaps))
    z = (sigma - exp_sigma) / se
    gap_std_results.append((n, sigma, z))
gap_std_results.sort(key=lambda x: -abs(x[2]))
print(f"  Top 10 anomalous gap-std  ({time.time()-t0:.1f}s)")
for n, s, z in gap_std_results[:10]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    #{n:>2}: gap std={s:.4f} (exp 3.46)  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("gap_std", f"n{n}", z, f"#{n} gap std")

# ═══════════════════════════════════════════════════════════════════
# TEST 7: PAIR → number at LAG 2 (not immediate next)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[7] PAIR → number at LAG 2 (skip one draw)")
print("="*70)
t0 = time.time()
# Build pair occurrence indices
pair_idx = defaultdict(list)
for i in range(N-2):
    nums = all_draws[i][1]
    for a_ in range(20):
        for b_ in range(a_+1, 20):
            pair_idx[(nums[a_], nums[b_])].append(i)
top_pairs2 = [(p_, idx) for p_, idx in pair_idx.items() if len(idx) > 12000]
print(f"  {len(top_pairs2)} pairs with >12K samples  ({time.time()-t0:.1f}s)")
p2n_lag2 = []
for (a, b), idx_list in top_pairs2:
    next2_indices = np.array([i+2 for i in idx_list if i+2 < N], dtype=np.int32)
    if len(next2_indices) < 12000: continue
    next2_M = M[next2_indices]
    counts = next2_M.sum(axis=0)
    for c_ in range(1, 81):
        if c_ in (a, b): continue
        cnt = int(counts[c_])
        exp = len(next2_indices) * p
        var = len(next2_indices) * p * (1-p)
        z = (cnt - exp) / sqrt(var)
        if abs(z) > 4.0:
            p2n_lag2.append(((a,b), c_, cnt, len(next2_indices), z))
p2n_lag2.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(p2n_lag2)} pair→number@lag2 with |z|>4.0  ({time.time()-t0:.1f}s)")
print(f"  Top 10:")
for (a,b), c_, cnt, t_, z in p2n_lag2[:10]:
    pc = cnt/t_
    print(f"    ({a:>2},{b:>2}) →lag2→ {c_:>2}  P={pc:.4f}  z={z:+.2f}")
    if abs(z) > 4:
        add_signal("p2n_lag2", f"({a},{b})→{c_}", z, f"pair ({a},{b}) → {c_} lag 2")

# ═══════════════════════════════════════════════════════════════════
# TEST 8: NUMBER RUNS — longest absence per number
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[8] NUMBER LONGEST ABSENCE")
print("="*70)
t0 = time.time()
max_absence = []
for n in range(1, 81):
    arr = M[:, n]
    cur = 0; mx = 0
    for b in arr:
        if not b: cur += 1; mx = max(mx, cur)
        else: cur = 0
    max_absence.append((n, mx))
max_absence.sort(key=lambda x: -x[1])
# For geometric P(absence > k) ≈ (1-p)^k. Expected max ≈ ln(N)/ln(1/(1-p))
expected_max_abs = log(N) / log(1/(1-p))
print(f"  Expected max absence: ~{expected_max_abs:.1f}")
print(f"  Top 10 absences:")
for n, mx in max_absence[:10]:
    p_extreme = N * (0.75**mx)
    z_approx = -log(max(p_extreme, 1e-300)) / log(10)
    print(f"    #{n:>2}: max absence = {mx}  ≈ z {z_approx:.1f}")
    if p_extreme < 0.001:
        add_signal("absence", f"n{n}", z_approx, f"#{n} max absence {mx}")
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 9: MOST-FREQUENT PARTNER per number
# For each n, who is its strongest co-occurrence partner?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[9] FAVOURITE PARTNER per number")
print("="*70)
t0 = time.time()
pair_total = M32.T @ M32
exp_pair_g = N * 20*19/(80*79)
favs = []
for n in range(1, 81):
    # Find partner with largest deviation (positive)
    best_z = 0
    best_partner = None
    for m in range(1, 81):
        if m == n: continue
        cnt = int(pair_total[n, m])
        z = (cnt - exp_pair_g) / sqrt(exp_pair_g)
        if abs(z) > abs(best_z):
            best_z = z; best_partner = m
    favs.append((n, best_partner, best_z))
favs.sort(key=lambda x: -abs(x[2]))
print(f"  Top 15 'soulmates'  ({time.time()-t0:.1f}s)")
for n, m, z in favs[:15]:
    flag = " ★" if abs(z) > 3.5 else ""
    print(f"    #{n:>2} ↔ #{m:>2}  z={z:+.2f}{flag}")
    if abs(z) > 3.5:
        add_signal("partner", f"n{n}↔{m}", z, f"#{n} partner #{m}")

# ═══════════════════════════════════════════════════════════════════
# TEST 10: REVERSE conditional — P(a in i-1 | a in i)
# Should equal P(a in i | a in i-1) by symmetry but check
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[10] REVERSE conditional P(n_{i-1} | n_i)")
print("="*70)
t0 = time.time()
rev = []
for n in range(1, 81):
    in_i = M[1:, n] == 1
    in_prev = M[:-1, n] == 1
    cnt_i = int(in_i.sum())
    if cnt_i == 0: continue
    cnt_both = int((in_i & in_prev).sum())
    p_rev = cnt_both / cnt_i
    z = (p_rev - p) / sqrt(p*(1-p)/cnt_i)
    rev.append((n, p_rev, z))
rev.sort(key=lambda x: -abs(x[2]))
print(f"  Top 5  ({time.time()-t0:.1f}s)")
for n, pc, z in rev[:5]:
    flag = " ✓" if abs(z) > 2 else ""
    print(f"    #{n:>2}: P(prev | current) = {pc:.4f}  z={z:+.2f}{flag}")

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ — {len(all_signals)} νέα σήματα (6η ομάδα)")
print("="*70)
all_signals.sort(key=lambda x: -abs(x[2]))
print(f"\n  Top 30:")
for i, (cat, name, z, det) in enumerate(all_signals[:30], 1):
    print(f"  {i:>3}.  [{cat:>12}]  {name:>25}  |z|={abs(z):>6.2f}  {'+' if z>0 else '-'}")

print(f"\n  Σύνοψη ανά κατηγορία:")
cats = defaultdict(list)
for cat, name, z, det in all_signals:
    cats[cat].append(abs(z))
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"    {cat:>14}: {len(arr):>3} σήματα, max |z|={max(arr):.2f}")

out = {'signals': [(c, n, float(z), d) for c, n, z, d in all_signals], 'N': N}
with open('/home/user/Game/sixth_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: sixth_signals.json ({len(all_signals)} σήματα)")
print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
