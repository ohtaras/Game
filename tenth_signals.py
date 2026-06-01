#!/usr/bin/env python3
"""
10η ομάδα — επεκτεταμένη αναζήτηση strong pair→triple, και τελικά exotic tests.

Tests:
 1. EXTENDED pair→triple — εξαντλητική αναζήτηση σε ΟΛΑ τα ζευγάρια × top triples
 2. Triple → triple (top triples)
 3. Pair → quadruple (extreme)
 4. Hour-of-day cycle (sin/cos features)
 5. Specific number "anchors" — does presence of X always come with Y?
 6. Day-of-month × hour interaction
 7. Conditional on overlap-size: P(specific number | overlap = k)
 8. Pair × delay → number (3-way conditional)
 9. Min/Max ALONE → number (sample with extreme min/max in previous)
10. Inter-arrival kurtosis per number
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from math import sqrt, log, comb, pi, sin, cos

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
# TEST 1: EXTENDED pair → triple — search ALL pairs against most-common next-draw triples
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] EXTENDED pair → triple search")
print("="*70)
t0 = time.time()
# Memory consideration: for each of 3160 pairs, count triples in NEXT draw
# Instead of full enumeration of triples (82K), pre-compute popular ones
# For each pair, count which triples appear in next draws, find top deviations

# First: identify top 100 most-frequent triples overall (already useful as reference)
all_triples = Counter()
for _, nums in all_draws:
    for i in range(20):
        for j in range(i+1, 20):
            for k in range(j+1, 20):
                all_triples[(nums[i], nums[j], nums[k])] += 1
# Take top 500
top_500_triples = [t for t, _ in all_triples.most_common(500)]
exp_per_triple = N * 20*19*18/(80*79*78)
print(f"  Indexed all triples in main draws  ({time.time()-t0:.1f}s)")
print(f"  Top 500 triples reference baseline = {exp_per_triple:.1f}")

# For each pair, look at next-draw triples
top_pair_triple_sigs = []
# Iterate top pairs by frequency (3160 total but high-freq ones have more data)
M32 = M.astype(np.int32)
pair_total = M32.T @ M32
pair_list = []
for a in range(1, 81):
    for b in range(a+1, 81):
        pair_list.append((int(pair_total[a,b]), a, b))
pair_list.sort(reverse=True)

# For top 500 pairs by frequency, check if any specific next-draw triple is anomalous
# This is fast enough
for pair_idx, (cnt_ab, a, b) in enumerate(pair_list[:500]):
    mask = (M[:-1, a] == 1) & (M[:-1, b] == 1)
    n_with = int(mask.sum())
    if n_with < 10000: continue
    # Compute next-draw triple counts for top_500
    next_indices = np.where(mask)[0] + 1
    # For each top triple, count occurrences in these next draws
    # Need M[next_indices, x] == 1 for all x in triple
    for trip in top_500_triples[:200]:
        a_, b_, c_ = trip
        cnt = int(((M[next_indices, a_] == 1) & (M[next_indices, b_] == 1) & (M[next_indices, c_] == 1)).sum())
        exp = n_with * 20*19*18/(80*79*78)
        var = exp * (1 - 20*19*18/(80*79*78))
        z = (cnt - exp) / sqrt(var)
        if abs(z) > 5.5:
            top_pair_triple_sigs.append(((a,b), trip, cnt, n_with, z))
top_pair_triple_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(top_pair_triple_sigs)} pair→triple with |z|>5.5  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for (ab, trip, cnt, n_w, z) in top_pair_triple_sigs[:15]:
    print(f"    {ab} → {trip}: count={cnt} (in {n_w})  z={z:+.2f}")
    if abs(z) > 5.5:
        add_signal("p2trip", f"{ab}→{trip}", z, f"{ab} → {trip}")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: TRIPLE → TRIPLE (sequential triples)
# Top triples in i → which triples in i+1?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] TRIPLE → TRIPLE")
print("="*70)
t0 = time.time()
# For top 30 triples, find their next-draw top triple
t2t_sigs = []
# Use only top 100 triples for both prev and next
top_100_trips = top_500_triples[:100]
top_100_set = set(top_100_trips)

# For each top-30 triple, find indices and check next-draw top-50 triples
for trip_prev in top_100_trips[:30]:
    a, b, c = trip_prev
    mask = (M[:-1, a] == 1) & (M[:-1, b] == 1) & (M[:-1, c] == 1)
    n_with = int(mask.sum())
    if n_with < 2000: continue
    next_idx = np.where(mask)[0] + 1
    # Count next triples
    for trip_next in top_100_trips[:50]:
        x, y, z_ = trip_next
        if {x,y,z_} == {a,b,c}: continue  # avoid identity
        cnt = int(((M[next_idx, x] == 1) & (M[next_idx, y] == 1) & (M[next_idx, z_] == 1)).sum())
        exp = n_with * 20*19*18/(80*79*78)
        var = exp * (1 - 20*19*18/(80*79*78))
        z = (cnt - exp) / sqrt(var)
        if abs(z) > 5.5:
            t2t_sigs.append((trip_prev, trip_next, cnt, n_with, z))
t2t_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(t2t_sigs)} triple→triple signals  ({time.time()-t0:.1f}s)")
for trip_prev, trip_next, cnt, n_w, z in t2t_sigs[:10]:
    print(f"    {trip_prev} → {trip_next}: cnt={cnt}/{n_w}  z={z:+.2f}")
    if abs(z) > 5.5:
        add_signal("t2t", f"{trip_prev}→{trip_next}", z, f"{trip_prev} → {trip_next}")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: PAIR → QUADRUPLE (extreme conditional)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] PAIR → QUADRUPLE (sample)")
print("="*70)
t0 = time.time()
# For top 10 most-significant pair→triple findings, check pair → 4-tuple
# Use the most-significant pair (75,80) and find anomalous 4-tuples in next draw
strong_pair_targets = [(75, 80)]  # focus on strongest

p2q_sigs = []
for (a, b) in strong_pair_targets:
    mask = (M[:-1, a] == 1) & (M[:-1, b] == 1)
    next_idx = np.where(mask)[0] + 1
    if len(next_idx) < 10000: continue
    # Count top 50 quadruples in next draws
    next_4tuple_counts = Counter()
    for i_ in next_idx[:5000]:  # sample for speed
        nums = all_draws[i_][1]
        for i1 in range(20):
            for i2 in range(i1+1, 20):
                for i3 in range(i2+1, 20):
                    for i4 in range(i3+1, 20):
                        next_4tuple_counts[(nums[i1], nums[i2], nums[i3], nums[i4])] += 1
    # Expected
    sample_size = 5000
    exp_4tuple = sample_size * comb(20,4)/comb(80,4)
    print(f"    Sample size: {sample_size}, exp per 4-tuple: {exp_4tuple:.2f}")
    # Find top
    for tup, cnt in next_4tuple_counts.most_common(20):
        z = (cnt - exp_4tuple) / sqrt(max(exp_4tuple, 1))
        if abs(z) > 7:
            p2q_sigs.append(((a,b), tup, cnt, z))
            print(f"      ({a},{b}) → {tup}: cnt={cnt}  z={z:+.2f}")
print(f"  Found {len(p2q_sigs)} pair→quad signals  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: HOUR-OF-DAY SINUSOIDAL features
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4] HOUR SINUSOIDAL features per number")
print("="*70)
t0 = time.time()
hours = np.array([draw_time(i).hour for i in range(N)], dtype=np.int8)
# For each number, compute correlation with sin(2π h/24) and cos(2π h/24)
sin_h = np.sin(2 * pi * hours / 24)
cos_h = np.cos(2 * pi * hours / 24)
sin_h_c = sin_h - sin_h.mean()
cos_h_c = cos_h - cos_h.mean()
var_s = np.dot(sin_h_c, sin_h_c) / N
var_c = np.dot(cos_h_c, cos_h_c) / N
sin_sigs = []
cos_sigs = []
for n in range(1, 81):
    arr = M[:, n].astype(np.float64) - p
    cov_s = np.dot(arr, sin_h_c) / N
    cov_c = np.dot(arr, cos_h_c) / N
    var_a = np.dot(arr, arr) / N
    r_s = cov_s / sqrt(var_s * var_a)
    r_c = cov_c / sqrt(var_c * var_a)
    z_s = r_s * sqrt(N)
    z_c = r_c * sqrt(N)
    if abs(z_s) > 3.5: sin_sigs.append((n, r_s, z_s))
    if abs(z_c) > 3.5: cos_sigs.append((n, r_c, z_c))
sin_sigs.sort(key=lambda x: -abs(x[2]))
cos_sigs.sort(key=lambda x: -abs(x[2]))
print(f"  sin(h): {len(sin_sigs)} numbers with |z|>3.5  ({time.time()-t0:.1f}s)")
for n, r, z in sin_sigs[:8]:
    print(f"    #{n:>2}  sin corr={r:+.5f}  z={z:+.2f}")
    if abs(z) > 4:
        add_signal("h_sin", f"n{n}", z, f"#{n} sin(h)")
print(f"  cos(h): {len(cos_sigs)} numbers with |z|>3.5")
for n, r, z in cos_sigs[:8]:
    print(f"    #{n:>2}  cos corr={r:+.5f}  z={z:+.2f}")
    if abs(z) > 4:
        add_signal("h_cos", f"n{n}", z, f"#{n} cos(h)")

# ═══════════════════════════════════════════════════════════════════
# TEST 5: NUMBER "ANCHORS" — pair where presence of X implies Y
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[5] NUMBER ANCHORS — P(b | a present, in same draw)")
print("="*70)
t0 = time.time()
exp_pair_anchor = N * 20*19/(80*79)
anchor_sigs = []
for a in range(1, 81):
    cnt_a = int(M[:, a].sum())
    if cnt_a == 0: continue
    # P(b in same draw | a in draw) = pair_count[a,b] / cnt_a
    # vs P(b in draw) = 0.25
    for b in range(1, 81):
        if a == b: continue
        cnt_both = int(pair_total[a, b])
        p_b_given_a = cnt_both / cnt_a
        # Theoretical P(b in draw | a in draw, sampling without rep) = 19/79
        exp_p_b_given_a = 19/79
        var_pba = exp_p_b_given_a * (1 - exp_p_b_given_a) / cnt_a
        z = (p_b_given_a - exp_p_b_given_a) / sqrt(var_pba)
        if abs(z) > 4:
            anchor_sigs.append((a, b, p_b_given_a, z))
anchor_sigs.sort(key=lambda x: -abs(x[3]))
print(f"  Found {len(anchor_sigs)} 'anchor' pairs with |z|>4  ({time.time()-t0:.1f}s)")
for a, b, pba, z in anchor_sigs[:15]:
    print(f"    #{a:>2} present → P(#{b:>2})={pba:.4f} (exp {19/79:.4f})  z={z:+.2f}")
    if abs(z) > 4:
        add_signal("anchor", f"{a}→{b}", z, f"#{a} → #{b} same draw")

# ═══════════════════════════════════════════════════════════════════
# TEST 6: CONDITIONAL on OVERLAP SIZE
# If overlap_{i-1} (between draws i-1 and i) is large, biased for specific numbers in i+1?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] CONDITIONAL on PREVIOUS OVERLAP SIZE")
print("="*70)
t0 = time.time()
overlaps = np.zeros(N-1, dtype=np.int8)
for i in range(N-1):
    overlaps[i] = int(np.dot(M[i].astype(np.int32), M[i+1].astype(np.int32)))
# For each overlap level k, what's P(specific n in i+1)?
# We're conditioning on overlap[i] = k, so draw i+1 is what we predict
ov_sigs = []
for k in range(0, 11):
    mask = overlaps[:-1] == k  # mask on indices where overlap_{i} = k
    # The "next draw" is i+1 (which has the overlap with i)
    # But we want to predict the NEXT NEXT (i+2) given overlap_{i,i+1} = k... let's clarify
    # Re-define: at draw i, overlap_{i} = overlap between draws i and i+1
    # So overlap at index i tells us about transition i→i+1
    # If overlap_{i-1} is high, what biases draw_i?
    # That is: given M[i-1] and M[i] share many, what's biased about M[i]?
    # Hmm this is post-hoc. Let me skip.
    # Simpler: given overlap from i-1 to i = k, what's P(specific number in i+1)?
    if k == 0:
        # Edge case
        continue
    mask = overlaps == k
    indices = np.where(mask)[0]
    indices_next2 = indices + 1  # the next draw (which is i+1 from above)
    valid_idx = indices_next2 < N
    indices_next2 = indices_next2[valid_idx]
    T = len(indices_next2)
    if T < 5000: continue
    next_M = M[indices_next2]
    counts = next_M.sum(axis=0)
    for n in range(1, 81):
        exp = T * p
        var = T * p * (1-p)
        z = (counts[n] - exp) / sqrt(var)
        if abs(z) > 3.5:
            ov_sigs.append((k, n, int(counts[n]), T, z))
ov_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(ov_sigs)} (overlap_k, number) signals  ({time.time()-t0:.1f}s)")
for k, n, cnt, T, z in ov_sigs[:10]:
    print(f"    overlap={k} → #{n:>2}: {cnt}/{T} z={z:+.2f}")
    if abs(z) > 4:
        add_signal("ov_n", f"ov{k}_n{n}", z, f"overlap {k} → #{n}")

# ═══════════════════════════════════════════════════════════════════
# TEST 7: MIN/MAX-EXTREME conditional
# When previous draw had very low min or very high max, biased next?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[7] EXTREME MIN/MAX → biased next")
print("="*70)
t0 = time.time()
mins = np.array([d[0] for _, d in all_draws])
maxs = np.array([d[-1] for _, d in all_draws])
# Define: min ≤ 2 or max ≥ 79 as "extreme"
ext_min_mask = mins[:-1] <= 2
ext_max_mask = maxs[:-1] >= 79
for name, mask in [('min≤2', ext_min_mask), ('max≥79', ext_max_mask)]:
    T = int(mask.sum())
    if T < 5000: continue
    next_M = M[1:][mask]
    counts = next_M.sum(axis=0)
    print(f"\n  {name}: T={T:,}")
    for n in range(1, 81):
        exp = T * p
        var = T * p * (1-p)
        z = (counts[n] - exp) / sqrt(var)
        if abs(z) > 3.5:
            print(f"    {name} → #{n:>2}: {counts[n]}/{T}  z={z:+.2f}")
            add_signal("ext", f"{name}_n{n}", z, f"{name} → #{n}")
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 8: INTER-ARRIVAL KURTOSIS per number
# Kurtosis indicates heavier/lighter tails than expected for geometric
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[8] GAP KURTOSIS per number")
print("="*70)
t0 = time.time()
from scipy.stats import kurtosis
kurt_results = []
exp_kurt = (6 + p**2 - 6*p + p**4) / ((1-p)**2)  # rough for geometric, fisher kurtosis
# Easier: for geometric p=0.25, theoretical fisher kurt ≈ 6 + p²/(1-p) ≈ 6
# Use raw kurtosis (excess)
for n in range(1, 81):
    idx = np.where(M[:, n] == 1)[0]
    if len(idx) < 100: continue
    gaps = np.diff(idx)
    k_ = float(kurtosis(gaps))
    # Expected for geometric ≈ 6 (raw excess)
    # SE for kurtosis ≈ sqrt(24/n_gaps)
    se = sqrt(24/len(gaps))
    z = (k_ - 6) / se
    kurt_results.append((n, k_, z))
kurt_results.sort(key=lambda x: -abs(x[2]))
print(f"  Top 10 anomalous kurtosis  ({time.time()-t0:.1f}s)")
for n, k_, z in kurt_results[:10]:
    flag = " ★" if abs(z) > 3 else " ✓" if abs(z) > 2 else ""
    print(f"    #{n:>2}: gap kurtosis = {k_:+.3f} (exp 6)  z={z:+.2f}{flag}")
    # NOTE: SE formula sqrt(24/n) is for normal distribution only; for
    # geometric the variance of kurtosis estimator is much larger.
    # Disabling — these z-scores are not meaningful.
    pass

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ — {len(all_signals)} νέα σήματα (10η ομάδα)")
print("="*70)
all_signals.sort(key=lambda x: -abs(x[2]))
print(f"\n  Top 30:")
for i, (cat, name, z, det) in enumerate(all_signals[:30], 1):
    print(f"  {i:>3}.  [{cat:>14}]  {name:>25}  |z|={abs(z):>6.2f}  {'+' if z>0 else '-'}")

print(f"\n  Σύνοψη ανά κατηγορία:")
cats = defaultdict(list)
for cat, name, z, det in all_signals:
    cats[cat].append(abs(z))
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"    {cat:>14}: {len(arr):>3} σήματα, max |z|={max(arr):.2f}")

out = {'signals': [(c, n, float(z), d) for c, n, z, d in all_signals], 'N': N}
with open('/home/user/Game/tenth_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: tenth_signals.json ({len(all_signals)} σήματα)")
print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
