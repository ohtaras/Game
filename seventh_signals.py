#!/usr/bin/env python3
"""
7η ομάδα — combined conditional signals.

Tests:
 1. Hour-specific n→n Markov: P(n in i+1 | n in i AND hour=h)
 2. Triple → number (top triples by frequency)
 3. Sum-conditional number bias (next number given last draw sum bin)
 4. Pair → number at lag 3 and 5 (deeper temporal pattern)
 5. Conditional delay × hour fine-grained
 6. Number transitions on specific weekdays (Mon-Sun)
 7. Pair-frequency at lag 10+
 8. Triple → number at lag 2
 9. Sum bin → which pair more likely?
10. Window of 5 draws — collective signal
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
# TEST 1: HOUR-SPECIFIC n→n MARKOV
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] HOUR-SPECIFIC n→n Markov")
print("="*70)
t0 = time.time()
# For each (n, h): P(n in i+1 | n in i AND hour_i = h)
hxn_markov = []
for h in range(24):
    h_mask = hours[:-1] == h
    for n in range(1, 81):
        in_i = (M[:-1, n] == 1) & h_mask
        cnt_i = int(in_i.sum())
        if cnt_i < 500: continue
        in_next = M[1:, n][in_i] == 1
        cnt_next = int(in_next.sum())
        p_c = cnt_next / cnt_i
        z = (p_c - p) / sqrt(p*(1-p)/cnt_i)
        if abs(z) > 3.0:
            hxn_markov.append((h, n, cnt_i, p_c, z))
hxn_markov.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(hxn_markov)} (h, n) Markov bias with |z|>3.0  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for h, n, c, pc, z in hxn_markov[:15]:
    print(f"    h={h:>2} #{n:>2}: P(self continue)={pc:.4f}  n={c:,}  z={z:+.2f}")
    if abs(z) > 3.5:
        add_signal("hxn_mark", f"h{h}n{n}", z, f"h={h} #{n} self-continue")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: TRIPLE → NUMBER (top triples)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] TRIPLE → NUMBER")
print("="*70)
t0 = time.time()
# Build triple occurrences for top triples by frequency
triple_indices = defaultdict(list)
print(f"  Indexing triples (this is slow)...")
for i in range(N-1):
    nums = all_draws[i][1]
    for a_ in range(20):
        for b_ in range(a_+1, 20):
            for c_ in range(b_+1, 20):
                triple_indices[(nums[a_], nums[b_], nums[c_])].append(i)
print(f"  Indexed {len(triple_indices)} unique triples  ({time.time()-t0:.1f}s)")
# Focus on top 200 most frequent
top_triples = [(p_, idx) for p_, idx in triple_indices.items() if len(idx) > 3500]
top_triples.sort(key=lambda x: -len(x[1]))
top_triples = top_triples[:500]
print(f"  Testing top {len(top_triples)} triples with most data  ({time.time()-t0:.1f}s)")

t2n_sigs = []
for triple, idx_list in top_triples:
    next_indices = np.array(idx_list, dtype=np.int32)
    next_indices = next_indices[next_indices < N-1]
    if len(next_indices) < 3000: continue
    next_M = M[next_indices + 1]
    counts = next_M.sum(axis=0)
    for c_ in range(1, 81):
        if c_ in triple: continue
        cnt = int(counts[c_])
        exp = len(next_indices) * p
        var = len(next_indices) * p * (1-p)
        z = (cnt - exp) / sqrt(var)
        if abs(z) > 4.5:
            t2n_sigs.append((triple, c_, cnt, len(next_indices), z))
t2n_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(t2n_sigs)} triple→number signals with |z|>4.5  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for trip, c_, cnt, t_, z in t2n_sigs[:15]:
    pc = cnt/t_
    print(f"    {trip} → {c_:>2}  P={pc:.4f}  n={t_}  z={z:+.2f}")
    if abs(z) > 4.5:
        add_signal("t2n", f"{trip}→{c_}", z, f"triple {trip} → {c_}")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: SUM-CONDITIONAL NUMBER BIAS
# Given previous draw's sum in bin X, which numbers more likely next?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] SUM-CONDITIONAL NUMBER BIAS")
print("="*70)
t0 = time.time()
sums = np.array([sum(d) for _, d in all_draws], dtype=np.int32)
bin_edges = np.percentile(sums, np.linspace(0, 100, 11))
bins = np.digitize(sums, bin_edges) - 1
bins = np.clip(bins, 0, 9)  # 10 bins

sum_cond_sigs = []
for b in range(10):
    mask = bins[:-1] == b
    T = int(mask.sum())
    if T < 5000: continue
    M_next = M[1:][mask]
    counts = M_next.sum(axis=0)
    for n in range(1, 81):
        exp = T * p
        var = T * p * (1-p)
        z = (counts[n] - exp) / sqrt(var)
        if abs(z) > 3.5:
            sum_cond_sigs.append((b, n, int(counts[n]), T, z))
sum_cond_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(sum_cond_sigs)} (sum_bin, number) signals with |z|>3.5  ({time.time()-t0:.1f}s)")
print(f"  Top 10:")
for b, n, cnt, t_, z in sum_cond_sigs[:10]:
    print(f"    sum_bin={b} → #{n:>2}: P={cnt/t_:.4f}  n={t_}  z={z:+.2f}")
    if abs(z) > 3.5:
        add_signal("sum_cond", f"sb{b}_n{n}", z, f"sum bin {b} → #{n}")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: PAIR → NUMBER at LAG 3 and 5
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4] PAIR → NUMBER at LAG 3 and 5")
print("="*70)
t0 = time.time()
pair_idx = defaultdict(list)
for i in range(N-5):
    nums = all_draws[i][1]
    for a_ in range(20):
        for b_ in range(a_+1, 20):
            pair_idx[(nums[a_], nums[b_])].append(i)
print(f"  Indexed pairs  ({time.time()-t0:.1f}s)")

for LAG in [3, 5]:
    p2n_lag = []
    for (a, b), idx_list in pair_idx.items():
        next_indices = np.array([i + LAG for i in idx_list if i + LAG < N], dtype=np.int32)
        if len(next_indices) < 12000: continue
        next_M = M[next_indices]
        counts = next_M.sum(axis=0)
        for c_ in range(1, 81):
            if c_ in (a, b): continue
            cnt = int(counts[c_])
            exp = len(next_indices) * p
            var = len(next_indices) * p * (1-p)
            z = (cnt - exp) / sqrt(var)
            if abs(z) > 4.0:
                p2n_lag.append(((a,b), c_, cnt, len(next_indices), z))
    p2n_lag.sort(key=lambda x: -abs(x[4]))
    print(f"\n  LAG={LAG}: Found {len(p2n_lag)} signals  ({time.time()-t0:.1f}s)")
    for (a,b), c_, cnt, t_, z in p2n_lag[:8]:
        print(f"    ({a:>2},{b:>2}) →lag{LAG}→ {c_:>2}  P={cnt/t_:.4f}  z={z:+.2f}")
        if abs(z) > 4.0:
            add_signal(f"p2n_lag{LAG}", f"({a},{b})→{c_}", z, f"pair ({a},{b}) → {c_} lag {LAG}")

# ═══════════════════════════════════════════════════════════════════
# TEST 5: DELAY × HOUR fine — combinations not previously found
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[5] DELAY × HOUR — RELAXED threshold |z|>3.0")
print("="*70)
t0 = time.time()
# Recompute delays
delay_arr = np.full((N, 81), 999, dtype=np.int32)
last_seen = np.full(81, -999, dtype=np.int32)
for i in range(N):
    for n in range(1, 81):
        if last_seen[n] >= 0:
            delay_arr[i, n] = i - last_seen[n]
    for n in all_draws[i][1]:
        last_seen[n] = i
print(f"  Delays computed ({time.time()-t0:.1f}s)")

hxd_sigs = []
for h in range(24):
    h_mask = hours == h
    idx_h = np.where(h_mask)[0]
    if len(idx_h) < 5000: continue
    for d in range(1, 15):
        for n in range(1, 81):
            mask_dn = delay_arr[idx_h, n] == d
            total = int(mask_dn.sum())
            if total < 300: continue
            appeared = int(M[idx_h[mask_dn], n].sum())
            p_cond = appeared / total
            z = (p_cond - p) / sqrt(p*(1-p)/total)
            if abs(z) > 3.0:
                hxd_sigs.append((h, d, n, total, p_cond, z))
hxd_sigs.sort(key=lambda x: -abs(x[5]))
print(f"  Found {len(hxd_sigs)} (h, d, n) with |z|>3.0  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for h, d, n, t_, pc, z in hxd_sigs[:15]:
    print(f"    h={h:>2} d={d:>2} #{n:>2}: P={pc:.4f} n={t_} z={z:+.2f}")
    if abs(z) > 3.5:
        add_signal("hxd_v2", f"h{h}d{d}n{n}", z, f"h={h} d={d} #{n}")

# ═══════════════════════════════════════════════════════════════════
# TEST 6: NUMBER TRANSITIONS BY WEEKDAY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] NUMBER TRANSITIONS by WEEKDAY")
print("="*70)
t0 = time.time()
DOW = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
dow_trans = []
for d_ in range(7):
    mask = dows[:-1] == d_
    T = int(mask.sum())
    if T < 5000: continue
    M_curr = M[:-1][mask].astype(np.int32)
    M_next = M[1:][mask].astype(np.int32)
    trans = M_curr.T @ M_next
    count_a = M_curr.sum(axis=0)
    for a in range(1, 81):
        if count_a[a] < 200: continue
        for b in range(1, 81):
            if a == b: continue
            cnt = int(trans[a, b])
            exp = count_a[a] * p
            var = count_a[a] * p * (1-p)
            z = (cnt - exp) / sqrt(var)
            if abs(z) > 4.0:
                dow_trans.append((d_, a, b, cnt, count_a[a], z))
dow_trans.sort(key=lambda x: -abs(x[5]))
print(f"  Found {len(dow_trans)} (dow, a→b) with |z|>4.0  ({time.time()-t0:.1f}s)")
print(f"  Top 10:")
for d_, a, b, cnt, ca, z in dow_trans[:10]:
    print(f"    {DOW[d_]}  {a:>2}→{b:>2}  P={cnt/ca:.4f}  z={z:+.2f}")
    if abs(z) > 4.0:
        add_signal("dow_trans", f"{DOW[d_]}_{a}→{b}", z, f"{DOW[d_]} {a}→{b}")

# ═══════════════════════════════════════════════════════════════════
# TEST 7: WINDOW-OF-5 collective signal
# Sum of last 5 draws indicates next? Specific numbers more likely?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[7] WINDOW-OF-5 COLLECTIVE")
print("="*70)
t0 = time.time()
W = 5
# For each i >= W, count appearances in last 5 draws per number
# Then for each (n, count_in_W), compute P(n at i)
cum = np.zeros((N+1, 81), dtype=np.int32)
cum[1:] = np.cumsum(M, axis=0)
rolling = cum[W:N] - cum[:N-W]  # (N-W, 81), rolling at index i corresponds to draws i..i+W-1 from cum
# Actually for i in [W, N), rolling[i-W] = count in indices i-W..i-1
# So for next-draw prediction: at index i, lookback is rolling[i-W]
# We want P(n at idx | rolling[idx-W, n] = c)
window_5_sigs = []
for c_ in range(0, 6):  # 0..5 appearances in last 5
    mask = rolling[:, 1:] == c_  # (N-W, 80)
    total = int(mask.sum())
    if total < 5000: continue
    appears = M[W:, 1:][mask]
    appeared = int(appears.sum())
    p_c = appeared / total
    z = (p_c - p) / sqrt(p*(1-p)/total)
    print(f"    c={c_} in last 5: P(n appears next)={p_c:.4f} n={total:,} z={z:+.2f}")
    if abs(z) > 2.5:
        add_signal("win5", f"c{c_}", z, f"count in last 5 = {c_}")
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 8: ANTI-TRIPLE → NUMBER (if all 3 absent, P(c))
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[8] ANTI-TRIPLE → NUMBER (sampled)")
print("="*70)
t0 = time.time()
import random as _r
_r.seed(0)
test_triples = []
for _ in range(50):
    a, b, c_ = sorted(_r.sample(range(1, 81), 3))
    test_triples.append((a, b, c_))
anti_t_sigs = []
for (a, b, c_) in test_triples:
    mask = (M[:-1, a] == 0) & (M[:-1, b] == 0) & (M[:-1, c_] == 0)
    if mask.sum() < 50000: continue
    idx = np.where(mask)[0]
    next_M = M[idx + 1]
    counts = next_M.sum(axis=0)
    for n in range(1, 81):
        cnt = int(counts[n])
        exp = len(idx) * p
        var = len(idx) * p * (1-p)
        z = (cnt - exp) / sqrt(var)
        if abs(z) > 4.0:
            anti_t_sigs.append(((a,b,c_), n, cnt, len(idx), z))
anti_t_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(anti_t_sigs)} signals  ({time.time()-t0:.1f}s)")
for trip, n, cnt, t_, z in anti_t_sigs[:8]:
    print(f"    ¬{trip} → #{n:>2}  P={cnt/t_:.4f}  z={z:+.2f}")
    if abs(z) > 4.0:
        add_signal("anti_t2n", f"¬{trip}→{n}", z, f"¬{trip} → #{n}")

# ═══════════════════════════════════════════════════════════════════
# TEST 9: SUM bin → PAIR bias (which pairs cluster in high vs low sums?)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[9] SUM BIN → PAIR FREQUENCY")
print("="*70)
t0 = time.time()
# Use 5 bins for sum
sum_bin5 = np.clip(np.digitize(sums, np.percentile(sums, [20,40,60,80])), 0, 4)
sb_pair_sigs = []
for b in range(5):
    mask = sum_bin5 == b
    T = int(mask.sum())
    if T < 20000: continue
    M_b = M[mask].astype(np.int32)
    pair_b = M_b.T @ M_b
    exp_pair = T * 20*19/(80*79)
    var_pair = exp_pair * (1 - 20*19/(80*79))
    for a in range(1, 81):
        for c_ in range(a+1, 81):
            cnt = int(pair_b[a, c_])
            z = (cnt - exp_pair) / sqrt(var_pair)
            if abs(z) > 4.0:
                sb_pair_sigs.append((b, a, c_, cnt, z))
sb_pair_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(sb_pair_sigs)} (sum_bin, pair) with |z|>4.0  ({time.time()-t0:.1f}s)")
for b, a, c_, cnt, z in sb_pair_sigs[:10]:
    print(f"    bin={b}  ({a:>2},{c_:>2})  count={cnt}  z={z:+.2f}")
    if abs(z) > 4.0:
        add_signal("sb_pair", f"b{b}_({a},{c_})", z, f"sum bin {b} pair ({a},{c_})")

# ═══════════════════════════════════════════════════════════════════
# TEST 10: TRIPLE → NUMBER at LAG 2
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[10] TRIPLE → NUMBER at LAG 2")
print("="*70)
t0 = time.time()
t2n_lag2 = []
for triple, idx_list in top_triples[:300]:  # only top triples
    next2_indices = np.array([i+2 for i in idx_list if i+2 < N], dtype=np.int32)
    if len(next2_indices) < 3000: continue
    next2_M = M[next2_indices]
    counts = next2_M.sum(axis=0)
    for c_ in range(1, 81):
        if c_ in triple: continue
        cnt = int(counts[c_])
        exp = len(next2_indices) * p
        var = len(next2_indices) * p * (1-p)
        z = (cnt - exp) / sqrt(var)
        if abs(z) > 4.5:
            t2n_lag2.append((triple, c_, cnt, len(next2_indices), z))
t2n_lag2.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(t2n_lag2)} signals  ({time.time()-t0:.1f}s)")
for trip, c_, cnt, t_, z in t2n_lag2[:10]:
    print(f"    {trip} →lag2→ {c_:>2}  P={cnt/t_:.4f}  z={z:+.2f}")
    if abs(z) > 4.5:
        add_signal("t2n_lag2", f"{trip}→{c_}", z, f"triple {trip} → {c_} lag 2")

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ — {len(all_signals)} νέα σήματα (7η ομάδα)")
print("="*70)
all_signals.sort(key=lambda x: -abs(x[2]))
print(f"\n  Top 30:")
for i, (cat, name, z, det) in enumerate(all_signals[:30], 1):
    print(f"  {i:>3}.  [{cat:>12}]  {name:>22}  |z|={abs(z):>6.2f}  {'+' if z>0 else '-'}")

print(f"\n  Σύνοψη ανά κατηγορία:")
cats = defaultdict(list)
for cat, name, z, det in all_signals:
    cats[cat].append(abs(z))
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"    {cat:>14}: {len(arr):>3} σήματα, max |z|={max(arr):.2f}")

out = {'signals': [(c, n, float(z), d) for c, n, z, d in all_signals], 'N': N}
with open('/home/user/Game/seventh_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: seventh_signals.json ({len(all_signals)} σήματα)")
print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
