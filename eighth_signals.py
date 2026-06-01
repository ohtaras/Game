#!/usr/bin/env python3
"""
8η ομάδα — δημιουργικά tests (year, lunar, current streak, multi-pair).

Tests:
 1. Year bias per number (2024/2025/2026)
 2. Month-of-year × number deep
 3. Day-of-month bias
 4. Current streak length (count consecutive appearances) → next?
 5. Multi-pair conditioning: BOTH (a,b) AND (c,d) in i → P(e in i+1)
 6. Anti-Markov at lag 2: NOT a in i AND NOT a in i+1 → P(a in i+2)
 7. Specific consecutive triples (k, k+1, k+2 in same draw)
 8. 5-back Markov per number
 9. Cross-pair transitions: pair_i → pair_{i+1} (top combos)
10. Lunar phase × number (from date)
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from math import sqrt, log, comb, sin, cos, pi

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

# Pre-compute times
years = np.zeros(N, dtype=np.int16)
months = np.zeros(N, dtype=np.int8)
days = np.zeros(N, dtype=np.int8)
hours = np.zeros(N, dtype=np.int8)
dows = np.zeros(N, dtype=np.int8)
print("Computing dates...")
for i in range(N):
    dt = draw_time(i)
    years[i] = dt.year
    months[i] = dt.month
    days[i] = dt.day
    hours[i] = dt.hour
    dows[i] = dt.weekday()

# ═══════════════════════════════════════════════════════════════════
# TEST 1: YEAR BIAS per number
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] YEAR BIAS per number")
print("="*70)
t0 = time.time()
year_sigs = []
for y in sorted(set(years.tolist())):
    mask = years == y
    T = int(mask.sum())
    if T < 5000: continue
    counts = M[mask].sum(axis=0)
    for n in range(1, 81):
        exp = T * p
        var = T * p * (1-p)
        z = (counts[n] - exp) / sqrt(var)
        if abs(z) > 3.0:
            year_sigs.append((y, n, int(counts[n]), T, z))
year_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(year_sigs)} (year, number) with |z|>3.0  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for y, n, cnt, T, z in year_sigs[:15]:
    print(f"    {y}  #{n:>2}: {cnt}/{T}  z={z:+.2f}")
    if abs(z) > 3.5:
        add_signal("year", f"{y}_n{n}", z, f"#{n} in year {y}")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: DAY-OF-MONTH bias
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] DAY-OF-MONTH bias per number")
print("="*70)
t0 = time.time()
dom_sigs = []
for d_ in range(1, 32):
    mask = days == d_
    T = int(mask.sum())
    if T < 3000: continue
    counts = M[mask].sum(axis=0)
    for n in range(1, 81):
        exp = T * p
        var = T * p * (1-p)
        z = (counts[n] - exp) / sqrt(var)
        if abs(z) > 3.0:
            dom_sigs.append((d_, n, int(counts[n]), T, z))
dom_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(dom_sigs)} (day-of-month, number) with |z|>3.0  ({time.time()-t0:.1f}s)")
print(f"  Top 10:")
for d_, n, cnt, T, z in dom_sigs[:10]:
    print(f"    day={d_:>2}  #{n:>2}: {cnt}/{T}  z={z:+.2f}")
    if abs(z) > 3.5:
        add_signal("dom", f"d{d_}_n{n}", z, f"#{n} on day {d_}")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: CURRENT STREAK LENGTH → next probability
# At each i, for each n, count current streak (how many of last K draws had n)
# Then check P(n at i | streak length = k)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] CURRENT STREAK LENGTH → next P")
print("="*70)
t0 = time.time()
# Streak: number of consecutive draws (up to and including the previous) that had n
# So streak[i, n] = number of consecutive 1s ending at i-1
streak = np.zeros((N, 81), dtype=np.int8)
for i in range(1, N):
    streak[i] = (M[i-1] == 1).astype(np.int8) * (streak[i-1] + 1)
print(f"  Streak computed  ({time.time()-t0:.1f}s)")

streak_sigs = []
for s in range(0, 8):
    mask = streak[:, 1:] == s  # (N, 80)
    total = int(mask.sum())
    if total < 1000: continue
    appears = M[:, 1:][mask]
    appeared = int(appears.sum())
    p_c = appeared / total
    z = (p_c - p) / sqrt(p*(1-p)/total)
    print(f"    streak={s}: n={total:,}  P(next)={p_c:.4f}  z={z:+.2f}")
    if abs(z) > 2.5:
        add_signal("streak_len", f"s{s}", z, f"streak length = {s}")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: MULTI-PAIR CONDITIONING
# If BOTH (a,b) and (c,d) in i, what's P(e in i+1)?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4] MULTI-PAIR CONDITIONING")
print("="*70)
t0 = time.time()
# Take strongest p2n signals from before, see if pair (a,b) AND pair (c,d) together → strong signal
# Just test some combinations
M32 = M.astype(np.int32)
# Strongest pairs from earlier work
strong_pairs = [(75,80), (50,67), (68,80), (26,79), (15,22), (71,74), (28,38), (41,55)]
mp_sigs = []
for i_p in range(len(strong_pairs)):
    for j_p in range(i_p+1, len(strong_pairs)):
        a, b = strong_pairs[i_p]
        c, d = strong_pairs[j_p]
        if len({a, b, c, d}) < 4: continue
        # Find draws containing both pairs
        mask = (M[:-1, a] & M[:-1, b] & M[:-1, c] & M[:-1, d]) == 1
        T = int(mask.sum())
        if T < 100: continue
        next_M = M[1:][mask]
        counts = next_M.sum(axis=0)
        for e in range(1, 81):
            if e in (a, b, c, d): continue
            cnt = int(counts[e])
            exp = T * p
            var = T * p * (1-p)
            z = (cnt - exp) / sqrt(var)
            if abs(z) > 3.0:
                mp_sigs.append(((a,b), (c,d), e, T, cnt, z))
mp_sigs.sort(key=lambda x: -abs(x[5]))
print(f"  Found {len(mp_sigs)} multi-pair signals  ({time.time()-t0:.1f}s)")
for (ab, cd, e, T, cnt, z) in mp_sigs[:10]:
    print(f"    {ab} ∧ {cd} → {e:>2}  T={T}  z={z:+.2f}")
    if abs(z) > 3.5:
        add_signal("multi_pair", f"{ab}_{cd}→{e}", z, f"{ab}∧{cd} → {e}")

# ═══════════════════════════════════════════════════════════════════
# TEST 5: ANTI-MARKOV at LAG 2
# If NOT a in i AND NOT a in i+1, what's P(a in i+2)?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[5] ANTI-MARKOV at LAG 2")
print("="*70)
t0 = time.time()
am_sigs = []
for n in range(1, 81):
    cond = (M[:-2, n] == 0) & (M[1:-1, n] == 0)
    T = int(cond.sum())
    if T < 1000: continue
    next_appear = int((cond & (M[2:, n] == 1)).sum())
    p_c = next_appear / T
    z = (p_c - p) / sqrt(p*(1-p)/T)
    am_sigs.append((n, p_c, z))
am_sigs.sort(key=lambda x: -abs(x[2]))
print(f"  Top 10  ({time.time()-t0:.1f}s)")
for n, pc, z in am_sigs[:10]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    #{n:>2}: P(n at i+2 | not in i,i+1) = {pc:.4f}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("anti_mark_lag2", f"n{n}", z, f"#{n} anti-mark@2")

# ═══════════════════════════════════════════════════════════════════
# TEST 6: SPECIFIC CONSECUTIVE TRIPLES (k, k+1, k+2 all in draw)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] CONSECUTIVE TRIPLES (k, k+1, k+2) per draw")
print("="*70)
t0 = time.time()
# For each k=1..78, count draws containing all of {k, k+1, k+2}
exp_ct = N * comb(17, 17) * comb(3, 3) * comb(77, 17) / comb(80, 20)  # not quite
# Easier: 20*19*18/(80*79*78) for a specific ordered triple
exp_ct = N * 20*19*18/(80*79*78)
var_ct = exp_ct * (1 - 20*19*18/(80*79*78))
print(f"  Expected per consecutive triple: {exp_ct:.2f}")
ct_results = []
for k in range(1, 79):
    cnt = int(((M[:, k] & M[:, k+1] & M[:, k+2]) == 1).sum())
    z = (cnt - exp_ct) / sqrt(var_ct)
    ct_results.append((k, cnt, z))
ct_results.sort(key=lambda x: -abs(x[2]))
print(f"  Top 10 most anomalous consecutive triples  ({time.time()-t0:.1f}s)")
for k, cnt, z in ct_results[:10]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    ({k:>2},{k+1:>2},{k+2:>2})  count={cnt:>5}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("ct", f"({k},{k+1},{k+2})", z, f"consec ({k},{k+1},{k+2})")

# ═══════════════════════════════════════════════════════════════════
# TEST 7: 5-BACK MARKOV per number
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[7] 5-BACK MARKOV per number")
print("="*70)
t0 = time.time()
res_5back = []
for n in range(1, 81):
    cond = (M[:-5, n] == 1) & (M[1:-4, n] == 1) & (M[2:-3, n] == 1) & (M[3:-2, n] == 1) & (M[4:-1, n] == 1)
    T = int(cond.sum())
    if T < 30: continue
    cont = int((cond & (M[5:, n] == 1)).sum())
    p_c = cont / T
    z = (p_c - p) / sqrt(p*(1-p)/T)
    res_5back.append((n, T, p_c, z))
res_5back.sort(key=lambda x: -abs(x[3]))
print(f"  Top 10  ({time.time()-t0:.1f}s)")
for n, T, pc, z in res_5back[:10]:
    flag = " ✓" if abs(z) > 2 else ""
    print(f"    #{n:>2}: T={T}  P={pc:.4f}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("5back", f"n{n}", z, f"#{n} 5-back")

# ═══════════════════════════════════════════════════════════════════
# TEST 8: TOP PAIR → PAIR transitions (sparse, broader threshold)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[8] TOP PAIR → PAIR transitions (relaxed)")
print("="*70)
t0 = time.time()
pair_total = M32.T @ M32
top_pairs_for_test = []
for a in range(1, 81):
    for b in range(a+1, 81):
        top_pairs_for_test.append((int(pair_total[a,b]), a, b))
top_pairs_for_test.sort(reverse=True)
top20 = [(a,b) for _, a, b in top_pairs_for_test[:20]]
p2p_sigs = []
exp_pp = 0
for (a, b) in top20[:15]:
    mask = (M[:-1, a] == 1) & (M[:-1, b] == 1)
    T = int(mask.sum())
    if T < 10000: continue
    next_M = M[1:][mask].astype(np.int32)
    pair_next = next_M.T @ next_M
    exp_pp = T * 20*19/(80*79)
    var_pp = exp_pp * (1 - 20*19/(80*79))
    for c in range(1, 81):
        for d in range(c+1, 81):
            cnt = int(pair_next[c, d])
            z = (cnt - exp_pp) / sqrt(var_pp)
            if abs(z) > 4.0:
                p2p_sigs.append(((a,b), (c,d), cnt, T, z))
p2p_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(p2p_sigs)} p2p signals with |z|>4  ({time.time()-t0:.1f}s)")
for (ab, cd, cnt, T, z) in p2p_sigs[:10]:
    print(f"    {ab} → {cd}: count={cnt}/{T}  z={z:+.2f}")
    if abs(z) > 4.0:
        add_signal("p2p_v2", f"{ab}→{cd}", z, f"{ab} → {cd}")

# ═══════════════════════════════════════════════════════════════════
# TEST 9: LUNAR PHASE × NUMBER
# Compute lunar age (0-29.5 days) from date
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[9] LUNAR PHASE × NUMBER")
print("="*70)
t0 = time.time()
# Reference new moon: 2000-01-06 18:14 UTC (Julian Day 2451550.26)
# Synodic period: 29.5305882 days
SYNODIC = 29.5305882
REF_NM = datetime(2000, 1, 6, 18, 14, tzinfo=timezone(timedelta(hours=0)))
lunar_phase = np.zeros(N, dtype=np.int8)  # 0..7 (8 phases)
for i in range(N):
    dt = draw_time(i)
    days_since = (dt - REF_NM).total_seconds() / 86400.0
    phase_frac = (days_since % SYNODIC) / SYNODIC
    lunar_phase[i] = int(phase_frac * 8)
print(f"  Lunar phases computed  ({time.time()-t0:.1f}s)")

lunar_sigs = []
for ph in range(8):
    mask = lunar_phase == ph
    T = int(mask.sum())
    if T < 5000: continue
    counts = M[mask].sum(axis=0)
    for n in range(1, 81):
        exp = T * p
        var = T * p * (1-p)
        z = (counts[n] - exp) / sqrt(var)
        if abs(z) > 3.0:
            lunar_sigs.append((ph, n, int(counts[n]), T, z))
lunar_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(lunar_sigs)} (phase, number) with |z|>3.0")
PHASES = ['New','Wax Cres','First Q','Wax Gibb','Full','Wan Gibb','Last Q','Wan Cres']
for ph, n, cnt, T, z in lunar_sigs[:10]:
    print(f"    {PHASES[ph]}  #{n:>2}: {cnt}/{T}  z={z:+.2f}")
    if abs(z) > 3.5:
        add_signal("lunar", f"ph{ph}_n{n}", z, f"#{n} at lunar {PHASES[ph]}")

# ═══════════════════════════════════════════════════════════════════
# TEST 10: COMBINED HOUR + PAIR → NUMBER (sparse but specific)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[10] HOUR + STRONG PAIR → NUMBER")
print("="*70)
t0 = time.time()
# Top 10 pairs from earlier analysis
strong_pairs2 = [(63,64), (53,80), (12,17), (18,43), (31,47), (75,80), (50,67), (68,80), (26,79), (71,74)]
hpn_sigs = []
for (a, b) in strong_pairs2:
    for h in range(24):
        mask = (M[:-1, a] == 1) & (M[:-1, b] == 1) & (hours[:-1] == h)
        T = int(mask.sum())
        if T < 300: continue
        next_M = M[1:][mask]
        counts = next_M.sum(axis=0)
        for c in range(1, 81):
            if c in (a, b): continue
            cnt = int(counts[c])
            exp = T * p
            var = T * p * (1-p)
            z = (cnt - exp) / sqrt(var)
            if abs(z) > 3.5:
                hpn_sigs.append((h, a, b, c, T, cnt, z))
hpn_sigs.sort(key=lambda x: -abs(x[6]))
print(f"  Found {len(hpn_sigs)} (h, pair, n) signals  ({time.time()-t0:.1f}s)")
for h, a, b, c, T, cnt, z in hpn_sigs[:15]:
    print(f"    h={h:>2} ({a:>2},{b:>2}) → {c:>2}: P={cnt/T:.4f} T={T} z={z:+.2f}")
    if abs(z) > 4:
        add_signal("h_p_n", f"h{h}_({a},{b})→{c}", z, f"h={h} ({a},{b}) → {c}")

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ — {len(all_signals)} νέα σήματα (8η ομάδα)")
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
with open('/home/user/Game/eighth_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: eighth_signals.json ({len(all_signals)} σήματα)")
print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
