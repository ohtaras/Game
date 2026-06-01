#!/usr/bin/env python3
"""
ΣΥΝΟΠΤΙΚΟ ΤΕΣΤ ΟΛΩΝ ΤΩΝ ΣΗΜΑΤΩΝ — Prior: ΔΕΝ είναι τυχαίο.

Τρέχουμε όλα τα tests που έχουμε σκεφτεί, μαζεύουμε z-scores, και
βγάζουμε κατάταξη "πιο μη-τυχαίο → λιγότερο μη-τυχαίο".

Tests:
 1. Hour bias (24h × 80 αριθμοί) — γνωστό σήμα
 2. Day-of-week bias (7 × 80)
 3. Month bias (12 × 80)
 4. Pair frequency (3160 ζευγάρια)
 5. Triplet frequency (top 100)
 6. Markov n→n (memory of single number)
 7. Gap-from-mean per number
 8. Column/Row bias (8 rows × 10 cols)
 9. Sum mean/std
10. Consecutive overlap mean
11. Max streak per number
12. Bit-stream autocorr (lag 80, 160, 800)
13. Compression vs matched baseline
14. Chain top shape frequency

Output: λίστα σημάτων sorted by |z|, και per-number score για το κουμπί.
"""
import json, time, gzip, bz2, lzma, random, pickle
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from math import sqrt, comb, log

DATA_DIR = Path('/home/user/Game/data/raw')

print("="*70)
print("LOADING")
print("="*70)
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

# Anchor & time helpers
ANCHOR_ID = 1303293
ANCHOR_DT = datetime(2026,5,31,23,55,tzinfo=timezone(timedelta(hours=3)))
def draw_time(idx):
    return ANCHOR_DT + timedelta(minutes=(all_draws[idx][0] - ANCHOR_ID) * 5.28)

# Binary matrix [N, 81] — column 0 unused
M = np.zeros((N, 81), dtype=np.int8)
for i, (_, nums) in enumerate(all_draws):
    for n in nums:
        M[i, n] = 1

p = 20/80  # baseline per number per draw

# Pre-compute hour/dow/month for each draw
print("Computing draw times...")
t0 = time.time()
hours = np.zeros(N, dtype=np.int8)
dows = np.zeros(N, dtype=np.int8)
months = np.zeros(N, dtype=np.int8)
for i in range(N):
    dt = draw_time(i)
    hours[i] = dt.hour
    dows[i] = dt.weekday()
    months[i] = dt.month - 1
print(f"  ({time.time()-t0:.1f}s)")

# Signal collector
all_signals = []  # (category, name, z_score, details)
per_number_score = np.zeros(81, dtype=np.float64)  # cumulative bias score per number

def add_signal(cat, name, z, det):
    all_signals.append((cat, name, z, det))

def boost(n, delta):
    """Boost per-number score (signed)."""
    if 1 <= n <= 80:
        per_number_score[n] += delta

# ═══════════════════════════════════════════════════════════════════
# TEST 1: HOUR BIAS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] HOUR BIAS (24h × 80 numbers)")
print("="*70)
t0 = time.time()
hour_counts = np.zeros((24, 81), dtype=np.int32)
hour_totals = np.zeros(24, dtype=np.int32)
for h in range(24):
    mask = hours == h
    hour_totals[h] = mask.sum()
    hour_counts[h] = M[mask].sum(axis=0)

top_hour = []
for h in range(24):
    T = hour_totals[h]
    if T == 0: continue
    exp = T * p
    var = T * p * (1-p)
    for n in range(1, 81):
        z = (hour_counts[h, n] - exp) / sqrt(var)
        if abs(z) > 2.5:
            top_hour.append((n, h, z))
top_hour.sort(key=lambda x: -abs(x[2]))
print(f"  Found {len(top_hour)} (number, hour) pairs with |z|>2.5  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for n, h, z in top_hour[:15]:
    print(f"    #{n:>2}  h={h:02d}:00  z={z:+.2f}")
    add_signal("hour", f"n{n}@h{h}", z, f"#{n} at {h:02d}:00")
    boost(n, z * 0.05)  # weight 5% per |z|

# ═══════════════════════════════════════════════════════════════════
# TEST 2: DAY-OF-WEEK BIAS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] DAY-OF-WEEK BIAS (7 days × 80 numbers)")
print("="*70)
t0 = time.time()
DOW_NAMES = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
dow_counts = np.zeros((7, 81), dtype=np.int32)
dow_totals = np.zeros(7, dtype=np.int32)
for d in range(7):
    mask = dows == d
    dow_totals[d] = mask.sum()
    dow_counts[d] = M[mask].sum(axis=0)

top_dow = []
for d in range(7):
    T = dow_totals[d]
    if T == 0: continue
    exp = T * p
    var = T * p * (1-p)
    for n in range(1, 81):
        z = (dow_counts[d, n] - exp) / sqrt(var)
        if abs(z) > 2.5:
            top_dow.append((n, d, z))
top_dow.sort(key=lambda x: -abs(x[2]))
print(f"  Found {len(top_dow)} (number, dow) pairs with |z|>2.5  ({time.time()-t0:.1f}s)")
print(f"  Top 10:")
for n, d, z in top_dow[:10]:
    print(f"    #{n:>2}  {DOW_NAMES[d]}  z={z:+.2f}")
    add_signal("dow", f"n{n}@{DOW_NAMES[d]}", z, f"#{n} on {DOW_NAMES[d]}")
    boost(n, z * 0.03)

# ═══════════════════════════════════════════════════════════════════
# TEST 3: MONTH BIAS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] MONTH BIAS (12 months × 80 numbers)")
print("="*70)
t0 = time.time()
month_counts = np.zeros((12, 81), dtype=np.int32)
month_totals = np.zeros(12, dtype=np.int32)
for m_ in range(12):
    mask = months == m_
    month_totals[m_] = mask.sum()
    month_counts[m_] = M[mask].sum(axis=0)

top_month = []
for m_ in range(12):
    T = month_totals[m_]
    if T == 0: continue
    exp = T * p
    var = T * p * (1-p)
    for n in range(1, 81):
        z = (month_counts[m_, n] - exp) / sqrt(var)
        if abs(z) > 2.5:
            top_month.append((n, m_, z))
top_month.sort(key=lambda x: -abs(x[2]))
print(f"  Found {len(top_month)} (number, month) pairs with |z|>2.5  ({time.time()-t0:.1f}s)")
print(f"  Top 10:")
for n, m_, z in top_month[:10]:
    print(f"    #{n:>2}  month={m_+1:>2}  z={z:+.2f}")
    add_signal("month", f"n{n}@m{m_+1}", z, f"#{n} in month {m_+1}")
    boost(n, z * 0.02)

# ═══════════════════════════════════════════════════════════════════
# TEST 4: PAIR FREQUENCY (numpy matrix multiplication)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4] PAIR FREQUENCY (3160 unique pairs)")
print("="*70)
t0 = time.time()
M32 = M.astype(np.int32)
pair_count = M32.T @ M32  # 81x81

# Expected pair frequency: N * 20*19/(80*79)
exp_pair = N * 20 * 19 / (80 * 79)
var_pair_approx = exp_pair * (1 - 20*19/(80*79))

print(f"  Expected per pair: {exp_pair:.1f}")
print(f"  Std approx: {sqrt(var_pair_approx):.1f}")

# Find top |z| pairs (only off-diagonal, i<j)
pair_z = []
for i in range(1, 81):
    for j in range(i+1, 81):
        c = pair_count[i, j]
        z = (c - exp_pair) / sqrt(var_pair_approx)
        if abs(z) > 3.0:
            pair_z.append((i, j, c, z))
pair_z.sort(key=lambda x: -abs(x[3]))
print(f"  Pairs with |z|>3: {len(pair_z)}  ({time.time()-t0:.1f}s)")
print(f"  Top 15:")
for i, j, c, z in pair_z[:15]:
    print(f"    ({i:>2},{j:>2})  count={c:>6}  z={z:+.2f}")
    add_signal("pair", f"({i},{j})", z, f"pair ({i},{j})")
    boost(i, z * 0.02)
    boost(j, z * 0.02)

# ═══════════════════════════════════════════════════════════════════
# TEST 5: MARKOV n→n (memory of single number)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[5] MARKOV n→n (consecutive memory per number)")
print("="*70)
t0 = time.time()
markov = []
for n in range(1, 81):
    in_prev = M[:-1, n] == 1
    in_next = M[1:, n] == 1
    cnt_prev = int(in_prev.sum())
    cnt_both = int((in_prev & in_next).sum())
    if cnt_prev > 0:
        p_cond = cnt_both / cnt_prev
        z = (p_cond - p) / sqrt(p*(1-p)/cnt_prev)
        markov.append((n, p_cond, z))
markov.sort(key=lambda x: -abs(x[2]))
print(f"  Top 10 |P(n→n) - 0.25|  ({time.time()-t0:.1f}s)")
for n, pc, z in markov[:10]:
    flag = " ★" if abs(z) > 3 else ""
    print(f"    #{n:>2}: P(n_t+1 | n_t) = {pc:.4f}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("markov", f"n{n}", z, f"#{n} consecutive")
        boost(n, z * 0.04)

# ═══════════════════════════════════════════════════════════════════
# TEST 6: GAP FROM EXPECTED MEAN per number
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] GAP MEAN per number (expected = 4.0)")
print("="*70)
t0 = time.time()
expected_mean_gap = 80/20
gap_results = []
for n in range(1, 81):
    appearances = np.where(M[:, n] == 1)[0]
    if len(appearances) > 100:
        gaps = np.diff(appearances)
        m_ = gaps.mean()
        s_ = gaps.std()
        z = (m_ - expected_mean_gap) / (s_ / sqrt(len(gaps)))
        gap_results.append((n, m_, z))
gap_results.sort(key=lambda x: -abs(x[2]))
print(f"  Top 10 most anomalous mean-gap  ({time.time()-t0:.1f}s)")
for n, m_, z in gap_results[:10]:
    print(f"    #{n:>2}: mean gap = {m_:.4f}  z={z:+.2f}")
    if abs(z) > 2.5:
        add_signal("gap", f"n{n}", z, f"#{n} gap")
        boost(n, -z * 0.03)  # smaller gap = more frequent = positive boost

# ═══════════════════════════════════════════════════════════════════
# TEST 7: ROW/COLUMN/QUADRANT BIAS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[7] GRID ZONE BIAS")
print("="*70)
t0 = time.time()
# Total appearances per number across all draws
freq = M.sum(axis=0)  # [81]
total_picks = N * 20

# Column bias
col_count = np.zeros(10, dtype=np.int64)
row_count = np.zeros(8, dtype=np.int64)
for n in range(1, 81):
    col_count[(n-1) % 10] += freq[n]
    row_count[(n-1) // 10] += freq[n]

exp_col = total_picks * 8/80
exp_row = total_picks * 10/80
print(f"  Expected per col: {exp_col:,.0f}  per row: {exp_row:,.0f}")
print(f"  Columns:")
for c in range(10):
    var_c = total_picks * (8/80) * (72/80)
    z = (col_count[c] - exp_col) / sqrt(var_c)
    flag = " ★" if abs(z) > 3 else ""
    print(f"    Col {c+1:>2}: {col_count[c]:>10,}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("col", f"col{c+1}", z, f"column {c+1}")
        for n in range(c+1, 81, 10):
            boost(n, z * 0.02)

print(f"  Rows:")
for r in range(8):
    var_r = total_picks * (10/80) * (70/80)
    z = (row_count[r] - exp_row) / sqrt(var_r)
    flag = " ★" if abs(z) > 3 else ""
    print(f"    Row {r+1}: {row_count[r]:>10,}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("row", f"row{r+1}", z, f"row {r+1}")
        for n in range(r*10+1, r*10+11):
            boost(n, z * 0.02)

# ═══════════════════════════════════════════════════════════════════
# TEST 8: DRAW SUM
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[8] DRAW SUM")
print("="*70)
sums = np.array([sum(d) for _, d in all_draws])
exp_sum_mean = 810
# Empirical std from Monte Carlo (matched baseline)
import random as _r
_r.seed(0)
mc_sums = [sum(_r.sample(range(1,81), 20)) for _ in range(10000)]
exp_sum_std = float(np.std(mc_sums))
z_mean = (sums.mean() - exp_sum_mean) / (exp_sum_std/sqrt(N))
print(f"  Mean: {sums.mean():.3f} (exp {exp_sum_mean}, σ={exp_sum_std:.2f}, z={z_mean:+.2f})")
print(f"  Std:  {sums.std():.3f} (exp {exp_sum_std:.2f})")
if abs(z_mean) > 2:
    add_signal("sum", "mean", z_mean, "draw sum mean")

# ═══════════════════════════════════════════════════════════════════
# TEST 9: CONSECUTIVE OVERLAP
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[9] CONSECUTIVE OVERLAP")
print("="*70)
t0 = time.time()
overlaps = np.zeros(N-1, dtype=np.int8)
for i in range(N-1):
    overlaps[i] = int(np.dot(M[i].astype(np.int32), M[i+1].astype(np.int32)))
exp_ov = 5.0
exp_ov_std = sqrt(20 * (60/80) * (20/79) * (60/79))
z_ov = (overlaps.mean() - exp_ov) / (exp_ov_std/sqrt(N-1))
print(f"  Mean overlap: {overlaps.mean():.4f}  (exp 5.000, z={z_ov:+.2f})  ({time.time()-t0:.1f}s)")
if abs(z_ov) > 2:
    add_signal("overlap", "mean", z_ov, "consec overlap")

# ═══════════════════════════════════════════════════════════════════
# TEST 10: MAX STREAK per number
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[10] MAX CONSECUTIVE APPEARANCES per number")
print("="*70)
t0 = time.time()
streaks = []
for n in range(1, 81):
    arr = M[:, n]
    cur = 0; mx = 0
    for b in arr:
        if b: cur += 1; mx = max(mx, cur)
        else: cur = 0
    streaks.append((n, mx))
streaks.sort(key=lambda x: -x[1])
print(f"  Top 10  ({time.time()-t0:.1f}s)")
for n, mx in streaks[:10]:
    # P(streak ≥ mx) ≈ N * 0.25^mx
    p_extreme = N * (0.25**mx)
    z_approx = -log(max(p_extreme, 1e-300)) / log(10)  # -log10(p)
    print(f"    #{n:>2}: max streak = {mx}  -log10(p≈{p_extreme:.2e})  z≈{z_approx:.1f}")
    if p_extreme < 0.01:
        add_signal("streak", f"n{n}", z_approx, f"#{n} streak={mx}")
        boost(n, z_approx * 0.01)

# ═══════════════════════════════════════════════════════════════════
# TEST 11: BIT-STREAM AUTOCORR (lag 80 critical)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[11] BIT-STREAM AUTOCORR (lag 80, 160, 800)")
print("="*70)
t0 = time.time()
bits = np.zeros(N * 80, dtype=np.uint8)
for i in range(N):
    for n in all_draws[i][1]:
        bits[i*80 + n-1] = 1
n_bits = len(bits)
p_bit = bits.mean()
centered = bits.astype(np.float64) - p_bit
def acf_at(lag):
    return float(np.dot(centered[:-lag], centered[lag:]) / n_bits) / (p_bit*(1-p_bit))
a80 = acf_at(80); z80 = a80 * sqrt(n_bits)
a160 = acf_at(160); z160 = a160 * sqrt(n_bits)
a800 = acf_at(800); z800 = a800 * sqrt(n_bits)
print(f"  Lag  80: acf={a80:+.6f}  z={z80:+.2f}  ({time.time()-t0:.1f}s)")
print(f"  Lag 160: acf={a160:+.6f}  z={z160:+.2f}")
print(f"  Lag 800: acf={a800:+.6f}  z={z800:+.2f}")
for lag, z in [(80, z80), (160, z160), (800, z800)]:
    if abs(z) > 2.5:
        add_signal("autocorr", f"lag{lag}", z, f"bit autocorr lag {lag}")

# ═══════════════════════════════════════════════════════════════════
# TEST 12: COMPRESSION (vs matched baseline)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[12] COMPRESSION (matched baseline)")
print("="*70)
t0 = time.time()
def to_bytes(draws):
    b = bytearray()
    for d in draws: b.extend(d)
    return bytes(b)
real_bytes = to_bytes([d for _, d in all_draws])
_r.seed(42)
fake_draws = [sorted(_r.sample(range(1,81), 20)) for _ in range(N)]
fake_bytes = to_bytes(fake_draws)

for name, fn in [('gzip', lambda d: gzip.compress(d, 6)),
                 ('bz2',  lambda d: bz2.compress(d, 9))]:
    cr = len(fn(real_bytes))
    cf = len(fn(fake_bytes))
    diff = cr - cf
    diff_pct = diff / cf * 100
    z_approx = diff_pct  # rough proxy
    print(f"  {name}: KINO={cr:,}  Fake={cf:,}  diff={diff:+,} ({diff_pct:+.3f}%)")
    if abs(diff) > 5000:
        add_signal("compress", name, z_approx, f"{name} vs baseline")
print(f"  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ — {len(all_signals)} σήματα")
print("="*70)
all_signals.sort(key=lambda x: -abs(x[2]))
print(f"\n  Top 30 (sorted by |z|):")
print(f"  {'#':>3}  {'cat':>10}  {'name':>15}  {'|z|':>6}  {'sign':>5}  {'det':>20}")
for i, (cat, name, z, det) in enumerate(all_signals[:30], 1):
    print(f"  {i:>3}.  {cat:>10}  {name:>15}  {abs(z):>6.2f}  {'+' if z>0 else '-':>5}  {det:>20}")

# Per-category summary
print(f"\n  Σύνοψη ανά κατηγορία:")
cats = defaultdict(list)
for cat, name, z, det in all_signals:
    cats[cat].append(abs(z))
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"    {cat:>10}: {len(arr):>3} σήματα, max |z|={max(arr):.2f}, mean={np.mean(arr):.2f}")

# Per-number top scores
print(f"\n  Top 20 αριθμοί με μεγαλύτερο cumulative score:")
top_numbers = sorted(range(1, 81), key=lambda n: -per_number_score[n])
for i, n in enumerate(top_numbers[:20], 1):
    print(f"  {i:>3}.  #{n:>2}  score={per_number_score[n]:+.3f}")

print(f"\n  Bottom 10 αριθμοί (πιο αρνητικά μη-τυχαίοι):")
for i, n in enumerate(top_numbers[-10:], 1):
    print(f"  {i:>3}.  #{n:>2}  score={per_number_score[n]:+.3f}")

# Save for later
out = {
    'signals': [(c, n, float(z), d) for c, n, z, d in all_signals],
    'per_number_score': {int(n): float(per_number_score[n]) for n in range(1, 81)},
    'N': N,
}
with open('/home/user/Game/all_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: all_signals.json ({len(all_signals)} σήματα)")

print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
