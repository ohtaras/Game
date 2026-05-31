#!/usr/bin/env python3
"""
Στοχευμένο backtest 6+/8 για όλες τις στρατηγικές της εφαρμογής.

Στρατηγικές που δοκιμάζω:
  1. Random baseline (control)
  2. Hot 8 (most appeared last 60 draws)
  3. Cold 8 (least appeared last 60 draws — highest delay)
  4. Distribution-based [3,0,2,1,0,2,0,0,0,0] (app default)
  5. Reverse distribution
  6. Subtractive (exclude all predictions, pick top-delay from remainder)
  7. Hour-bias top 8
  8. Cold + Hour bias
  9. Cold chain (lowest-activity zone)
 10. ML-style: top 8 by (delay × hour_bias × frequency)

Test: για κάθε στρατηγική, πάρε 10K random draw indices i, χρησιμοποίησε
draws[0..i-1] για features, πρόβλεψε 8 αριθμούς, μέτρα overlap με draws[i].

Στόχος: ποια στρατηγική έχει P(6+/8) > 0.254%?
"""
import json, time, numpy as np, random
from pathlib import Path
from collections import Counter
from math import comb

DATA_DIR = Path('/home/user/Game/data/raw')
print("Loading...")
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        all_draws.append((d['id'], sorted(d['n'])))
all_draws.sort(key=lambda x: x[0])
N = len(all_draws)
print(f"  {N} draws  in {time.time()-t0:.1f}s")

# Binary matrix for fast lookup
M = np.zeros((N, 81), dtype=np.int8)
for i, (_, nums) in enumerate(all_draws):
    for n in nums:
        M[i, n] = 1

# Hour-bias from CLAUDE.md (top 10 with z>2.5)
HOUR_BIAS = {
    63: {21: +3.62, 22: +3.62},
    71: {23: +3.39, 0: +3.39},
    59: {12: -3.20, 13: -3.20},
    80: {20: -3.18, 21: -3.18},
    74: {1: +2.92, 2: +2.92},
    45: {14: +2.71, 15: +2.71},
    8:  {17: +2.69, 18: +2.69},
    37: {1: +2.69, 2: +2.69},
    43: {20: -2.67, 21: -2.67},
    13: {15: -2.66, 16: -2.66},
}

def get_hour_bias(n, hour):
    return HOUR_BIAS.get(n, {}).get(hour, 0)

# Compute delays from history M[start:end]
def compute_delays(M, start, end, W=60):
    """For each number 1..80, the delay (draws since last appearance) up to end."""
    delays = np.full(81, W+1, dtype=np.int32)
    for back in range(1, min(end-start, W+1)+1):
        idx = end - back
        if idx < start: break
        for n in range(1, 81):
            if delays[n] > W and M[idx, n]:
                delays[n] = back
    return delays

# ── Strategy implementations ──────────────────────────────────────────
def strat_random(delays, draw_idx, hour):
    return random.sample(range(1, 81), 8)

def strat_hot(delays, draw_idx, hour):
    # Top 8 by appearance count in last 60 (= smallest delays)
    order = sorted(range(1, 81), key=lambda n: delays[n])
    return order[:8]

def strat_cold(delays, draw_idx, hour):
    order = sorted(range(1, 81), key=lambda n: -delays[n])
    return order[:8]

def strat_distribution_default(delays, draw_idx, hour):
    """App's default: [3,0,2,1,0,2,0,0,0,0] — buckets by delay"""
    # Group by delay group: 0-5, 6-11, 12-17, ..., 54+
    cfg = [3, 0, 2, 1, 0, 2, 0, 0, 0, 0]
    picks = []
    for grp_idx, cnt in enumerate(cfg):
        if cnt == 0: continue
        lo, hi = grp_idx*6, grp_idx*6+5
        candidates = [n for n in range(1, 81) if lo <= delays[n] <= hi]
        candidates.sort(key=lambda n: -delays[n])  # within group, prefer higher delay
        picks.extend(candidates[:cnt])
    while len(picks) < 8:  # fill with random unique
        n = random.randint(1, 80)
        if n not in picks: picks.append(n)
    return picks[:8]

def strat_subtractive(delays, draw_idx, hour):
    """Exclude all predictions from main strats, pick top-delay from remainder."""
    excluded = set()
    excluded.update(strat_hot(delays, draw_idx, hour))
    excluded.update(strat_cold(delays, draw_idx, hour))
    excluded.update(strat_distribution_default(delays, draw_idx, hour))
    remainder = [n for n in range(1, 81) if n not in excluded]
    remainder.sort(key=lambda n: (-delays[n], -get_hour_bias(n, hour)))
    if len(remainder) >= 8:
        return remainder[:8]
    return remainder + random.sample([n for n in range(1, 81) if n not in remainder],
                                     8 - len(remainder))

def strat_hour_bias(delays, draw_idx, hour):
    """Top 8 by hour bias score, ties broken by delay."""
    scores = [(n, get_hour_bias(n, hour), delays[n]) for n in range(1, 81)]
    scores.sort(key=lambda x: (-x[1], -x[2]))
    return [n for n, _, _ in scores[:8]]

def strat_cold_plus_hour(delays, draw_idx, hour):
    """Cold + hour bias as tiebreaker."""
    scores = [(n, delays[n], get_hour_bias(n, hour)) for n in range(1, 81)]
    scores.sort(key=lambda x: (-x[1], -x[2]))
    return [n for n, _, _ in scores[:8]]

def strat_cold_chain(delays, draw_idx, hour):
    """Find the column with highest avg delay; pick 8 from there + cold neighbors."""
    col_delays = [(c, sum(delays[c+10*r] for r in range(8))/8) for c in range(1, 11)]
    col_delays.sort(key=lambda x: -x[1])
    best_col = col_delays[0][0]
    nums_in_col = [best_col + 10*r for r in range(8)]
    # Add 0 if we want exactly 8
    return nums_in_col[:8] if len(nums_in_col) >= 8 else nums_in_col + [
        n for n in range(1,81) if n not in nums_in_col][:8-len(nums_in_col)]

# ── Run backtest ──────────────────────────────────────────────────────
print("\n══ Backtest 10K random plays per strategy ══")
random.seed(42)
np.random.seed(42)

# Pick 10K random indices from draws beyond #1000 (need history)
test_indices = random.sample(range(1000, N), 10000)

strategies = [
    ('Random', strat_random),
    ('Hot (top freq)', strat_hot),
    ('Cold (top delay)', strat_cold),
    ('Distribution [3,0,2,1,0,2,0,0,0,0]', strat_distribution_default),
    ('Subtractive', strat_subtractive),
    ('Hour bias', strat_hour_bias),
    ('Cold + Hour', strat_cold_plus_hour),
    ('Cold chain (best column)', strat_cold_chain),
]

# Baseline reference (exact hypergeometric)
def p_hits(k):
    return comb(20, k) * comb(60, 8-k) / comb(80, 8)

baseline = {k: p_hits(k) for k in range(9)}
print(f"\n  Baseline (exact): P(6/8)={baseline[6]*100:.4f}%  P(7/8)={baseline[7]*100:.4f}%  "
      f"P(8/8)={baseline[8]*100:.4f}%  P(6+)={sum(baseline[k] for k in [6,7,8])*100:.4f}%")
print(f"  → Σε 10K plays, αναμένουμε {sum(baseline[k] for k in [6,7,8])*10000:.1f} 6+/8 hits\n")

print(f"  {'Strategy':>40} {'4/8':>6} {'5/8':>6} {'6/8':>5} {'7/8':>5} {'8/8':>5} "
      f"{'6+/8':>6} {'P(6+)%':>8} {'z':>6}")

results = []
for name, fn in strategies:
    t0 = time.time()
    hit_counter = Counter()
    for idx in test_indices:
        # Compute delays from history before idx
        delays = compute_delays(M, max(0, idx-60), idx, W=60)
        # Hour from draw_id and anchor
        # Anchor: draw 1303293 at 23:55 EEST → minutes per draw = 5.28
        from datetime import datetime, timezone, timedelta
        anchor_id, anchor_dt = 1303293, datetime(2026,5,31,23,55,tzinfo=timezone(timedelta(hours=3)))
        draw_dt = anchor_dt + timedelta(minutes=(all_draws[idx][0] - anchor_id) * 5.28)
        hour = draw_dt.hour
        picks = fn(delays, idx, hour)
        actual = set(all_draws[idx][1])
        hits = len(set(picks) & actual)
        hit_counter[hits] += 1

    h6 = hit_counter.get(6, 0)
    h7 = hit_counter.get(7, 0)
    h8 = hit_counter.get(8, 0)
    h6plus = h6 + h7 + h8
    p_6plus = h6plus / 10000 * 100
    expected_6plus = sum(baseline[k] for k in [6,7,8]) * 10000
    z = (h6plus - expected_6plus) / (expected_6plus**0.5)
    flag = ' ★★★' if z > 3 else (' ★' if z > 1.96 else '')

    print(f"  {name:>40} {hit_counter.get(4,0):>6} {hit_counter.get(5,0):>6} "
          f"{h6:>5} {h7:>5} {h8:>5} {h6plus:>6} {p_6plus:>7.3f}% {z:>+6.2f}{flag}")
    results.append((name, h6plus, p_6plus, z, time.time()-t0))

# ── Summary ─────────────────────────────────────────────────────────
print(f"\n══ Σύνοψη ══")
best = max(results, key=lambda x: x[1])
print(f"  Καλύτερη: '{best[0]}' με P(6+/8) = {best[2]:.3f}% (z={best[3]:+.2f})")
print(f"  Baseline:  P(6+/8) = {sum(baseline[k] for k in [6,7,8])*100:.3f}%")

best_z = max(results, key=lambda x: x[3])
print(f"\n  Υψηλότερο z-score: '{best_z[0]}' με z={best_z[3]:+.2f}")
if best_z[3] > 2:
    print(f"  ★ Στατιστικά σημαντικό σήμα!")
else:
    print(f"  Όλες οι στρατηγικές στα όρια του θορύβου.")
