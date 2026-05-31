#!/usr/bin/env python3
"""
Μεγάλο backtest 50K plays για τις 3 καλύτερες στρατηγικές από το προηγούμενο test:
  - Hour bias (z=+1.92, 0.350%)
  - Cold (z=+1.33, 0.320%)
  - Distribution default (z=+1.33, 0.320%)
+ Random ως control.

Με 50K plays, αν P(6+) = 0.354%, αναμένουμε 177 hits ± 13 (SD).
Baseline 0.253% → 127 hits ± 11.
Διαφορά 50 hits → z > 4 αν είναι αληθινό.
"""
import json, time, numpy as np, random
from pathlib import Path
from collections import Counter
from math import comb, sqrt
from datetime import datetime, timezone, timedelta

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

M = np.zeros((N, 81), dtype=np.int8)
for i, (_, nums) in enumerate(all_draws):
    for n in nums:
        M[i, n] = 1

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

# Faster delay computation: vectorized
def compute_delays_fast(M, idx, W=60):
    """Delays at position idx, looking back W draws."""
    start = max(0, idx-W)
    delays = np.full(81, W+1, dtype=np.int32)
    # Reverse: for each step back, mark numbers that appear
    for back in range(1, idx-start+1):
        row = M[idx-back]
        # For each n where row[n]==1 and delays[n] not yet set
        not_set = delays > W
        update = (row == 1) & not_set
        delays[update] = back
    return delays

ANCHOR_ID = 1303293
ANCHOR_DT = datetime(2026,5,31,23,55,tzinfo=timezone(timedelta(hours=3)))

def get_hour(idx):
    draw_dt = ANCHOR_DT + timedelta(minutes=(all_draws[idx][0] - ANCHOR_ID) * 5.28)
    return draw_dt.hour

# Strategies
def strat_random(delays, hour):
    return random.sample(range(1, 81), 8)

def strat_cold(delays, hour):
    return sorted(range(1, 81), key=lambda n: -delays[n])[:8]

def strat_hour(delays, hour):
    return sorted(range(1, 81), key=lambda n: (-get_hour_bias(n, hour), -delays[n]))[:8]

def strat_dist(delays, hour):
    cfg = [3, 0, 2, 1, 0, 2, 0, 0, 0, 0]
    picks = []
    for grp_idx, cnt in enumerate(cfg):
        if cnt == 0: continue
        lo, hi = grp_idx*6, grp_idx*6+5
        candidates = [n for n in range(1, 81) if lo <= delays[n] <= hi]
        candidates.sort(key=lambda n: -delays[n])
        picks.extend(candidates[:cnt])
    while len(picks) < 8:
        n = random.randint(1, 80)
        if n not in picks: picks.append(n)
    return picks[:8]

# Run
random.seed(123)
PLAYS = 50000
test_indices = random.sample(range(1000, N), PLAYS)

strategies = [
    ('Random', strat_random),
    ('Cold (top delay)', strat_cold),
    ('Hour bias', strat_hour),
    ('Distribution default', strat_dist),
]

baseline_p = sum(comb(20,k)*comb(60,8-k)/comb(80,8) for k in [6,7,8])
expected_hits = PLAYS * baseline_p
sd_expected = sqrt(PLAYS * baseline_p * (1 - baseline_p))

print(f"\n══ Backtest {PLAYS:,} plays ══")
print(f"  Baseline P(6+/8) = {baseline_p*100:.4f}%")
print(f"  Αναμενόμενα hits: {expected_hits:.1f} ± {sd_expected:.1f}\n")

print(f"  {'Strategy':>22} {'4/8':>5} {'5/8':>5} {'6/8':>5} {'7/8':>4} {'8/8':>3} "
      f"{'6+':>5} {'P(6+)%':>8} {'z':>6}")

for name, fn in strategies:
    t0 = time.time()
    hit_counter = Counter()
    for idx in test_indices:
        delays = compute_delays_fast(M, idx, W=60)
        hour = get_hour(idx)
        picks = fn(delays, hour)
        hits = sum(1 for p in picks if M[idx, p])
        hit_counter[hits] += 1

    h6, h7, h8 = hit_counter.get(6,0), hit_counter.get(7,0), hit_counter.get(8,0)
    h6plus = h6 + h7 + h8
    p6plus = h6plus / PLAYS * 100
    z = (h6plus - expected_hits) / sd_expected
    flag = ' ★★★' if z > 3 else (' ★★' if z > 2 else (' ★' if z > 1.5 else ''))
    print(f"  {name:>22} {hit_counter.get(4,0):>5} {hit_counter.get(5,0):>5} "
          f"{h6:>5} {h7:>4} {h8:>3} {h6plus:>5} {p6plus:>7.4f}% {z:>+6.2f}{flag}"
          f"  ({time.time()-t0:.0f}s)")

print(f"\n══ ΤΕΛΟΣ ══")
