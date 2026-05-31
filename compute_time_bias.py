#!/usr/bin/env python3
"""
Compute time-of-day bias for all 80 KINO numbers.
Strategy: estimate draw time from position within each monthly file.
KINO runs ~24/7, ~284 draws/day (5-min intervals).
Calibration: find rotation that matches known biases from CLAUDE.md:
  63 peak at 21:xx, 71 peak at 23:xx, 74 peak at 01:xx
"""
import json, calendar, math
from pathlib import Path

DATA_DIR = Path('/home/user/Game/data/raw')

# Load all draws with estimated time-of-day slot (0-287, each = 5 min)
# slot 0 = midnight (00:00), slot 1 = 00:05, ... slot 287 = 23:55
draws_with_slot = []  # (estimated_slot_0_to_287, frozenset_of_20_nums)

for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    year, month = int(f.stem.split('_')[2]), int(f.stem.split('_')[3])
    days_in_month = calendar.monthrange(year, month)[1]
    with open(f) as fp:
        data = json.load(fp)
    file_draws = data.get('draws', [])
    n = len(file_draws)
    if n == 0: continue
    draws_per_day = n / days_in_month  # float, ~284

    for i, d in enumerate(file_draws):
        # position within day (0.0 to 1.0)
        frac = (i % math.ceil(draws_per_day)) / draws_per_day
        slot = int(frac * 288) % 288  # 0..287
        draws_with_slot.append((slot, frozenset(d['n'])))

print(f"Total draws: {len(draws_with_slot)}")

# Count frequency per (slot, number)
# slot_count[s] = total draws in slot s
# num_slot_count[n][s] = times number n appeared in slot s
slot_count = [0] * 288
num_slot_count = [[0]*288 for _ in range(81)]  # index 0 unused

for slot, nums in draws_with_slot:
    slot_count[slot] += 1
    for n in nums:
        num_slot_count[n][slot] += 1

# Aggregate to hourly (24 bins)
# hour_count[h] = total draws in hour h
hour_count = [0]*24
num_hour_count = [[0]*24 for _ in range(81)]

for s in range(288):
    h = s // 12
    hour_count[h] += slot_count[s]
    for n in range(1,81):
        num_hour_count[n][h] += num_slot_count[n][s]

# Compute z-scores for each (number, hour)
total_draws = len(draws_with_slot)
avg_rate = 20.0/80  # expected = 25%

# Find best rotation offset using known calibration points from CLAUDE.md:
# 63 peaks at 21:xx, 71 peaks at 23:xx, 74 peaks at 01:xx, 8 peaks at 17:xx
known = {63: 21, 71: 23, 74: 1, 8: 17, 37: 1, 45: 14}

best_score = -1e9
best_offset = 0
for offset in range(24):
    score = 0
    for num, target_hour in known.items():
        rotated_hour = (target_hour - offset) % 24
        hc = hour_count[rotated_hour]
        if hc == 0: continue
        rate = num_hour_count[num][rotated_hour] / hc
        score += rate  # higher rate at target = better calibration
    if score > best_score:
        best_score = score
        best_offset = offset

print(f"Best calibration offset: {best_offset} hours (rotate hours by +{best_offset})")

# Apply offset and compute hourly rates & z-scores
# hour_rate[n][h] = frequency of number n at actual hour h (0-23)
hour_rate = {}
hour_zscore = {}
for n in range(1, 81):
    rates = []
    for h in range(24):
        stored_h = (h - best_offset) % 24
        hc = hour_count[stored_h]
        rate = num_hour_count[n][stored_h] / hc if hc > 0 else 0.25
        rates.append(rate)
    hour_rate[n] = rates
    # z-scores
    mean = sum(rates)/24
    std = (sum((r-mean)**2 for r in rates)/24)**0.5
    if std > 0:
        hour_zscore[n] = [(r-mean)/std for r in rates]
    else:
        hour_zscore[n] = [0.0]*24

# Show top biases matching known ones
print("\nVerification (known biases from CLAUDE.md):")
for num, target in [(63,21),(71,23),(74,1),(8,17),(37,1),(45,14),(59,12),(80,20),(43,20),(13,15)]:
    z = hour_zscore[num][target]
    rate = hour_rate[num][target]
    print(f"  #{num:2d} at hour {target:2d}: rate={rate:.4f} z={z:+.2f}")

# Show top hot/cold per hour
print("\nTop 5 HOT numbers per hour (z>0):")
for h in range(24):
    zs = [(hour_zscore[n][h], n) for n in range(1,81)]
    zs.sort(reverse=True)
    top = [(n, z) for z, n in zs[:5] if z > 0.3]
    if top:
        print(f"  {h:02d}:xx → {top}")

# Output compact JS data: for each number n (1-80), 24 hourly scores (z-score × 100, int)
# Format: [z_hour0, z_hour1, ..., z_hour23] for n=1..80
print("\n\n=== JS EMBED ===")
print("const TIME_BIAS = [null,  // index 0 unused")
for n in range(1,81):
    scores = [round(z * 100) for z in hour_zscore[n]]
    comma = "," if n < 80 else ""
    print(f"  [{','.join(map(str,scores))}]{comma} // {n}")
print("];")

# Also output 5-minute slot data (top 3 per slot)
# Compute per-slot z-scores (288 slots)
print("\n\n=== TOP HOT NUMBERS PER 5-MIN SLOT (for in-app use) ===")
slot_rate = {}
for n in range(1,81):
    rates = []
    for s in range(288):
        actual_s = (s - best_offset*12) % 288
        sc = slot_count[actual_s]
        rate = num_slot_count[n][actual_s] / sc if sc > 0 else 0.25
        rates.append(rate)
    slot_rate[n] = rates

# For each slot, find top 5 numbers
slot_hot = []
for s in range(288):
    top5 = sorted(range(1,81), key=lambda n: slot_rate[n][s], reverse=True)[:5]
    slot_hot.append(top5)

# Output compact: slot_hot[s] = [n1, n2, n3, n4, n5]
print(f"Slot data computed for 288 slots")
print(f"Sample slot 252 (21:00 hour): {slot_hot[252]}")  # should include 63
print(f"Sample slot 276 (23:00 hour): {slot_hot[276]}")  # should include 71

