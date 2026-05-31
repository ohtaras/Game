"""
SCRIPT 1: Accurate time-of-day analysis using anchor timestamps.

Anchor: draw id=1303293 is at 2026-05-31 23:55 EEST (UTC+3).
anchor_utc = datetime(2026, 5, 31, 20, 55, 0, tzinfo=timezone.utc)
Minutes per draw = 5.28.

Athens local time:
  Summer (Apr–Sep): UTC+3
  Winter (Oct–Mar): UTC+2
"""

import json
import glob
import numpy as np
from datetime import datetime, timezone, timedelta

# ── Load all draws ──────────────────────────────────────────────────────────
DATA_GLOB = "/home/user/Game/data/raw/kino_raw_*.json"

def load_draws():
    draws = []
    for path in sorted(glob.glob(DATA_GLOB)):
        with open(path) as f:
            obj = json.load(f)
        draws.extend(obj["draws"])
    draws.sort(key=lambda d: d["id"])
    return draws

draws = load_draws()
N = len(draws)
print(f"Loaded {N:,} draws  (ids {draws[0]['id']} … {draws[-1]['id']})")

# ── Anchor ──────────────────────────────────────────────────────────────────
ANCHOR_ID  = 1303293
ANCHOR_UTC = datetime(2026, 5, 31, 20, 55, 0, tzinfo=timezone.utc)
MINS_PER_DRAW = 5.28

def athens_hour(draw_id: int) -> int:
    """Return Athens local hour (0-23) for a given draw id."""
    delta_min = (draw_id - ANCHOR_ID) * MINS_PER_DRAW
    utc_dt = ANCHOR_UTC + timedelta(minutes=delta_min)
    # Summer: Apr-Sep → UTC+3; Winter: Oct-Mar → UTC+2
    if 4 <= utc_dt.month <= 9:
        offset = 3
    else:
        offset = 2
    local_dt = utc_dt + timedelta(hours=offset)
    return local_dt.hour

# ── Build arrays ─────────────────────────────────────────────────────────────
# hours[i] = Athens hour for draw i
hours = np.array([athens_hour(d["id"]) for d in draws], dtype=np.int8)

# matrix M[i, n-1] = 1 if number n appeared in draw i
M = np.zeros((N, 80), dtype=np.int8)
for i, d in enumerate(draws):
    for n in d["n"]:
        M[i, n - 1] = 1

# ── Per-hour statistics ──────────────────────────────────────────────────────
draws_per_hour  = np.bincount(hours, minlength=24)          # how many draws per hour bin
total_hits_hour = np.zeros((24, 80), dtype=np.int64)        # total_hits_hour[h, n] = appearances

for h in range(24):
    mask = (hours == h)
    if mask.any():
        total_hits_hour[h] = M[mask].sum(axis=0)

# ── Z-scores ────────────────────────────────────────────────────────────────
# p = 20/80 = 0.25
p = 20.0 / 80.0
z_scores = np.zeros((24, 80))

for h in range(24):
    nh = draws_per_hour[h]
    if nh < 10:
        continue
    expected = nh * p
    # binomial std dev
    std = np.sqrt(nh * p * (1.0 - p))
    z_scores[h] = (total_hits_hour[h] - expected) / std

# For each number: find the hour with max |z| and report it
max_abs_z = np.max(np.abs(z_scores), axis=0)   # shape (80,)
peak_hour  = np.argmax(np.abs(z_scores), axis=0)

# ── Print top 20 numbers by |z| ─────────────────────────────────────────────
print("\n" + "="*70)
print("TOP 20 NUMBERS  (highest |z-score| across any hour bin)")
print("="*70)
print(f"{'Num':>4}  {'PeakHour':>8}  {'Observed':>9}  {'Expected':>9}  "
      f"{'Rate%':>7}  {'BaseRate%':>10}  {'z':>7}")
print("-"*70)

top20_idx = np.argsort(max_abs_z)[::-1][:20]
for idx in top20_idx:
    num = idx + 1
    h   = peak_hour[idx]
    obs = total_hits_hour[h, idx]
    nh  = draws_per_hour[h]
    exp = nh * p
    rate     = 100.0 * obs / nh if nh else 0.0
    base     = 100.0 * p
    z        = z_scores[h, idx]
    print(f"{num:>4}  {h:>5}:00-{h+1:02d}:00  {obs:>9}  {exp:>9.1f}  "
          f"{rate:>7.2f}  {base:>10.2f}  {z:>+7.3f}")

# ── Per-hour global hit rate ─────────────────────────────────────────────────
print("\n" + "="*70)
print("PER-HOUR GLOBAL HIT RATE  (all numbers combined)")
print("Baseline: 20/80 = 25.00%")
print("="*70)
print(f"{'Hour':>6}  {'Draws':>7}  {'TotalHits':>10}  {'Rate%':>7}  "
      f"{'Expected':>9}  {'z_global':>9}")
print("-"*70)

for h in range(24):
    nh  = draws_per_hour[h]
    if nh == 0:
        print(f"{h:>2}:00    {'—':>7}")
        continue
    total_hits = total_hits_hour[h].sum()
    rate = 100.0 * total_hits / (nh * 80)
    expected_hits = nh * 80 * p
    std_global = np.sqrt(nh * 80 * p * (1 - p))
    z_global = (total_hits - expected_hits) / std_global
    print(f"{h:>2}:00   {nh:>7,}  {total_hits:>10,}  {rate:>7.3f}  "
          f"{expected_hits:>9.1f}  {z_global:>+9.3f}")

# ── Draws per hour distribution ──────────────────────────────────────────────
print("\n" + "="*70)
print("DRAWS PER HOUR BIN  (verify uniform distribution)")
print("="*70)
expected_per_hour = N / 24.0
print(f"Expected per hour (uniform): {expected_per_hour:,.0f}")
print()
for h in range(24):
    bar = "#" * (draws_per_hour[h] // 500)
    print(f"{h:>2}:00  {draws_per_hour[h]:>7,}  {bar}")

print("\nDone.")
