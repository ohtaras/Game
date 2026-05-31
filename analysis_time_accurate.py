"""
SCRIPT 1: Accurate time-of-day analysis using anchor timestamps.

Anchor: draw id=1303293 is at 2026-05-31 23:55 EEST = 2026-05-31 20:55 UTC.
Minutes per draw = 5.28.
Athens DST: UTC+3 Apr-Sep, UTC+2 Oct-Mar.
"""

import json
import glob
import numpy as np
from datetime import datetime, timezone, timedelta

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading draws...")
files = sorted(glob.glob("/home/user/Game/data/raw/kino_raw_*.json"))
draws = []
for f in files:
    with open(f) as fp:
        d = json.load(fp)
    draws.extend(d["draws"])
draws.sort(key=lambda x: x["id"])
N = len(draws)
print(f"  {N:,} draws  |  ID range {draws[0]['id']} – {draws[-1]['id']}")

# ── Anchor & timestamp computation ───────────────────────────────────────────
ANCHOR_ID     = 1303293
ANCHOR_UTC    = datetime(2026, 5, 31, 20, 55, 0, tzinfo=timezone.utc)
MINS_PER_DRAW = 5.28

ids        = np.array([d["id"] for d in draws], dtype=np.int64)
delta_mins = (ids - ANCHOR_ID) * MINS_PER_DRAW          # minutes offset from anchor
anchor_epoch = ANCHOR_UTC.timestamp()                    # unix seconds
utc_epochs   = anchor_epoch + delta_mins * 60.0          # unix seconds per draw

# Vectorised Athens hour:
# Convert epoch → UTC datetime, determine DST offset, compute local hour
utc_dts    = [datetime.fromtimestamp(float(e), tz=timezone.utc) for e in utc_epochs]
months     = np.array([dt.month for dt in utc_dts], dtype=np.int8)
utc_hours  = np.array([dt.hour + dt.minute / 60.0 for dt in utc_dts])
offset_hrs = np.where((months >= 4) & (months <= 9), 3, 2)
local_hrs  = (utc_hours + offset_hrs) % 24.0
hour_bin   = local_hrs.astype(int)   # 0–23

# ── Build number matrix ───────────────────────────────────────────────────────
print("Building M matrix...")
M = np.zeros((N, 80), dtype=np.int8)
for i, d in enumerate(draws):
    for n in d["n"]:
        M[i, n - 1] = 1
assert np.all(M.sum(axis=1) == 20), "Row sums sanity check failed"

# ── Per-hour, per-number counts ───────────────────────────────────────────────
draws_per_hour = np.bincount(hour_bin, minlength=24).astype(np.int64)  # (24,)
count_hn = np.zeros((24, 80), dtype=np.int64)                           # (24, 80)
for h in range(24):
    mask = (hour_bin == h)
    if mask.any():
        count_hn[h] = M[mask].sum(axis=0)

# ── Z-scores (Binomial approximation) ────────────────────────────────────────
p = 20.0 / 80.0   # 0.25
expected_hn = np.outer(draws_per_hour, np.full(80, p))          # (24, 80)
std_hn      = np.sqrt(np.outer(draws_per_hour, [p * (1 - p)]))  # (24, 80)
std_hn      = np.maximum(std_hn, 1e-12)
z_hn        = (count_hn - expected_hn) / std_hn                 # (24, 80)

# For hours with very few draws set z to 0 (unreliable)
for h in range(24):
    if draws_per_hour[h] < 10:
        z_hn[h, :] = 0.0

# ── Per-number peak-hour stats ────────────────────────────────────────────────
max_abs_z  = np.max(np.abs(z_hn), axis=0)           # (80,)
peak_hour  = np.argmax(np.abs(z_hn), axis=0)         # (80,)
peak_rate  = np.array([
    count_hn[peak_hour[n], n] / max(draws_per_hour[peak_hour[n]], 1)
    for n in range(80)
])

top20_idx = np.argsort(max_abs_z)[::-1][:20]

print("\n" + "=" * 74)
print("TOP 20 NUMBERS BY MAX |Z-SCORE| ACROSS ALL HOUR BINS")
print(f"  Baseline appearance rate: {p*100:.2f}%  (20 from 80 per draw)")
print("=" * 74)
print(f"{'Num':>4}  {'PeakHr':>7}  {'Observed':>9}  {'Expected':>9}  "
      f"{'Rate%':>7}  {'BaseRate%':>10}  {'z':>8}  Dir")
print("-" * 74)
for n in top20_idx:
    h        = peak_hour[n]
    obs      = count_hn[h, n]
    nh       = draws_per_hour[h]
    exp      = nh * p
    rate_pct = peak_rate[n] * 100
    z        = z_hn[h, n]
    direction = "HIGH" if z > 0 else "LOW "
    print(f"{n+1:>4}  {h:>4}h–{(h+1)%24:02d}h  {obs:>9,}  {exp:>9.1f}  "
          f"{rate_pct:>7.3f}%  {p*100:>10.3f}%  {z:>+8.3f}  {direction}")

# ── Per-hour global hit rate (all 80 numbers combined) ────────────────────────
print("\n" + "=" * 74)
print("PER-HOUR GLOBAL HIT RATE  (all 80 numbers combined)")
print(f"  Expected baseline: {p*100:.3f}%")
print("=" * 74)
print(f"{'Hour':>6}  {'Draws':>7}  {'TotalHits':>10}  {'HitRate%':>9}  "
      f"{'Expected':>10}  {'z_global':>9}")
print("-" * 74)
for h in range(24):
    nh = draws_per_hour[h]
    if nh == 0:
        print(f"{h:>2}:00   {'—':>7}")
        continue
    total_hits    = int(count_hn[h].sum())
    expected_hits = nh * 80 * p
    hit_rate      = total_hits / (nh * 80) * 100
    # global z: treat all nh*80 as independent Bernoulli(p)
    std_global    = np.sqrt(nh * 80 * p * (1 - p))
    z_global      = (total_hits - expected_hits) / std_global
    print(f"{h:>2}:00  {nh:>7,}  {total_hits:>10,}  {hit_rate:>9.4f}%  "
          f"{expected_hits:>10.1f}  {z_global:>+9.3f}")

# ── Draws per hour (uniformity check) ────────────────────────────────────────
print("\n" + "=" * 74)
print("DRAWS PER HOUR BIN  (uniformity check; expected ~uniform)")
print("=" * 74)
exp_per_h = N / 24.0
print(f"  Expected per hour (perfectly uniform): {exp_per_h:,.0f}\n")
print(f"{'Hour':>6}  {'Draws':>7}  {'Pct%':>6}  {'Dev':>8}  Histogram")
print("-" * 74)
for h in range(24):
    nh  = draws_per_hour[h]
    pct = nh / N * 100
    dev = nh - exp_per_h
    bar = "#" * int(nh // 400)
    print(f"{h:>2}:00  {nh:>7,}  {pct:>6.2f}%  {dev:>+8.0f}  {bar}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("SUMMARY")
print("=" * 74)
print(f"  Total draws  : {N:,}")
print(f"  Anchor draw  : id={ANCHOR_ID} → {ANCHOR_UTC.isoformat()} UTC")
print(f"  Mins/draw    : {MINS_PER_DRAW}")
print(f"  Max |z| seen : {max_abs_z.max():.3f}  "
      f"(number {top20_idx[0]+1}, hour {peak_hour[top20_idx[0]]})")

Z_THRESH = 3.0
strong = [(n + 1, int(peak_hour[n]), float(z_hn[peak_hour[n], n]))
          for n in range(80) if max_abs_z[n] >= Z_THRESH]
strong.sort(key=lambda x: -abs(x[2]))
print(f"  Numbers with |z| >= {Z_THRESH}: {len(strong)}")
for num, h, z in strong:
    direction = "HIGH" if z > 0 else "LOW"
    print(f"    Number {num:>2}  hour {h:>2}:xx  z={z:+.3f}  ({direction})")

print("\nDone.")
